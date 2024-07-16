import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import os
from os.path import join
from tqdm import tqdm
import time
from datetime import datetime
from collections import defaultdict
import scipy.io as scio
import imageio
from einops import rearrange
import matplotlib.pyplot as plt
from model import Build_GFNet
from load_dataset import TestSetLoader
from utils import *


#########################################################################################################
parser = argparse.ArgumentParser(description="Ghost-free high dynamic range light field imaging -- test mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ang_res", type=int, default=7, help="Angular resolution of light field")
parser.add_argument("--crf_gamma", type=float, default=1/0.7, help="Gamma value of camera response function")
parser.add_argument("--u_law", type=float, default=5000.0, help="u value of dynamic range compressor")
parser.add_argument("--model_path", type=str, default="checkpoints", help="Checkpoints path")
parser.add_argument("--testset_path", type=str, default="Dataset/TestData_7x7", help="Test data path")
parser.add_argument("--output_path", type=str, default="results/", help="Output path")
parser.add_argument("--crop", type=int, default=1, help="Crop the LF image into LF patches when out of memory")
parser.add_argument("--patch_size", type=int, default=96, help="Cropped LF patch size, i.e., spatial resolution")
parser.add_argument("--mini_batch", type=int, default=4, help="Mini batch during testing (crop)")

cfg = parser.parse_args()
print(cfg)

#######################################################################################
torch.cuda.set_device(cfg.device)

################################################################################################
def test(opt, test_loader):

    print('==>testing')
    # Pretrained model path
    pretrained_model_path = opt.model_path + '/Trained_model.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path, map_location='cuda:0')
    ckp_dict = checkpoints['model']

    ###########################################################################################
    # Build model
    print("Building GFNet")
    model_test = Build_GFNet(opt).to(opt.device)

    print('loaded model from ' + pretrained_model_path)
    model_test_dict = model_test.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_test_dict}
    model_test_dict.update(ckp_dict_refine)
    model_test.load_state_dict(model_test_dict)

    # Model parallel test
    # if torch.cuda.device_count() > 1:
    #     model_test = nn.DataParallel(model_test)

    # output folder
    mk_dir(opt.output_path)

    #######################################################################################
    # Test
    model_test.eval()
    with torch.no_grad():

        ave_linear_psnr = 0.
        ave_linear_ssim = 0.
        ave_log_psnr = 0.
        ave_log_ssim = 0.

        for idx_iter, (multi_data, expo_data, label_hdr) in tqdm(enumerate(test_loader), total=len(test_loader)):

            start_time = datetime.now()

            ###############################  Input data  ###############################
            # data: [b,18,ah,aw,h,w], [b,3,1];  label: [b,3,ah,aw,h,w]
            multi_data, expo_data, label_hdr = \
                Variable(multi_data).to(opt.device), Variable(expo_data).to(opt.device), Variable(label_hdr).to(opt.device)

            # mask: [b,1,ah,aw,h,w]  mask region is 1
            hl_mask = get_highlight_mask(in_data=multi_data[:, 3:6, :, :, :, :], thr=0.5)    # middle-exposure
            ld_mask = get_lowdark_mask(in_data=multi_data[:, 3:6, :, :, :, :], thr=0.5)

            sai_h = multi_data.shape[-2]
            sai_w = multi_data.shape[-1]

            ###########################  Forward inference  ###########################
            if (opt.crop == 1):
                multi_sub_lfs = LFdivide(lf=multi_data[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
                hl_sub_masks = LFdivide(lf=hl_mask[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
                ld_sub_masks = LFdivide(lf=ld_mask[0], patch_size=opt.patch_size, stride=opt.patch_size//2)

                n1, n2, u, v, c, h, w = multi_sub_lfs.shape
                multi_sub_lfs = rearrange(multi_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')
                hl_sub_masks = rearrange(hl_sub_masks, 'n1 n2 u v c h w -> (n1 n2) c u v h w')
                ld_sub_masks = rearrange(ld_sub_masks, 'n1 n2 u v c h w -> (n1 n2) c u v h w')

                test_hdr1 = []
                test_hdr2 = []
                test_hdr3 = []
                num_infer = (n1 * n2) // opt.mini_batch
                for idx_infer in range(num_infer):
                    multi_lf_patch = multi_sub_lfs[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    hl_mask_patch = hl_sub_masks[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    ld_mask_patch = ld_sub_masks[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    test_hdr_patch1, test_hdr_patch2, test_hdr_patch3 = model_test(multi_lf_patch, expo_data, hl_mask_patch, ld_mask_patch)
                    test_hdr1.append(test_hdr_patch1)
                    test_hdr2.append(test_hdr_patch2)
                    test_hdr3.append(test_hdr_patch3)
                if (n1 * n2) % opt.mini_batch:
                    multi_lf_patch = multi_sub_lfs[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                    hl_mask_patch = hl_sub_masks[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                    ld_mask_patch = ld_sub_masks[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                    test_hdr_patch1, test_hdr_patch2, test_hdr_patch3 = model_test(multi_lf_patch, expo_data, hl_mask_patch, ld_mask_patch)
                    test_hdr1.append(test_hdr_patch1)
                    test_hdr2.append(test_hdr_patch2)
                    test_hdr3.append(test_hdr_patch3)

                test_hdr1 = torch.cat(test_hdr1, dim=0)
                test_hdr2 = torch.cat(test_hdr2, dim=0)
                test_hdr3 = torch.cat(test_hdr3, dim=0)
                test_hdr1 = rearrange(test_hdr1, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                test_hdr2 = rearrange(test_hdr2, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                test_hdr3 = rearrange(test_hdr3, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                test_hdr1 = LFintegrate(test_hdr1,  patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)
                test_hdr2 = LFintegrate(test_hdr2,  patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)
                test_hdr3 = LFintegrate(test_hdr3,  patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)

            else:
                test_hdr1, test_hdr2, test_hdr3 = model_test(multi_data, expo_data, hl_mask, ld_mask)  # [b,c,ah,aw,h,w]

            elapsed_time = datetime.now() - start_time

            ###################################  Tone mapping  ###################################
            test_mask1 = hl_mask.repeat([1, 3, 1, 1, 1, 1]).squeeze(0).cpu().numpy()
            test_mask2 = ld_mask.repeat([1, 3, 1, 1, 1, 1]).squeeze(0).cpu().numpy()
            test_tm1 = log_transformation(test_hdr1, param_u=opt.u_law).squeeze(0).cpu().numpy()    # [c,ah,aw,h,w]
            test_tm2 = log_transformation(test_hdr2, param_u=opt.u_law).squeeze(0).cpu().numpy()
            test_tm3 = log_transformation(test_hdr3, param_u=opt.u_law).squeeze(0).cpu().numpy()
            label_tm = log_transformation(label_hdr, param_u=opt.u_law).squeeze(0).cpu().numpy()

            #################################  Calculate metrics  #################################
            test_hdr1 = test_hdr1.squeeze(0).cpu().numpy()
            test_hdr2 = test_hdr2.squeeze(0).cpu().numpy()
            test_hdr3 = test_hdr3.squeeze(0).cpu().numpy()
            label_hdr = label_hdr.squeeze(0).cpu().numpy()

            linear_psnr, linear_ssim = calculate_quality(test_hdr3, label_hdr)
            log_psnr, log_ssim = calculate_quality(test_tm3, label_tm)
            print('Test image.%d,  Linear PSNR: %s,  Linear SSIM: %s,  Log PSNR: %s,  Log SSIM: %s,  Elapsed time: %s'
                  % (idx_iter + 1, linear_psnr, linear_ssim, log_psnr, log_ssim, elapsed_time))

            ave_linear_psnr += linear_psnr / len(test_loader)
            ave_linear_ssim += linear_ssim / len(test_loader)
            ave_log_psnr += log_psnr / len(test_loader)
            ave_log_ssim += log_ssim / len(test_loader)

            ###################################  Save results  ###################################
            test_mask1 = lfi2mlia(test_mask1)
            test_mask2 = lfi2mlia(test_mask2)
            test_hdr1 = lfi2mlia(test_hdr1)
            test_hdr2 = lfi2mlia(test_hdr2)
            test_hdr3 = lfi2mlia(test_hdr3)
            test_tm1 = lfi2mlia(test_tm1)
            test_tm2 = lfi2mlia(test_tm2)
            test_tm3 = lfi2mlia(test_tm3)

            imageio.imwrite(opt.output_path + str(idx_iter + 1) + '_hl_mask.png', (test_mask1.clip(0, 1) * 255.0).astype(np.uint8))
            imageio.imwrite(opt.output_path + str(idx_iter + 1) + '_ld_mask.png', (test_mask2.clip(0, 1) * 255.0).astype(np.uint8))
            imageio.imwrite(opt.output_path + str(idx_iter + 1) + '_infer1.png', (test_tm1.clip(0, 1) * 255.0).astype(np.uint8))
            imageio.imwrite(opt.output_path + str(idx_iter + 1) + '_infer2.png', (test_tm2.clip(0, 1) * 255.0).astype(np.uint8))
            imageio.imwrite(opt.output_path + str(idx_iter + 1) + '_infer3.png', (test_tm3.clip(0, 1) * 255.0).astype(np.uint8))
            radiance_writer(opt.output_path + str(idx_iter + 1) + '_infer1.hdr', test_hdr1)
            radiance_writer(opt.output_path + str(idx_iter + 1) + '_infer2.hdr', test_hdr2)
            radiance_writer(opt.output_path + str(idx_iter + 1) + '_infer3.hdr', test_hdr3)

            file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
            file_handle.write('Img.%d,  Linear PSNR: %s,  Linear SSIM: %s,  Log PSNR: %s,  Log SSIM: %s,\n'
                              % (idx_iter + 1, linear_psnr, linear_ssim, log_psnr, log_ssim))
            file_handle.close()

        print('Test end!  Average metric: Linear PSNR: %s,  Linear SSIM: %s,  Log PSNR: %s,  Log SSIM: %s'
              % (ave_linear_psnr, ave_linear_ssim, ave_log_psnr, ave_log_ssim))

        file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
        file_handle.write('Average,  Linear PSNR: %s,  Linear SSIM: %s,  Log PSNR: %s,  Log SSIM: %s,\n'
                          % (ave_linear_psnr, ave_linear_ssim, ave_log_psnr, ave_log_ssim))
        file_handle.close()


def main(opt):
    time1 = datetime.now()
    test_set = TestSetLoader(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} test image from {}'.format(len(test_loader), opt.testset_path))
    test(opt, test_loader)

    time2 = datetime.now() - time1
    print('testing end, and taking: ', time2)


##############################################
if __name__ == '__main__':
    main(cfg)
