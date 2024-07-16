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
import imageio
from model import Build_GFNet
from load_dataset import TrainSetLoader, TestSetLoader
from loss import get_loss
from utils import mk_dir, log_transformation, lfi2mlia, to_2d, get_highlight_mask, get_lowdark_mask
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


#########################################################################################################
parser = argparse.ArgumentParser(description="Ghost-free high dynamic range light field imaging -- train mode")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ang_res", type=int, default=7, help="Angular resolution of light field")
parser.add_argument("--crf_gamma", type=float, default=1/0.7, help="Gamma value of camera response function")
parser.add_argument("--u_law", type=float, default=5000.0, help="u value of dynamic range compressor")
parser.add_argument("--model_dir", type=str, default="models/", help="Checkpoints path")
parser.add_argument("--trainset_path", type=str, default="Dataset/TrainingData_7x7", help="Training data path")

parser.add_argument("--patch_size", type=int, default=96, help="Cropped LF patch size, i.e., spatial resolution")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--train_epoch", type=int, default=100, help="Number of epochs to train")
parser.add_argument("--n_steps", type=int, default=25, help="Number of epochs to update learning rate")
parser.add_argument("--decay_value", type=float, default=0.5, help="Learning rate decaying factor")

parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--num_save", type=int, default=1, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")
parser.add_argument("--train_save", type=int, default=1, help="Save the image in training")

cfg = parser.parse_args()
print(cfg)

#######################################################################################
torch.cuda.set_device(cfg.device)

#####################################################################################################
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Weight initialization
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

# Loss functions
loss_all = get_loss(cfg)

###############################################################################################
def train(opt, train_loader, test_loader):

    print('==>training')
    start_time = datetime.now()

    train_results_dir = 'training_results/'
    if opt.train_save:
        mk_dir(train_results_dir)

    # model save folder
    mk_dir(opt.model_dir)

    #######################################################################################
    # Build model
    print("Building GFNet")
    model_train = Build_GFNet(opt).to(opt.device)
    # Initialize weight
    model_train.apply(weights_init_xavier)
    # for para_name in model_train.state_dict():  # print trained parameters
    #     print(para_name)

    total = sum([param.nelement() for param in model_train.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    #######################################################################################
    # Optimizer and loss logger
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.n_steps, gamma=opt.decay_value)
    losslogger = defaultdict(list)

    # Model parallel training
    # if torch.cuda.device_count() > 1:
    #     model_train = nn.DataParallel(model_train)

    #######################################################################################
    # Reload previous parameters
    if opt.resume_epoch:
        resume_path = join(opt.model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>Loading model parameters '{}'".format(resume_path))
            checkpoints = torch.load(resume_path, map_location={'cuda:0': opt.device})
            model_train.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            scheduler.load_state_dict(checkpoints['scheduler'])
            losslogger = checkpoints['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    #######################################################################################
    # start training
    epoch_state = opt.resume_epoch + 1
    for idx_epoch in range(epoch_state, opt.train_epoch + 1):   # epochs
        # Train
        model_train.train()
        print('Current epoch learning rate: %e' % (optimizer.state_dict()['param_groups'][0]['lr']))
        loss_epoch = 0.     # Total loss per epoch

        for idx_iter, (multi_data, expo_data, label_hdr) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # data: [b,9,ah,aw,h,w], [b,3,1];  label: [b,3,ah,aw,h,w]
            multi_data, expo_data, label_hdr = \
                Variable(multi_data).to(opt.device), Variable(expo_data).to(opt.device), Variable(label_hdr).to(opt.device)

            # mask: [b,1,ah,aw,h,w]  mask region is 1
            hl_mask = get_highlight_mask(in_data=multi_data[:, 3:6, :, :, :, :], thr=0.5)    # middle-exposure
            ld_mask = get_lowdark_mask(in_data=multi_data[:, 3:6, :, :, :, :], thr=0.5)

            ############################  Forward inference  ############################
            # [b,3,ah,aw,h,w]
            train_hdr1, train_hdr2, train_hdr3 = model_train(multi_data, expo_data, hl_mask, ld_mask)

            ###############################  Tone mapping  ###############################
            train_tm1 = log_transformation(train_hdr1, param_u=opt.u_law)
            train_tm2 = log_transformation(train_hdr2, param_u=opt.u_law)
            train_tm3 = log_transformation(train_hdr3, param_u=opt.u_law)
            label_tm = log_transformation(label_hdr, param_u=opt.u_law)

            ###############################  Calculate loss  ###############################
            ssim_loss1 = loss_all['ssim_loss'](infer=to_2d(train_tm1), gt=to_2d(label_tm))
            pixel_loss1 = loss_all['pixel_loss'](infer=train_tm1 * hl_mask, gt=label_tm * hl_mask)
            loss1 = ssim_loss1 + pixel_loss1 * 10

            ssim_loss2 = loss_all['ssim_loss'](infer=to_2d(train_tm2), gt=to_2d(label_tm))
            pixel_loss2 = loss_all['pixel_loss'](infer=train_tm2 * ld_mask, gt=label_tm * ld_mask)
            loss2 = ssim_loss2 + pixel_loss2 * 2.0

            ssim_loss3 = loss_all['ssim_loss'](infer=to_2d(train_tm3), gt=to_2d(label_tm))
            pixel_loss3 = loss_all['pixel_loss'](infer=train_tm3, gt=label_tm)
            loss3 = ssim_loss3 + pixel_loss3 * 1.5

            loss = loss1 + loss2 + loss3 * 2.0

            # Cumulative loss
            loss_epoch += loss.item()

            ###########################  Backward and optimize  ###########################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###########################  Save training results  ###########################
            if opt.train_save:
                if idx_iter % 1000 == 0 or idx_iter == len(train_loader)-1:
                    in_name1 = '{}/epoch{}_iter{}_input1.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    in_name2 = '{}/epoch{}_iter{}_input2.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    in_name3 = '{}/epoch{}_iter{}_input3.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    mask_name1 = '{}/epoch{}_iter{}_mask1.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    mask_name2 = '{}/epoch{}_iter{}_mask2.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    infer_name1 = '{}/epoch{}_iter{}_infer1.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    infer_name2 = '{}/epoch{}_iter{}_infer2.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    infer_name3 = '{}/epoch{}_iter{}_infer3.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    label_name = '{}/epoch{}_iter{}_label.jpg'.format(train_results_dir, idx_epoch, idx_iter)

                    # [c,ah,aw,h,w]
                    save_in1 = (multi_data[0, 0:3, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_in2 = (multi_data[0, 3:6, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_in3 = (multi_data[0, 6:9, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_mask1 = (hl_mask[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_mask2 = (ld_mask[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_infer1 = (train_tm1[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_infer2 = (train_tm2[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_infer3 = (train_tm3[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_label = (label_tm[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)

                    # [3,ah,aw,h,w] --> [h*ah,w*aw,3]
                    imageio.imwrite(in_name1, lfi2mlia(save_in1).astype(np.uint8))
                    imageio.imwrite(in_name2, lfi2mlia(save_in2).astype(np.uint8))
                    imageio.imwrite(in_name3, lfi2mlia(save_in3).astype(np.uint8))
                    imageio.imwrite(mask_name1, lfi2mlia(save_mask1).astype(np.uint8))
                    imageio.imwrite(mask_name2, lfi2mlia(save_mask2).astype(np.uint8))
                    imageio.imwrite(infer_name1, lfi2mlia(save_infer1).astype(np.uint8))
                    imageio.imwrite(infer_name2, lfi2mlia(save_infer2).astype(np.uint8))
                    imageio.imwrite(infer_name3, lfi2mlia(save_infer3).astype(np.uint8))
                    imageio.imwrite(label_name, lfi2mlia(save_label).astype(np.uint8))

            if (idx_iter % 1000 == 0):
                print('Training==>>Epoch: %d,  ssim_loss1:  %s,  pixel_loss1:  %s,  loss1: %s,  '
                      'ssim_loss2:  %s,  pixel_loss2:  %s,  loss2: %s,  '
                      'ssim_loss3:  %s,  pixel_loss3:  %s,  loss3: %s'
                      % (idx_epoch, ssim_loss1.item(), pixel_loss1.item(), loss1.item(),
                         ssim_loss2.item(), pixel_loss2.item(), loss2.item(),
                         ssim_loss3.item(), pixel_loss3.item(), loss3.item()))

        scheduler.step()

        ####################################  Print loss  ####################################
        losslogger['epoch'].append(idx_epoch)
        losslogger['loss'].append(loss_epoch/len(train_loader))
        elapsed_time = datetime.now() - start_time
        print('Training==>>Epoch: %d,  loss: %s,  elapsed time: %s'
              % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))

        # write loss
        file_handle = open('loss.txt', mode='a')
        file_handle.write('epoch: %d,  loss: %s,  elapsed time: %s\n'
                          % (idx_epoch, loss_epoch/len(train_loader), elapsed_time))
        file_handle.close()

        # save trained model's parameters
        if idx_epoch % opt.num_save == 0:
            model_save_path = join(opt.model_dir, "model_epoch_{}.pth".format(idx_epoch))
            state = {'epoch': idx_epoch, 'model': model_train.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoints saved to {}".format(model_save_path))

        # save loss figure
        if idx_epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(opt.model_dir + "loss.png")
            plt.close('all')


def main(opt):
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('Loaded {} training image from {}'.format(len(train_loader), opt.trainset_path))

    test_set = TestSetLoader(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    train(opt, train_loader, test_loader)


##############################################
if __name__ == '__main__':
    main(cfg)
