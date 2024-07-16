import torch
import torch.nn.functional as functional
import numpy as np
import scipy.io as sio
from scipy.signal import convolve2d
import os
import random
import math
import h5py
from einops import rearrange
import OpenEXR, Imath
import pyexr


class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


# Write HDR image using OpenEXR
def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        R = (img[:, :, 0]).astype(np.float32).tostring()
        G = (img[:, :, 1]).astype(np.float32).tostring()
        B = (img[:, :, 2]).astype(np.float32).tostring()
        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)


def loadEXR(name_hdr):
    return pyexr.read_all(name_hdr)['default'][:, :, 0:3]


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def mk_dir(required_dir):
    if not os.path.exists(required_dir):
        os.makedirs(required_dir)


def lfi2mlia(in_lfi):
    # [3,ah,aw,h,w] to [h*ah,w*aw,c]
    out_mlia = rearrange(in_lfi, 'c ah aw h w -> (h ah) (w aw) c')
    return out_mlia


def to_2d(in_data):
    # [b,c,ah,aw,h,w]
    out_data = rearrange(in_data, 'b c ah aw h w -> (b ah aw) c h w')
    return out_data


def cal_lum(in_rgb_img):
    # [b,c,ah,aw,h,w]
    in_r = in_rgb_img[:, 0, :, :, :, :]
    in_g = in_rgb_img[:, 1, :, :, :, :]
    in_b = in_rgb_img[:, 2, :, :, :, :]
    lum_img = 0.299 * in_r + 0.587 * in_g + 0.114 * in_b
    lum_img = torch.unsqueeze(lum_img, dim=1)   # [b,1,ah,aw,h,w]
    return lum_img


def get_highlight_mask(in_data, thr=0.5):
    # [b,1,ah,aw,h,w]
    in_lum_data = cal_lum(in_data)
    ones = torch.ones_like(in_lum_data)
    zeros = torch.zeros_like(in_lum_data)
    hl_msk = torch.where(in_lum_data >= thr, ones, zeros)
    return hl_msk


def get_lowdark_mask(in_data, thr=0.5):
    # [b,1,ah,aw,h,w]
    in_lum_data = cal_lum(in_data)
    ones = torch.ones_like(in_lum_data)
    zeros = torch.zeros_like(in_lum_data)
    ld_msk = torch.where(in_lum_data <= thr, ones, zeros)
    return ld_msk


def log_transformation(in_hdr, param_u=5000.):
    out_tm = torch.log(torch.tensor(1.) + torch.tensor(param_u) * in_hdr) / torch.log(torch.tensor(1. + param_u))
    return out_tm


def inverse_log_transformation(log_img, log_param=5000.):
    scale_param = torch.log(torch.tensor(1.) + log_param)
    radia_img = (torch.exp(scale_param * log_img) - torch.tensor(1.)) / torch.tensor(log_param)
    return radia_img


def log_transformation_np(in_hdr, param_u=5000.):
    out_tm = np.log(1.0 + param_u * in_hdr) / np.log(1.0 + param_u)
    return out_tm


def inverse_log_transformation_np(log_img, log_param=5000.):
    scale_param = np.log(1.0 + log_param)
    radia_img = (np.exp(scale_param * log_img) - 1.0) / log_param
    return radia_img


def LDRtoHDR(in_ldr, expo, gamma=1/0.7):
    in_ldr = np.clip(in_ldr, 0, 1)
    out_hdr = in_ldr ** gamma
    out_hdr = out_hdr / expo
    return out_hdr


def HDRtoLDR(in_hdr, expo, gamma=1/0.7):
    in_hdr = in_hdr * expo
    in_hdr = np.clip(in_hdr, 0, 1)
    out_ldr = in_hdr ** (1/gamma)
    return out_ldr


def LDRtoLDR(in_ldr, expo1, expo2, gamma=1/0.7):
    Radiance = LDRtoHDR(in_ldr, expo1, gamma)
    out_ldr = HDRtoLDR(Radiance, expo2, gamma)
    return out_ldr


def tensor_LDRtoHDR(in_ldr, expo, gamma=1/0.7):
    in_ldr = torch.clamp(in_ldr, min=0, max=1)
    out_hdr = in_ldr ** gamma
    out_hdr = out_hdr / expo
    return out_hdr


def tensor_HDRtoLDR(in_hdr, expo, gamma=1/0.7):
    in_hdr = in_hdr * expo
    in_hdr = torch.clamp(in_hdr, min=0, max=1)
    out_ldr = in_hdr ** (1/gamma)
    return out_ldr


def tensor_LDRtoLDR(in_ldr, expo1, expo2, gamma=1/0.7):
    Radiance = tensor_LDRtoHDR(in_ldr, expo1, gamma)
    out_ldr = tensor_HDRtoLDR(Radiance, expo2, gamma)
    return out_ldr


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 10 * math.log10(pixel_max / mse)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumg = g.sum()
    if sumg != 0:
        g /= sumg
    return g


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def calculate_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1.0):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'same')
    mu2 = filter2(im2, window, 'same')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'same') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'same') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'same') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))


def calculate_quality(im1, im2):
    # [c,ah,aw,h,w]
    score1 = calculate_psnr(im1, im2)
    score2 = 0.
    an1, an2 = im1.shape[1:3]
    for a1 in range(an1):
        for a2 in range(an2):
            for ci in range(3):
                score2 = score2 + calculate_ssim(im1[ci, a1, a2, :, :], im2[ci, a1, a2, :, :])
    score2 = score2 / (an1 * an2 * 3.)
    return score1, score2


def ColorAugmentation(lfi, index):
    # [ah,aw,h,w,c]
    if index == 1:
        order = [0, 1, 2]
    elif index == 2:
        order = [0, 2, 1]
    elif index == 3:
        order = [1, 0, 2]
    elif index == 4:
        order = [1, 2, 0]
    elif index == 5:
        order = [2, 1, 0]
    else:
        order = [2, 0, 1]
    aug_lfi = lfi[:, :, :, :, order]
    return aug_lfi


def crop_lf_patch(LF_data, spa_length, spa_bound):
    # LF_data: [b,c,ah,aw,h,w]
    test_b, test_c, test_ah, test_aw, test_h, test_w = LF_data.shape

    if test_h % spa_length == 0:
        row_num = test_h // spa_length - 1
    else:
        row_num = test_h // spa_length

    if test_w % spa_length == 0:
        col_num = test_w // spa_length - 1
    else:
        col_num = test_w // spa_length

    # [1,b,c,ah,aw,h,w]
    LF_patch_volume = torch.zeros((1, test_b, test_c, test_ah, test_aw, spa_length + spa_bound, spa_length + spa_bound)).to(LF_data.device)

    # left top
    for row_cp in range(row_num):
        for col_cp in range(col_num):
            crop_LF_patch = LF_data[:, :, :, :, row_cp * spa_length:(row_cp + 1) * spa_length + spa_bound, col_cp * spa_length:(col_cp + 1) * spa_length + spa_bound]
            crop_LF_patch = crop_LF_patch.unsqueeze(0)
            LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    h_bound_start = test_h - spa_length - spa_bound
    w_bound_start = test_w - spa_length - spa_bound

    # right
    for row_cp in range(row_num):
        crop_LF_patch = LF_data[:, :, :, :, row_cp * spa_length:(row_cp + 1) * spa_length + spa_bound, w_bound_start:]
        crop_LF_patch = crop_LF_patch.unsqueeze(0)
        LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # bottom
    for col_cp in range(col_num):
        crop_LF_patch = LF_data[:, :, :, :, h_bound_start:, col_cp * spa_length:(col_cp + 1) * spa_length + spa_bound]
        crop_LF_patch = crop_LF_patch.unsqueeze(0)
        LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # right bottom
    crop_LF_patch = LF_data[:, :, :, :, h_bound_start:, w_bound_start:]
    crop_LF_patch = crop_LF_patch.unsqueeze(0)
    LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # [num,b,c,ah,aw,h,w]
    LF_patch_volume = LF_patch_volume[1:, :, :, :, :, :, :]
    return LF_patch_volume, row_num, col_num


def merge_lf_patch(in_lf_patch_volume, rnum, cnum, sr_h, sr_w, spa_length, spa_bound):
    # in_lf_patch_volume: [num,b,c,ah,aw,h,w]
    rec_n, rec_b, rec_c, rec_ah, rec_aw, rec_h, rec_w = in_lf_patch_volume.shape

    h_bound = sr_h - spa_length * rnum
    w_bound = sr_w - spa_length * cnum
    spa_bound_sub = spa_bound // 2

    # left top
    rec_lf_data = torch.zeros(rec_b, rec_c, rec_ah, rec_aw, sr_h, sr_w).to(in_lf_patch_volume.device)
    pvx = 0
    for pvi in range(rnum):
        for pvj in range(cnum):
            if (pvi==0 and pvj==0):
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]

            elif (pvi == 0 and pvj > 0):
                pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]

            elif (pvi>0 and pvj==0):
                pre_lf_patch = in_lf_patch_volume[pvx - cnum, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]

            else:
                pre_lf_patch1 = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
                pre_lf_patch2 = in_lf_patch_volume[pvx - cnum, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch1[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]
                tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch2[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]
            rec_lf_data[:, :, :, :, pvi*spa_length:(pvi+1)*spa_length, pvj*spa_length:(pvj+1)*spa_length] = tmp_lf_patch[:, :, :, :, :spa_length, :spa_length]
            pvx = pvx + 1

    # right
    for pvk in range(rnum):
        if (pvk==0):
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
        else:
            pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
            tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]

        rec_lf_data[:, :, :, :, pvk*spa_length:(pvk+1)*spa_length, -w_bound:] = tmp_lf_patch[:, :, :, :, :spa_length, -w_bound:]
        pvx = pvx + 1

    # bottom
    for pvl in range(cnum):
        if (pvl == 0):
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
        else:
            pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
            tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]

        rec_lf_data[:, :, :, :, -h_bound:, pvl*spa_length:(pvl+1)*spa_length] = tmp_lf_patch[:, :, :, :, -h_bound:, :spa_length]
        pvx = pvx + 1

    # right bottom
    tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
    rec_lf_data[:, :, :, :, -h_bound:, -w_bound:] = tmp_lf_patch[:, :, :, :, -h_bound:, -w_bound:]
    return rec_lf_data


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out


def LFdivide(lf, patch_size, stride):
    # [c,ah,aw,h,w]
    _, ah, aw, sai_h, sai_w = lf.shape
    data = rearrange(lf, 'c ah aw h w -> (ah aw) c h w')

    bdr = (patch_size - stride) // 2
    numU = (sai_h + bdr * 2 - 1) // stride
    numV = (sai_w + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = functional.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(ah aw) (c h w) (n1 n2) -> n1 n2 ah aw c h w',
                      n1=numU, n2=numV, ah=ah, aw=aw, h=patch_size, w=patch_size)
    return subLF


def LFintegrate(subLFs, patch_size, stride, sai_h, sai_w):
    # [n1 n2,ah,aw,c,h,w]
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> c u v (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:sai_h, 0:sai_w]
    return outLF