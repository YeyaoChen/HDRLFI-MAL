import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
from math import exp
import Vgg19


# ssim functions
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# Functions
class cal_grad_map(nn.Module):
    def __init__(self):
        super(cal_grad_map, self).__init__()
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).repeat([3, 3, 1, 1])    # [3,3,3,3]
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).repeat([3, 3, 1, 1])    # [3,3,3,3]
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        # [b,1,h,w]
        x_h = functional.conv2d(x, self.weight_h, padding=1, groups=3)
        x_v = functional.conv2d(x, self.weight_v, padding=1, groups=3)
        grad_x = torch.sqrt(torch.pow(x_h, 2) + torch.pow(x_v, 2) + 1e-8)
        return grad_x


#################### Loss functions ####################
class cal_pixel_loss(nn.Module):
    def __init__(self):
        super(cal_pixel_loss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, infer, gt):
        pixel_loss = self.loss(infer, gt)    # [b,c,h,w]
        return pixel_loss


class cal_ssim_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(cal_ssim_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 3
        self.window = create_window(window_size, self.channels)

    def forward(self, infer, gt):
        # [b,3,h,w]
        channels = infer.shape[1]
        if channels == self.channels and self.window.data.type() == infer.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channels)

            if infer.is_cuda:
                window = window.cuda(infer.get_device())
            window = window.type_as(infer)

            self.window = window
            self.channels = channels

        ssim_value = _ssim(infer, gt, window, self.window_size, channels, self.size_average)
        ssim_loss = 1.0 - ssim_value
        return ssim_loss


class cal_perce_loss(nn.Module):
    def __init__(self, device):
        super(cal_perce_loss, self).__init__()
        self.loss = torch.nn.L1Loss()
        self.device = device

    def forward(self, infer, gt):
        # [b,3,h,w]
        vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        infer_relu = vgg19(infer)
        gt_relu = vgg19(gt)
        per_loss = self.loss(infer_relu, gt_relu)
        return per_loss


class cal_grad_loss(nn.Module):
    def __init__(self):
        super(cal_grad_loss, self).__init__()
        self.loss = torch.nn.L1Loss()
        self.grad_map = cal_grad_map()

    def forward(self, infer, gt):
        infer_grad = self.grad_map(infer)
        gt_grad = self.grad_map(gt)
        return self.loss(infer_grad, gt_grad)


def gradient(in_5d_lfi):
    # [b,c,ah,aw,h,w]
    D_dy = in_5d_lfi[:, :, :, :, 1:, :] - in_5d_lfi[:, :, :, :, :-1, :]
    D_dx = in_5d_lfi[:, :, :, :, :, 1:] - in_5d_lfi[:, :, :, :, :, :-1]
    D_day = in_5d_lfi[:, :, 1:, :, :, :] - in_5d_lfi[:, :, :-1, :, :, :]
    D_dax = in_5d_lfi[:, :, :, 1:, :, :] - in_5d_lfi[:, :, :, :-1, :, :]
    return D_dx, D_dy, D_dax, D_day


class cal_epi_grad_loss(nn.Module):
    def __init__(self):
        super(cal_epi_grad_loss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, infer, gt):
        # [b,c,ah,aw,h,w]
        infer_dx, infer_dy, infer_dax, infer_day = gradient(infer)
        gt_dx, gt_dy, gt_dax, gt_day = gradient(gt)
        return self.loss(infer_dx, gt_dx) + self.loss(infer_dy, gt_dy) + self.loss(infer_dax, gt_dax) + self.loss(infer_day, gt_day)


def get_loss(opt):
    losses = {}
    losses['pixel_loss'] = cal_pixel_loss()
    losses['ssim_loss'] = cal_ssim_loss()
    losses['perceptual_loss'] = cal_perce_loss(opt.device)
    losses['gradient_loss'] = cal_grad_loss()
    losses['epi_gradient_loss'] = cal_epi_grad_loss()
    return losses