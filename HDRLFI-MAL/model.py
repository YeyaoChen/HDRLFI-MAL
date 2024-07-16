import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from modules import input_process, SFE, MS_CrossAttFA, FFM, SFF, IRB, LEB, conv_softmax, SAB
import cv2
import scipy.io as scio
from utils import log_transformation, radiance_writer


class Build_GFNet(nn.Module):
    def __init__(self, par):
        super(Build_GFNet, self).__init__()
        self.crf_value = par.crf_gamma
        chan_num = 32
        self.SFE = SFE(nChannels=chan_num)
        self.MS_CrossAttFA1 = MS_CrossAttFA(nChannels=chan_num)
        self.MS_CrossAttFA2 = MS_CrossAttFA(nChannels=chan_num)
        self.FFM1 = FFM(nBlocks=2, nChannels=chan_num, ang_res2=par.ang_res**2, num_heads=4, expansion_factor=2)
        self.FFM2 = FFM(nBlocks=2, nChannels=chan_num, ang_res2=par.ang_res**2, num_heads=4, expansion_factor=2)
        self.LEB1 = LEB(nLayers=3, nChannels=chan_num)
        self.LEB2 = LEB(nLayers=3, nChannels=chan_num)
        self.conv_1 = conv_softmax(nChannels=chan_num)
        self.conv_2 = conv_softmax(nChannels=chan_num)
        self.IRB1 = IRB(nChannels=chan_num)
        self.IRB2 = IRB(nChannels=chan_num)
        self.IRB3 = IRB(nChannels=chan_num)

    def forward(self, in_multi_lfs, in_expo_values, in_hl_mask, in_ld_mask):
        # in_multi_lfs: [b,9,ah,aw,h,w]; in_expo_values: [b,3,1]
        in_lf1, in_lf2, in_lf3, in_lf12, in_lf23 = \
            input_process(multi_lfs=in_multi_lfs, expo_values=in_expo_values, crf_gamma=self.crf_value)

        # Shallow feature extraction: [b,6,ah,aw,h,w] --> [b,c,ah,aw,h,w]
        feats1 = self.SFE(in_lf1)
        feats2 = self.SFE(in_lf2)
        feats3 = self.SFE(in_lf3)

        feats12 = self.SFE(in_lf12)
        feats23 = self.SFE(in_lf23)

        # Feature alignment based on cross attention: [b,c,ah,aw,h,w]
        aligned_feats1 = self.MS_CrossAttFA1(feats2, feats12, feats1)
        aligned_feats3 = self.MS_CrossAttFA2(feats23, feats3, feats3)

        # Aligned feature fusion based on self-attention: [b,c,ah,aw,h,w]
        fused_feats12 = self.FFM1(aligned_feats1, feats2, in_hl_mask)
        fused_feats32 = self.FFM2(aligned_feats3, feats2, in_ld_mask)

        # Local enhancement block
        local_feats12 = self.LEB1(feats1, feats2, in_hl_mask)
        local_feats32 = self.LEB2(feats3, feats2, in_ld_mask)

        # Motion feature: [b,c,ah,aw,h,w]
        motion_feats1 = torch.abs(feats1 - aligned_feats1)
        motion_feats3 = torch.abs(feats3 - aligned_feats3)

        # Feature fusion
        blend_feats12 = self.conv_1(motion_feats1, fused_feats12, local_feats12)
        blend_feats32 = self.conv_2(motion_feats3, fused_feats32, local_feats32)

        # Spatial fusion: [b,c,ah,aw,h,w]
        blend_feats123 = (blend_feats12 * in_hl_mask + blend_feats32 * in_ld_mask) / (in_hl_mask + in_ld_mask)

        # Image reconstruction: [b,c,ah,aw,h,w]
        net_img1 = self.IRB1(blend_feats12 + feats2)
        net_img2 = self.IRB2(blend_feats32 + feats2)
        net_img3 = self.IRB3(blend_feats123 + feats2)

        return net_img1, net_img2, net_img3
