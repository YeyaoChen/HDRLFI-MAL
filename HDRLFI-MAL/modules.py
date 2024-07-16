import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import numbers
from einops import rearrange
from utils import tensor_LDRtoHDR, tensor_LDRtoLDR


# network input
def input_process(multi_lfs, expo_values, crf_gamma):
    # in_multi_lfs: [b,9,ah,aw,h,w]; in_expo_values: [b,3,1]
    in_ldr1 = multi_lfs[:, 0:3, :, :, :, :]     # under-exposure
    in_ldr2 = multi_lfs[:, 3:6, :, :, :, :]     # middle-exposure
    in_ldr3 = multi_lfs[:, 6:9, :, :, :, :]     # over-exposure

    # [b,1,1,1,1,1]
    in_expo1 = expo_values[:, 0, :].unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
    in_expo2 = expo_values[:, 1, :].unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
    in_expo3 = expo_values[:, 2, :].unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

    # LDR to HDR: convert to linear domain
    in_hdr1 = tensor_LDRtoHDR(in_ldr=in_ldr1, expo=in_expo1, gamma=crf_gamma)
    in_hdr2 = tensor_LDRtoHDR(in_ldr=in_ldr2, expo=in_expo2, gamma=crf_gamma)
    in_hdr3 = tensor_LDRtoHDR(in_ldr=in_ldr3, expo=in_expo3, gamma=crf_gamma)

    # multi-exposure inputs: [b,6,ah,aw,h,w]
    in_lfs1 = torch.cat([in_ldr1, in_hdr1], dim=1)
    in_lfs2 = torch.cat([in_ldr2, in_hdr2], dim=1)
    in_lfs3 = torch.cat([in_ldr3, in_hdr3], dim=1)

    # exposure conversion1: [b,6,ah,aw,h,w]
    in_u2m_ldr = tensor_LDRtoLDR(in_ldr=in_ldr1, expo1=in_expo1, expo2=in_expo2, gamma=crf_gamma)
    in_u2m_hdr = tensor_LDRtoHDR(in_u2m_ldr, expo=in_expo2, gamma=crf_gamma)
    in_aux1 = torch.cat([in_u2m_ldr, in_u2m_hdr], dim=1)

    # exposure conversion2: [b,6,ah,aw,h,w]
    in_m2o_ldr = tensor_LDRtoLDR(in_ldr=in_ldr2, expo1=in_expo2, expo2=in_expo3, gamma=crf_gamma)
    in_m2o_hdr = tensor_LDRtoHDR(in_m2o_ldr, expo=in_expo3, gamma=crf_gamma)
    in_aux2 = torch.cat([in_m2o_ldr, in_m2o_hdr], dim=1)

    return in_lfs1, in_lfs2, in_lfs3, in_aux1, in_aux2


# Sub-modules
class size_interp(nn.Module):
    def __init__(self):
        super(size_interp, self).__init__()

    def forward(self, x, tar_size):
        return functional.interpolate(input=x, size=tar_size, mode='bilinear')


# LayerNorm (LN)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, nChannels, LN_type):
        super(LayerNorm, self).__init__()
        self.LN_type = LN_type
        self.body = BiasFree_LayerNorm(nChannels)

    def forward(self, x):
        in_b, in_c, in_ah, in_aw, _, _ = x.shape
        if (self.LN_type == "channel"):
            x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
            h, w = x.shape[-2:]
            out = to_4d(self.body(to_3d(x)), h, w)
            out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        else:
            x = rearrange(x, 'b c ah aw h w -> (b c) (ah aw) h w')   # [b*c,an2,h,w]
            h, w = x.shape[-2:]
            out = to_4d(self.body(to_3d(x)), h, w)
            out = rearrange(out, '(b c) (ah aw) h w -> b c ah aw h w', b=in_b, c=in_c, ah=in_ah, aw=in_aw)
        return out


# Residual Atrous Spatial Pyramid Pooling (RASPP)
class ResASPP(nn.Module):
    def __init__(self, nChannels):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                    nn.ReLU(inplace=True))
        self.conv_d = nn.Conv2d(in_channels=nChannels*3, out_channels=nChannels, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x):
        # [b,c,h,w]
        buffer_1 = self.conv_1(x)
        buffer_2 = self.conv_2(x)
        buffer_3 = self.conv_3(x)
        buffer = self.conv_d(torch.cat([buffer_1, buffer_2, buffer_3], dim=1))
        return x + buffer


# Shallow Feature Extraction (SFE)
class SFE(nn.Module):
    def __init__(self, nChannels):
        super(SFE, self).__init__()
        self.sfe = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResASPP(nChannels=nChannels)
        )

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x.shape
        x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        out = self.sfe(x)
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Cross attention-based Feature Alignment (CrossAttFA)
class CrossAttFA(nn.Module):
    def __init__(self, nChannels, patch_size, padding, strides):
        super(CrossAttFA, self).__init__()
        self.ps = patch_size
        self.padding = padding
        self.strides = strides
        self.conv = nn.Conv2d(in_channels=nChannels, out_channels=nChannels//8, kernel_size=1, stride=1, padding=0, bias=False)

    def aggregate(self, inputs, dim, index):
        views = [inputs.size(0)] + [1 if i != dim else -1 for i in range(1, len(inputs.size()))]
        expanse = list(inputs.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(inputs, dim, index)

    def forward(self, x1, x2, x3):
        # [b,c,ah,aw,h,w]
        # x1: middle-exposure; x2: corrected under/middle-exposure; x3: under/over-exposure

        # spatial-angular convolutional tokenization
        in_b, in_c, in_ah, in_aw, in_h, in_w = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        x3 = rearrange(x3, 'b c ah aw h w -> (b ah aw) c h w')

        query = self.conv(x1)     # [b*an2,c/8,h,w]
        key = self.conv(x2)
        value = x3

        # [b*an2,c*ps*ps/8,N] and [b*an2,c*ps*ps,N]
        query = functional.unfold(query, kernel_size=(self.ps, self.ps), padding=self.padding, stride=self.strides)
        key = functional.unfold(key, kernel_size=(self.ps, self.ps), padding=self.padding, stride=self.strides)
        value = functional.unfold(value, kernel_size=(self.ps, self.ps), padding=self.padding, stride=self.strides)
        in_p = value.shape[-2]

        # [b,an2*N,c*ps*ps] and [b,N,c*ps*ps]
        query = rearrange(query, '(b ah aw) p n -> b n (ah aw p)', b=in_b, ah=in_ah, aw=in_aw)
        key = rearrange(key, '(b ah aw) p n -> b n (ah aw p)', b=in_b, ah=in_ah, aw=in_aw)
        value = rearrange(value, '(b ah aw) p n -> b n (ah aw p)', b=in_b, ah=in_ah, aw=in_aw)

        # spatial-angular self-attention
        # [b,N,an2*c*ps*ps]
        query = functional.normalize(query, dim=-1)
        key = functional.normalize(key, dim=-1)

        # [b,N,N]
        attn_map = (query @ key.transpose(-2, -1))        # [b,N,N]
        _, hard_att_map = torch.max(attn_map, dim=-1)     # [b,N]

        # [b,N,an2*c*ps*ps] --> [b*an2,c*ps*ps,N]
        out = self.aggregate(value, 1, hard_att_map)
        out = rearrange(out, 'b n (ah aw p) -> (b ah aw) p n', ah=in_ah, aw=in_aw, p=in_p)

        # spatial-angular convolutional de-tokenization
        # [b*an2,c,h,w]
        out = functional.fold(out, output_size=(in_h, in_w), kernel_size=(self.ps, self.ps),
                              padding=self.padding, stride=self.strides)

        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


class MS_CrossAttFA(nn.Module):
    def __init__(self, nChannels):
        super(MS_CrossAttFA, self).__init__()
        self.ca_fa1 = CrossAttFA(nChannels=nChannels, patch_size=3, padding=1, strides=3)
        self.ca_fa2 = CrossAttFA(nChannels=nChannels, patch_size=5, padding=2, strides=5)
        self.ca_fa3 = CrossAttFA(nChannels=nChannels, patch_size=7, padding=3, strides=7)
        self.conv_attn = nn.Conv2d(in_channels=nChannels*3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2, x3):
        # [b,c,ah,aw,h,w]
        # x1: middle-exposure; x2: corrected under/middle-exposure; x3: under/over-exposure
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x3_scale1 = self.ca_fa1(x1, x2, x3)
        x3_scale2 = self.ca_fa2(x1, x2, x3)
        x3_scale3 = self.ca_fa3(x1, x2, x3)

        x3_scale1 = rearrange(x3_scale1, 'b c ah aw h w -> (b ah aw) c h w')      # [b*an2,c,h,w]
        x3_scale2 = rearrange(x3_scale2, 'b c ah aw h w -> (b ah aw) c h w')
        x3_scale3 = rearrange(x3_scale3, 'b c ah aw h w -> (b ah aw) c h w')

        x3_cat = torch.cat([x3_scale1, x3_scale2, x3_scale3], dim=1)     # [b*an2,3c,h,w]
        attn = self.conv_attn(x3_cat)
        attn = attn.softmax(dim=1)      # [b*an2,3,h,w]

        out = attn[:, 0, :, :].unsqueeze(1) * x3_scale1 + \
              attn[:, 1, :, :].unsqueeze(1) * x3_scale2 + \
              attn[:, 2, :, :].unsqueeze(1) * x3_scale3
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Channel-wise Self-Attention (CSA)
class CSA(nn.Module):
    def __init__(self, nChannels, num_heads):
        super(CSA, self).__init__()
        self.num_heads = num_heads
        self.scale = num_heads ** -0.5
        self.pw_conv = nn.Conv2d(in_channels=nChannels, out_channels=int(nChannels/2)*3, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw_conv = nn.Conv2d(in_channels=int(nChannels/2)*3, out_channels=int(nChannels/2)*3, kernel_size=3, stride=1, padding=1, groups=int(nChannels/2)*3, bias=False)
        self.project_out = nn.Conv2d(in_channels=int(nChannels/2), out_channels=nChannels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, in_c, in_ah, in_aw, in_h, in_w = x.shape
        x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')

        qkv = self.dw_conv(self.pw_conv(x))
        query, key, value = qkv.chunk(3, dim=1)

        query = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        key = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        value = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        query = functional.normalize(query, dim=-1)    # [b,head,c,hw]
        key = functional.normalize(key, dim=-1)

        att_map = (query @ key.transpose(-2, -1)) * self.scale
        att_map = att_map.softmax(dim=-1)    # [b,head,c,c]

        out = (att_map @ value)   # [b,head,c,hw]

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=in_h, w=in_w)
        out = self.project_out(out)

        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# View-wise Self-Attention (VSA)
class VSA(nn.Module):
    def __init__(self, ang_res2):
        super(VSA, self).__init__()
        self.pw_conv = nn.Conv2d(in_channels=ang_res2, out_channels=int(ang_res2/2)*3, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw_conv = nn.Conv2d(in_channels=int(ang_res2/2)*3, out_channels=int(ang_res2/2)*3, kernel_size=3, stride=1, padding=1, groups=int(ang_res2/2)*3, bias=False)
        self.project_out = nn.Conv2d(in_channels=int(ang_res2/2), out_channels=ang_res2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, in_c, in_ah, in_aw, in_h, in_w = x.shape
        x = rearrange(x, 'b c ah aw h w -> (b c) (ah aw) h w')

        qkv = self.dw_conv(self.pw_conv(x))
        query, key, value = qkv.chunk(3, dim=1)

        query = rearrange(query, 'b an2 h w -> b an2 (h w)')
        key = rearrange(key, 'b an2 h w -> b an2 (h w)')
        value = rearrange(value, 'b an2 h w -> b an2 (h w)')

        query = functional.normalize(query, dim=-1)      # [bc,an2,hw]
        key = functional.normalize(key, dim=-1)

        att_map = (query @ key.transpose(-2, -1))
        att_map = att_map.softmax(dim=-1)       # [b,an2,an2]

        out = (att_map @ value)    # [b,an2,hw]

        out = rearrange(out, 'b an2 (h w) -> b an2 h w', h=in_h, w=in_w)
        out = self.project_out(out)

        out = rearrange(out, '(b c) (ah aw) h w -> b c ah aw h w', b=in_b, c=in_c, ah=in_ah, aw=in_aw)
        return out


# Feed-Forward Network (FFN)
class FFN(nn.Module):
    def __init__(self, nChannels, expansion_factor):
        super(FFN, self).__init__()
        hidden_features = int(nChannels*expansion_factor)
        self.project_in = nn.Conv2d(in_channels=nChannels, out_channels=hidden_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.dw_conv = nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=False)
        self.project_out = nn.Conv2d(in_channels=hidden_features, out_channels=nChannels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x.shape
        x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')

        out = self.project_in(x)
        out = functional.gelu(self.dw_conv(out))
        out = self.project_out(out)

        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Channel-wise Transformer Block
class CTB(nn.Module):
    def __init__(self, nChannels, num_heads, expansion_factor):
        super(CTB, self).__init__()
        self.norm1 = LayerNorm(nChannels, "channel")
        self.chan_attn = CSA(nChannels, num_heads)
        self.norm2 = LayerNorm(nChannels, "channel")
        self.ffn = FFN(nChannels, expansion_factor)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        out = x + self.chan_attn(self.norm1(x))
        out = out + self.ffn(self.norm2(out))
        return out


# View-wise Transformer Block
class VTB(nn.Module):
    def __init__(self, nChannels, ang_res2, expansion_factor):
        super(VTB, self).__init__()
        self.norm1 = LayerNorm(ang_res2, "view")
        self.view_attn = VSA(ang_res2)
        self.norm2 = LayerNorm(nChannels, "channel")
        self.ffn = FFN(nChannels, expansion_factor)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        out = x + self.view_attn(self.norm1(x))
        out = out + self.ffn(self.norm2(out))
        return out


# Channel-view Transformer Block
class CVTB(nn.Module):
    def __init__(self, nChannels, ang_res2, num_heads, expansion_factor):
        super(CVTB, self).__init__()
        self.ctb = CTB(nChannels, num_heads, expansion_factor)
        self.vtb = VTB(nChannels, ang_res2, expansion_factor)

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        out = self.ctb(x)
        out = self.vtb(out)
        return out


# Convolutional Layer plus softmax function
class conv_softmax(nn.Module):
    def __init__(self, nChannels):
        super(conv_softmax, self).__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nChannels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x1, x2, x3):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        x3 = rearrange(x3, 'b c ah aw h w -> (b ah aw) c h w')
        attn = self.attn_conv(x1)
        attn = attn.softmax(dim=1)
        out = attn[:, 0, :, :].unsqueeze(1) * x2 + attn[:, 1, :, :].unsqueeze(1) * x3
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Feature Fusion Module (FFM)
class FFM(nn.Module):
    def __init__(self, nBlocks, nChannels, ang_res2, num_heads, expansion_factor):
        super(FFM, self).__init__()
        self.conv_d = nn.Conv2d(in_channels=nChannels*2+1, out_channels=nChannels, kernel_size=1, stride=1, padding=0, bias=False)

        CVTBs = []
        for cvt_index in range(nBlocks):
            CVTBs.append(CVTB(nChannels, ang_res2, num_heads, expansion_factor))
        self.CVTBs = torch.nn.ModuleList(CVTBs)

    def forward(self, x1, x2, msk):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        msk = rearrange(msk, 'b c ah aw h w -> (b ah aw) c h w')
        out = self.conv_d(torch.cat([x1, x2, msk], dim=1))
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)

        for ic in range(len(self.CVTBs)):
            out = self.CVTBs[ic](out)
        return out


# Enhanced Spatial-Angular Separable convolutional layer (ESAS)
class ESAS(nn.Module):
    def __init__(self, nChannels):
        super(ESAS, self).__init__()
        self.spa_conv = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ang_conv = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, in_h, in_w = x.shape
        spa_x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')
        spa_x = self.spa_conv(spa_x)

        ang_x = rearrange(spa_x, '(b ah aw) c h w -> (b h w) c ah aw', b=in_b, ah=in_ah, aw=in_aw)
        ang_x = self.ang_conv(ang_x)

        out = rearrange(ang_x, '(b h w) c ah aw -> b c ah aw h w', b=in_b, h=in_h, w=in_w)
        return out


# Local Enhancement Block (LEB)
class LEB(nn.Module):
    def __init__(self, nLayers, nChannels):
        super(LEB, self).__init__()
        self.conv_f = nn.Conv2d(in_channels=nChannels*2+1, out_channels=nChannels, kernel_size=1, stride=1, padding=0, bias=False)

        LEBs = []
        for sas_index in range(nLayers):
            LEBs.append(ESAS(nChannels))
        self.LEBs = torch.nn.ModuleList(LEBs)

    def forward(self, x1, x2, msk):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        msk = rearrange(msk, 'b c ah aw h w -> (b ah aw) c h w')

        out = self.conv_f(torch.cat([x1, x2, msk], dim=1))
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)

        for ix in range(len(self.LEBs)):
            out = self.LEBs[ix](out)
        return out


# Spatial Attention block (SAB)
class SAB(nn.Module):
    def __init__(self, nChannels):
        super(SAB, self).__init__()
        self.sab = nn.Sequential(
            nn.Conv2d(in_channels=nChannels*2, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        attn = self.sab(torch.cat([x1, x2], dim=1))
        out = attn * x1
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Spatial Feature Fusion (SFF)
class SFF(nn.Module):
    def __init__(self, nChannels):
        super(SFF, self).__init__()
        # self.sff = nn.Conv2d(in_channels=nChannels*2, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self.sff = nn.Sequential(
            nn.Conv2d(in_channels=nChannels*2, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nChannels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x1, x2):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x1.shape
        x1 = rearrange(x1, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        x2 = rearrange(x2, 'b c ah aw h w -> (b ah aw) c h w')
        attn = self.sff(torch.cat([x1, x2], dim=1))
        attn = attn.softmax(dim=1)         # [b*an2,2,h,w]
        out = attn[:, 0, :, :].unsqueeze(1) * x1 + attn[:, 1, :, :].unsqueeze(1) * x2
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out


# Image Reconstruction Block (IRB)
class IRB(nn.Module):
    def __init__(self, nChannels):
        super(IRB, self).__init__()
        self.irb = nn.Sequential(
            nn.Conv2d(in_channels=nChannels, out_channels=nChannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nChannels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [b,c,ah,aw,h,w]
        in_b, _, in_ah, in_aw, _, _ = x.shape
        x = rearrange(x, 'b c ah aw h w -> (b ah aw) c h w')     # [b*an2,c,h,w]
        out = self.irb(x)
        out = rearrange(out, '(b ah aw) c h w -> b c ah aw h w', b=in_b, ah=in_ah, aw=in_aw)
        return out
