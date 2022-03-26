import torch
from torch import nn
from torch.nn import functional as F

from models.layers.conv_bn_act import ConvNormAct
from .trace_utils import _assert


class BilinearAttnTransform(nn.Module):
    
    def __init__(self, in_channels, block_size, groups, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BilinearAttnTransform, self).__init__()

        self.conv1 = ConvNormAct(in_channels, groups, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_p = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(block_size, 1))
        self.conv_q = nn.Conv2d(groups, block_size * block_size * groups, kernel_size=(1, block_size))
        self.conv2 = ConvNormAct(in_channels, in_channels, 1, act_layer=act_layer, norm_layer=norm_layer)
        self.block_size = block_size
        self.groups = groups
        self.in_channels = in_channels

    def resize_mat(self, x, t: int):
        B, C, block_size, block_size1 = x.shape
        _assert(block_size == block_size1, '')
        if t <= 1:
            return x
        x = x.view(B * C, -1, 1, 1)
        x = x * torch.eye(t, t, dtype=x.dtype, device=x.device)
        x = x.view(B * C, block_size, block_size, t, t)
        x = torch.cat(torch.split(x, 1, dim=1), dim=3)
        x = torch.cat(torch.split(x, 1, dim=2), dim=4)
        x = x.view(B, C, block_size * t, block_size * t)
        return x

    def forward(self, x):
        _assert(x.shape[-1] % self.block_size == 0, '')
        _assert(x.shape[-2] % self.block_size == 0, '')
        B, C, H, W = x.shape
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.block_size, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.block_size))
        p = self.conv_p(rp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        q = self.conv_q(cp).view(B, self.groups, self.block_size, self.block_size).sigmoid()
        p = p / p.sum(dim=3, keepdim=True)
        q = q / q.sum(dim=2, keepdim=True)
        p = p.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        p = p.view(B, C, self.block_size, self.block_size)
        q = q.view(B, self.groups, 1, self.block_size, self.block_size).expand(x.size(
            0), self.groups, C // self.groups, self.block_size, self.block_size).contiguous()
        q = q.view(B, C, self.block_size, self.block_size)
        p = self.resize_mat(p, H // self.block_size)
        q = self.resize_mat(q, W // self.block_size)
        y = p.matmul(x)
        y = y.matmul(q)

        y = self.conv2(y)
        return y