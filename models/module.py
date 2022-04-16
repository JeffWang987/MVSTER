import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import local_pcd
from modules.deform_conv import DeformConvPack


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class DCNConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(DCNConv2d, self).__init__()

        self.conv = DeformConvPack(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=(not bn), im2col_step=16)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class PConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, padding=1, init_method="xavier", **kwargs):
        super(PConv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size_xy = (1, kernel_size, kernel_size)
        self.kernel_size_d = (kernel_size, 1, 1)
        assert stride in [1, 2]
        self.stride_xy = (1, stride, stride)
        self.stride_d = (stride, 1, 1)
        self.padding_xy = (0, padding, padding)
        self.padding_d = (padding, 0, 0)

        self.convxy = nn.Conv3d(in_channels, in_channels, self.kernel_size_xy, stride=self.stride_xy, padding=self.padding_xy, bias=(not bn), **kwargs)
        self.convd = nn.Conv3d(in_channels, out_channels, self.kernel_size_d, stride=self.stride_d, padding=self.padding_d, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.convxy(x)
        x = self.convd(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.convxy, init_method)
        init_uniform(self.convd, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class PDeconv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,output_padding=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(PDeconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride
        self.kernel_size_xy = (1, kernel_size,kernel_size)
        self.kernel_size_d = (kernel_size, 1,1)
        self.stride_xy = (1, stride, stride)
        self.stride_d = (stride, 1, 1)
        self.padding_xy = (0, padding, padding)
        self.padding_d = (padding, 0, 0)
        self.outpadding_xy = (0, output_padding, output_padding)
        self.outpadding_d = (output_padding, 0, 0)
        self.convxy = nn.ConvTranspose3d(in_channels, in_channels, self.kernel_size_xy, stride=self.stride_xy, padding=self.padding_xy, output_padding=self.outpadding_xy, bias=(not bn))
        self.convd = nn.ConvTranspose3d(in_channels, out_channels, self.kernel_size_d, stride=self.stride_d, padding=self.padding_d, output_padding=self.outpadding_d, bias=(not bn))
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.convxy(x)
        y = self.convd(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.convxy, init_method)
        init_uniform(self.convd, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values, align_corners=False):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear', padding_mode='zeros', align_corners=align_corners)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs

class FPNDCNpath(nn.Module):
    """
    FPN+DCN pathway"""
    def __init__(self, base_channels, stride=4):
        super(FPNDCNpath, self).__init__()
        self.stride = stride
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out_channels = [4 * base_channels]

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 2, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 2, base_channels * 2, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 2, base_channels * 2, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out2pathconv = nn.Conv2d(base_channels * 4, base_channels * 2, 3,  stride=1, padding=1)
        self.out3 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 1, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 1, base_channels * 1, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 1, base_channels * 1, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out3pathconv = nn.Conv2d(base_channels * 2, base_channels * 1, 3,  stride=1, padding=1)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv1)
        out2 = self.out2(intra_feat)
        out2 = out2 + self.out2pathconv(F.interpolate(out1, scale_factor=2, mode="bilinear", align_corners=True))

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv0)
        out3 = self.out3(intra_feat)
        out3 = out3 + self.out3pathconv(F.interpolate(out2, scale_factor=2, mode="bilinear", align_corners=True))

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3

        return outputs

class FPNDCN(nn.Module):
    """
    FPN+DCN"""
    def __init__(self, base_channels, stride=4):
        super(FPNDCN, self).__init__()
        self.stride = stride
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 4, base_channels * 4, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out_channels = [4 * base_channels]

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 2, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 2, base_channels * 2, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 2, base_channels * 2, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out3 = nn.Sequential(
            DCNConv2d(base_channels * 4, base_channels * 1, 3,  stride=1, padding=1),
            DCNConv2d(base_channels * 1, base_channels * 1, 3,  stride=1, padding=1),
            DeformConvPack(base_channels * 1, base_channels * 1, 3,  stride=1, padding=1, bias=False, im2col_step=16)
        )
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv1)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv0)
        out3 = self.out3(intra_feat)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3

        return outputs

class FPNA(nn.Module):
    """
    FPN aligncorners"""
    def __init__(self, base_channels, stride=4):
        super(FPNA, self).__init__()
        self.stride = stride
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv1)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv0)
        out3 = self.out3(intra_feat)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3

        return outputs

class FPNA4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels):
        super(FPNA4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.out_channels.append(base_channels * 4)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        outputs = {}
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv2)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv1)
        out3 = self.out3(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv0)
        out4 = self.out4(intra_feat)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3
        outputs["stage4"] = out4

        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(CostRegNet, self).__init__()
        self.down_size = down_size
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        if down_size >= 2:
            self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
            self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        if down_size >= 3:
            self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
            self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)
            self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        if down_size >= 2:
            self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
            
        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)
            x = self.prob(x)
        return x

class P3DConv(nn.Module):
    """
    Pseudo 3D conv: 3x3x1 + 1x3x3
    """
    def __init__(self, in_channels, base_channels):
        super(P3DConv, self).__init__()
        self.conv0 = PConv3d(in_channels, base_channels, padding=1)

        self.conv1 = PConv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = PConv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = PConv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = PConv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = PConv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = PConv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = PDeconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = PDeconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = PDeconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth

def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss

def cas_mvsnet_T_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    l1ce_lw = kwargs.get("l1ce_lw", [0.1, 1])
    range_thres = kwargs.get("range_thres", [84.8, 10.6])
    cas_method = kwargs.get("cascade_method", None)
    last_conv3d = kwargs.get("last_conv3d", False)
    visual = kwargs.get("visual", False)
    wt = kwargs.get("wt", False)
    fl = kwargs.get("fl", False)
    shrink_method = kwargs.get("shrink_method", 'schedule')
    upsampled_loss = kwargs.get("upsampled_loss", False)
    selected_loss = kwargs.get("selected_loss", False)
    mask_range_loss = kwargs.get("mask_range_loss", False)
    det = kwargs.get("det", False)
    if visual:
        f, axs = plt.subplots(figsize=(30, 10),ncols=3)  # depth offset
        f2, axs2 = plt.subplots(figsize=(30, 10),ncols=3)  # attn weight max
        f3, axs3 = plt.subplots(figsize=(30, 10),ncols=3)  # attn weight gt val
        f4, axs4 = plt.subplots(figsize=(30, 10),ncols=3)  # max gt offset
        err_848_str = ''
        err_106_str = ''
        err_002_str = ''

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_depth_loss = []
    stage_ce_loss = []
    range_err_ratio = []
    upsampled_depth_losses = []
    det_offset_losses = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_est = stage_inputs["depth"]
        B,H,W = depth_est.shape
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        if upsampled_loss:
            if stage_idx!=0 :
                upsampled_depth = stage_inputs["upsampled_depth"]
                upsampled_depth_loss = F.smooth_l1_loss(upsampled_depth[mask], depth_gt[mask], reduction='mean')
                upsampled_depth_losses.append(upsampled_depth_loss)
        else:
            if stage_idx!=0 :
                upsampled_depth_losses.append(torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False))
        
        if mask_range_loss:
            if stage_idx != 0:
                depth_offset = next_stage_depth_hypo - depth_gt  # B H W
                this_stage_mask_range = torch.abs(depth_offset)<range_thres[stage_idx-1]
                mask = mask & this_stage_mask_range  # B H W
            next_stage_depth_hypo = F.interpolate(depth_est.unsqueeze(1), scale_factor=2, mode='bilinear', align_corners=True).squeeze(1)
            

        if stage_idx != len(range_thres):
            depth_offset = depth_est - depth_gt
            depth_offset[~mask] = 0
            depth_offset = depth_offset # B H W
            range_err_ratio.append((torch.abs(depth_offset)>range_thres[stage_idx]).float().mean())


        if visual:
            depth_offset = depth_est - depth_gt
            depth_offset[~mask] = 0
            depth_offset = depth_offset.detach().cpu().numpy()[0] # H W  
            err_848_str += str((np.abs(depth_offset)>84.8).sum()) + ','
            err_106_str += str((np.abs(depth_offset)>10.6).sum()) + ','
            err_002_str += str((np.abs(depth_offset)>2).sum()) + ','
            sns.heatmap(depth_offset, annot=False, ax=axs[stage_idx])

            attn_weights = stage_inputs["attn_weights"][0]  # D H W
            attn_weights_max, ind_max = torch.max(attn_weights, 0)
            attn_weights_max = attn_weights_max.detach().cpu().numpy()  # H W
            sns.heatmap(attn_weights_max, annot=False, ax=axs2[stage_idx])

            this_stage_depth_val = stage_inputs['depth_values']  # B D H W
            depth_offsets = torch.abs(this_stage_depth_val- depth_gt[:,None,:,:])[0]  # D,H,W
            _, indices = torch.min(depth_offsets, dim=0, keepdim=True)  # [1, H, W]
            attn_gt = torch.gather(attn_weights, 0, indices)[0]  # [H W]
            attn_gt = attn_gt.detach().cpu().numpy()
            sns.heatmap(attn_gt, annot=False, ax=axs3[stage_idx])

            max_gt_offset = ind_max - indices[0]  # H W
            max_gt_offset = max_gt_offset.detach().cpu().numpy()
            sns.heatmap(max_gt_offset, annot=False, ax=axs4[stage_idx])

        if cas_method[stage_idx] == 't' or cas_method[stage_idx] == 'r' or cas_method[stage_idx] == 'p':
            # Loss for transformer 
            depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
            if last_conv3d:
                depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
            attn_weights = stage_inputs["attn_weights"].permute(0,2,3,1).reshape(B*H*W, -1)  # BHW D
            this_stage_depth_val = stage_inputs['depth_values']  # B D H W
            depth_offsets = torch.abs(this_stage_depth_val- depth_gt[:,None,:,:])  # B,D,H,W
            _, indices = torch.min(depth_offsets, dim=1)  # [B, H, W]
            indices = indices.reshape(-1)  # [BHW]
            mask = mask.reshape(-1)  # BHW
            if fl:  # -p(1-q)^a log(q)
                this_stage_ce_loss = F.nll_loss((1-attn_weights[mask])**2 * torch.log(attn_weights[mask]+1e-12), indices[mask], reduce='mean')
            else:  # -plog(q)
                this_stage_ce_loss = F.nll_loss(torch.log(attn_weights[mask]+1e-12), indices[mask], reduce='mean')
            stage_depth_loss.append(depth_loss)
            stage_ce_loss.append(this_stage_ce_loss)

            this_stage_loss = l1ce_lw[0]*depth_loss + l1ce_lw[1]*this_stage_ce_loss
        
        # Loss for 3D conv
        else: 
            if wt:
                depth_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
                stage_depth_loss.append(depth_loss)
                attn_weights = stage_inputs["attn_weights"].permute(0,2,3,1).reshape(B*H*W, -1)  # BHW D
                depth_offsets = torch.abs(stage_inputs['depth_values']- depth_gt[:,None,:,:])  # B,D,H,W
                indices = torch.min(depth_offsets, dim=1)[1].reshape(-1)  # [BHW]
                mask = mask.reshape(-1)  # BHW
                if fl:  # -p(1-q)^a log(q)
                    this_stage_ce_loss = F.nll_loss((1-attn_weights[mask])**2 * torch.log(attn_weights[mask]+1e-12), indices[mask], reduce='mean')
                else:  # -plog(q)
                    this_stage_ce_loss = F.nll_loss(torch.log(attn_weights[mask]+1e-12), indices[mask], reduce='mean')
                stage_ce_loss.append(this_stage_ce_loss)
            else:
                depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
                stage_depth_loss.append(depth_loss)
                this_stage_ce_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
                stage_ce_loss.append(this_stage_ce_loss)

            this_stage_loss = l1ce_lw[0]*depth_loss + l1ce_lw[1]*this_stage_ce_loss
        
        if upsampled_loss:
            if stage_idx!=0:
                this_stage_loss = this_stage_loss + upsampled_depth_loss * l1ce_lw[0]
        if shrink_method == 'DPF':
            if stage_idx!=0:
                depth_offsets = stage_inputs['depth_values'] - depth_gt[:,None,:,:]  # B,D,H,W
                depth_offset_clamp = torch.clamp(depth_offsets, -1, 1)
                this_stage_loss = this_stage_loss + torch.abs(depth_offset_clamp).permute(0,2,3,1).reshape(B*H*W, -1)[mask.reshape(-1)].mean()
        if selected_loss:
            select_weight = stage_inputs["select_weight"].permute(0,2,3,1).reshape(B*H*W, -1)  # BHW D
            depth_offsets = torch.abs(stage_inputs['depth_values']- depth_gt[:,None,:,:]) 
            indices = torch.min(depth_offsets, dim=1)[1]  # [B, H, W]
            indices = indices.reshape(-1)  # [BHW]
            mask = mask.reshape(-1)  # BHW
            this_stage_selected_loss = F.nll_loss(torch.log(select_weight[mask]+1e-12), indices[mask], reduce='mean')
            this_stage_loss = this_stage_loss + this_stage_selected_loss * 0.01*l1ce_lw[1]
        if det:
            assert wt
            depth_itv = stage_inputs['depth_values'][:,1,:,:] - stage_inputs['depth_values'][:,0,:,:]   # B H W
            pred_offset = stage_inputs['offset_reg'].reshape(-1)  # BHW
            offset_gt = (depth_gt - (depth_est - stage_inputs['offset_reg'])).reshape(-1) / depth_itv.reshape(-1) # BHW
            det_offset_loss = F.smooth_l1_loss(pred_offset[mask], offset_gt[mask], reduction='mean')
            det_offset_losses.append(det_offset_loss)
            this_stage_loss += det_offset_loss
        else:
            det_offset_losses.append(torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False))

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * this_stage_loss
        else:
            total_loss += 1.0 * this_stage_loss

    if visual:
        axs[1].set_title('err848:{}'.format(err_848_str) + 'err_106:{}'.format(err_106_str) + 'err_002:{}'.format(err_002_str))
        f.savefig('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/cascade-stereo/CasMVSNet/debug_figs/offset_heatmap.png')
        f.clf()
        f2.savefig('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/cascade-stereo/CasMVSNet/debug_figs/attn_max_heatmap.png')
        f2.clf()
        f3.savefig('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/cascade-stereo/CasMVSNet/debug_figs/attn_gt_heatmap.png')
        f3.clf()
        f4.savefig('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/cascade-stereo/CasMVSNet/debug_figs/max_gt_offset_heatmap.png')
        f4.clf()

    return total_loss, depth_loss, stage_depth_loss, stage_ce_loss, range_err_ratio, upsampled_depth_losses, det_offset_losses


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples



if __name__ == "__main__":
    # some testing code, just IGNORE it
    import sys
    sys.path.append("../")
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    # MVSDataset = find_dataset_def("colmap")
    # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    #                      128, interval_scale=1.06, max_h=1250, max_w=1024)

    MVSDataset = find_dataset_def("dtu_yao")
    num_depth = 48
    dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, num_depth, interval_scale=1.06 * 192 / num_depth)

    dataloader = DataLoader(dataset, batch_size=1)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # imgs = item["imgs"][:, :, :, :, :]
    proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    depth_values = item["depth_values"]     #(B, D)

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view

    src_proj_new = src_proj[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])

    warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)

    ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    cv2.imwrite('../tmp/ref.png', ref_img_np)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        img_np = img_np[:, :, ::-1] * 255

        alpha = 0.5
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
        cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))