import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
try:
    from modules.deform_conv import DeformConvPack
except:
    print('DeformConvPack not found, please install it from: https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch')
    pass
import math
import numpy as np

def homo_warping(src_fea, src_proj, ref_proj, depth_values, vis_ETA=False, fn=None):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    C = src_fea.shape[1]
    Hs,Ws = src_fea.shape[-2:]
    B,num_depth,Hr,Wr = depth_values.shape

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, Hr, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, Wr, dtype=torch.float32, device=src_fea.device)])
        y = y.reshape(Hr*Wr)
        x = x.reshape(Hr*Wr)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.reshape(B, 1, num_depth, -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.reshape(B, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # FIXME divide 0
        temp = proj_xyz[:, 2:3, :, :]
        temp[temp==0] = 1e-9
        proj_xy = proj_xyz[:, :2, :, :] / temp  # [B, 2, Ndepth, H*W]
        # proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]

        proj_x_normalized = proj_xy[:, 0, :, :] / ((Ws - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((Hs - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        if vis_ETA:
            tensor_saved = proj_xy.reshape(B,num_depth,Hs,Ws,2).cpu().numpy()
            np.save(fn+'_grid', tensor_saved)
        grid = proj_xy
    if len(src_fea.shape)==4:
        warped_src_fea = F.grid_sample(src_fea, grid.reshape(B, num_depth * Hr, Wr, 2), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_src_fea = warped_src_fea.reshape(B, C, num_depth, Hr, Wr)
    elif len(src_fea.shape)==5:
        warped_src_fea = []
        for d in range(src_fea.shape[2]):
            warped_src_fea.append(F.grid_sample(src_fea[:,:,d], grid.reshape(B, num_depth, Hr, Wr, 2)[:,d], mode='bilinear', padding_mode='zeros', align_corners=True))
        warped_src_fea = torch.stack(warped_src_fea, dim=2)

    return warped_src_fea

def init_range(cur_depth, ndepths, device, dtype, H, W):
    cur_depth_min = cur_depth[:, 0]  # (B,)
    cur_depth_max = cur_depth[:, -1]
    new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
    new_interval = new_interval[:, None, None]  # B H W
    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                requires_grad=False).reshape(1, -1) * new_interval.squeeze(1)) #(B, D)
    depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) #(B, D, H, W)
    return depth_range_samples

def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
    inverse_depth_max = 1. / cur_depth[:, -1]
    itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H, W)  / (ndepths - 1)  # 1 D H W
    inverse_depth_hypo = inverse_depth_max[:,None, None, None] + (inverse_depth_min - inverse_depth_max)[:,None, None, None] * itv

    return 1./inverse_depth_hypo

def schedule_inverse_range(inverse_min_depth, inverse_max_depth, ndepths, H, W):
    #cur_depth_min, (B, H, W)
    #cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype, requires_grad=False).reshape(1, -1,1,1).repeat(1, 1, H//2, W//2)  / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:,None, :, :] + (inverse_min_depth - inverse_max_depth)[:,None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1./inverse_depth_hypo

def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel[:,None,None])  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel[:,None,None])
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples

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

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D_CAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_CAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D*H*W).mean(2))
        max_attn = self.linear_agg(x.reshape(B,C,D*H*W).max(2)[0])  # B C
        attn = F.sigmoid(max_attn+avg_attn)[:,:,None,None,None]  # B C,1,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_DCAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_DCAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.linear_agg = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.ReLU(),
            nn.Linear(out_channels//2, out_channels)
        )

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        avg_attn = self.linear_agg(x.reshape(B,C,D,H*W).mean(3).permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)
        max_attn = self.linear_agg(x.reshape(B,C,D,H*W).max(3)[0].permute(0,2,1).reshape(B*D,C)).reshape(B,D,C).permute(0,2,1)  # B C D
        attn = F.sigmoid(max_attn+avg_attn)[:,:,:,None,None]  # B C,D,1,1
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.pixel_conv = nn.Conv2d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.reshape(B,C*D,H,W).max(1, keepdim=True)[0]
        avg_attn = x.reshape(B,C*D,H,W).mean(1, keepdim=True)  # B 1 H W
        attn = F.sigmoid(self.pixel_conv(torch.cat([max_attn, avg_attn], dim=1)))[:,:,None,:,:]  # B 1,1,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class ConvBnReLU3D_PDAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D_PDAM, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.spatial_conv = nn.Conv3d(2,1,7,stride=1,padding='same')

    def forward(self, input):
        x = self.conv(input)
        B,C,D,H,W = x.shape
        max_attn = x.max(1, keepdim=True)[0]
        avg_attn = x.mean(1, keepdim=True)  # B 1 D H W
        attn = F.sigmoid(self.spatial_conv(torch.cat([max_attn, avg_attn], dim=1)))  # B 1,D,H,W
        x = x * attn
        return F.relu(self.bn(x+input), inplace=True)

class Deconv3d(nn.Module):

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

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn_momentum=0.1, init_method="xavier", gn=False, group_channel=8, **kwargs):
        super(Conv2d, self).__init__()
        bn = not gn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(int(max(1, out_channels / group_channel)), out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        else:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

def NA_DCN(in_channels, kernel_size=3, stride=1, dilation=1, bias=True, group_channel=8, gn=False):
    if gn:
        return nn.Sequential(
            nn.GroupNorm(int(max(1, in_channels / group_channel)), in_channels),
            nn.ReLU(inplace=True),
            # DeformConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,  bias=bias),
            DeformConvPack(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=1, deformable_groups=1, bias=False, im2col_step=16)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            # DeformConv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,  bias=bias),
            DeformConvPack(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=1, deformable_groups=1, bias=False, im2col_step=16)
        )

class FPN4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, gn=False, dcn=False):
        super(FPN4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1, gn=gn),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1, gn=gn),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1, gn=gn),
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

        self.dcn = dcn
        if self.dcn:
            self.dcn1 = NA_DCN(base_channels * 8, 3, gn=gn)
            self.dcn2 = NA_DCN(base_channels * 4, 3, gn=gn)
            self.dcn3 = NA_DCN(base_channels * 2, 3, gn=gn)
            self.dcn4 = NA_DCN(base_channels * 1, 3, gn=gn)

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

        if self.dcn:
            out1 = self.dcn1(out1)
            out2 = self.dcn2(out2)
            out3 = self.dcn3(out3)
            out4 = self.dcn4(out4)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3
        outputs["stage4"] = out4

        return outputs

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class convnext_block(nn.Module):

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, 2*dim, kernel_size=7, stride=2, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(2*dim, eps=1e-6)
        self.pwconv1 = nn.Linear(2*dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, 2*dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((2*dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        # x = input + x
        return x

class convnext4_block(nn.Module):

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.sconv = nn.Conv2d(dim, 2*dim, kernel_size=2, stride=2, padding=0) # stride=2 conv
        self.dwconv = nn.Conv2d(2*dim, 2*dim, kernel_size=7, stride=1, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(2*dim, eps=1e-6)
        self.pwconv1 = nn.Linear(2*dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, 2*dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((2*dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = self.sconv(x)
        x = self.dwconv(input)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

class FPN4_convnext(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, gn=False, dcn=False):
        super(FPN4_convnext, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = convnext_block(base_channels)
        self.conv2 = convnext_block(2*base_channels)
        self.conv3 = convnext_block(4*base_channels)

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.dcn = dcn
        if self.dcn:
            self.dcn1 = NA_DCN(base_channels * 8, 3, gn=gn)
            self.dcn2 = NA_DCN(base_channels * 4, 3, gn=gn)
            self.dcn3 = NA_DCN(base_channels * 2, 3, gn=gn)
            self.dcn4 = NA_DCN(base_channels * 1, 3, gn=gn)

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

        if self.dcn:
            out1 = self.dcn1(out1)
            out2 = self.dcn2(out2)
            out3 = self.dcn3(out3)
            out4 = self.dcn4(out4)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3
        outputs["stage4"] = out4

        return outputs

class FPN4_convnext4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, gn=False, dcn=False):
        super(FPN4_convnext4, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1, gn=gn),
            Conv2d(base_channels, base_channels, 3, 1, padding=1, gn=gn),
        )

        self.conv1 = convnext4_block(base_channels)
        self.conv2 = convnext4_block(2*base_channels)
        self.conv3 = convnext4_block(4*base_channels)

        self.out_channels = [8 * base_channels]
        final_chs = base_channels * 8

        self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out1 = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        self.out2 = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.out4 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)

        self.dcn = dcn
        if self.dcn:
            self.dcn1 = NA_DCN(base_channels * 8, 3, gn=gn)
            self.dcn2 = NA_DCN(base_channels * 4, 3, gn=gn)
            self.dcn3 = NA_DCN(base_channels * 2, 3, gn=gn)
            self.dcn4 = NA_DCN(base_channels * 1, 3, gn=gn)

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

        if self.dcn:
            out1 = self.dcn1(out1)
            out2 = self.dcn2(out2)
            out3 = self.dcn3(out3)
            out4 = self.dcn4(out4)

        outputs["stage1"] = out1
        outputs["stage2"] = out2
        outputs["stage3"] = out3
        outputs["stage4"] = out4

        return outputs
    
class ASFF(nn.Module):
    def __init__(self, level):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [64,32,16,8]
        self.inter_dim = self.dim[self.level]
        if level==0:
            self.stride_level_1 = Conv2d(32, 64, 3, stride=2, padding=1)
            self.stride_level_2 = Conv2d(16, 64, 3, stride=2, padding=1)
            self.stride_level_3 = Conv2d(8, 64, 3, stride=2, padding=1)
            self.expand = Conv2d(64, 64, 3, stride=1, padding=1)
        elif level==1:
            self.compress_level_0 =  Conv2d(64, 32, 1, stride=1, padding=0)
            self.stride_level_2 = Conv2d(16, 32, 3, stride=2, padding=1)
            self.stride_level_3 = Conv2d(8, 32, 3, stride=2, padding=1)
            self.expand = Conv2d(32, 32, 3, stride=1, padding=1)
        elif level==2:
            self.compress_level_0 = Conv2d(64, 16, 1, stride=1, padding=0)
            self.compress_level_1 = Conv2d(32, 16, 1, stride=1, padding=0)
            self.stride_level_3 = Conv2d(8, 16, 3, stride=2, padding=1)
            self.expand = Conv2d(16, 16, 3, stride=1, padding=1)
        elif level==3:
            self.compress_level_0 = Conv2d(64, 8, 1, stride=1, padding=0)
            self.compress_level_1 = Conv2d(32, 8, 1, stride=1, padding=0)
            self.compress_level_2 = Conv2d(16, 8, 1, stride=1, padding=0)
            self.expand = Conv2d(8, 8, 3, stride=1, padding=1)

        self.weight_level_0 = Conv2d(self.dim[level], 8, 1, 1, 0)
        self.weight_level_1 = Conv2d(self.dim[level], 8, 1, 1, 0)
        self.weight_level_2 = Conv2d(self.dim[level], 8, 1, 1, 0)
        self.weight_level_3 = Conv2d(self.dim[level], 8, 1, 1, 0)

        self.weight_levels = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0)


    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 2, stride=2, padding=0)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            level_3_downsampled_inter = F.max_pool2d(x_level_3, 4, stride=4, padding=0)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_downsampled_inter = F.max_pool2d(x_level_3, 2, stride=2, padding=0)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)
        elif self.level==3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')
            level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')
            level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+\
                            level_3_resized * levels_weight[:,3:,:,:]

        out = self.expand(fused_out_reduced)

        return out

class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)
        self.h = h // kernel_size + 1
        self.w = w // kernel_size + 1
        # print("h=", self.h, " w=", self.w, h, w)
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 卷积

    def forward(self, x):
        # print('x size:', x.size())
        x1 = self.global_pooling(x)
        # print('# x1 size:', x1.size())
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * self.h * self.w)  # kitti 4x5
        x4 = self.relu(self.global_fc(x3))
        # print('# x4 size:', x4.size())
        x4 = x4.view(-1, 512, 1, 1)
        # print('# x4 size:', x4.size())
        x5 = self.conv1(x4)
        # out = self.upsample(x5)
        return x5

class mono_depth_decoder(nn.Module):

    def __init__(self):
        super(mono_depth_decoder, self).__init__()
        self.convblocks = nn.ModuleList(
            [Conv2d(64, 32, 3, 1, padding=1),
            Conv2d(32, 16, 3, 1, padding=1),
            Conv2d(16, 8, 3, 1, padding=1)]
        )
        self.conv3x3 = nn.ModuleList(
           [nn.Conv2d(64, 1, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Conv2d(16, 1, 3, 1, 1)]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, d_min, d_max):
        """
        d_max: B
        """
        for i in range(1,4):  # 1 2 3
            mono_small_feat = outputs['stage{}'.format(i)]['mono_feat']
            mono_large_feat = outputs['stage{}'.format(i+1)]['mono_feat']

            mono_small_feat = self.convblocks[i-1](mono_small_feat)
            mono_small_feat = F.interpolate(mono_small_feat, scale_factor=2, mode="nearest")

            mono_feat = self.conv3x3[i-1](torch.cat([mono_small_feat, mono_large_feat], 1))  # B C H W

            disp = self.sigmoid(mono_feat)
            min_disp = (1 / d_max)[:,None,None,None]  # B 1 1 1
            max_disp = (1 / d_min)[:,None,None,None]
            scaled_disp = min_disp + (max_disp - min_disp) * disp
            depth = 1 / scaled_disp
            outputs['stage{}'.format(i+1)]['mono_depth'] = depth.squeeze(1)
        return outputs

class reg2d(nn.Module):
    def __init__(self, input_channel=128, base_channel=32, conv_name='ConvBnReLU3D'):
        super(reg2d, self).__init__()
        module = importlib.import_module("models.mvs4net_utils")
        stride_conv_name = 'ConvBnReLU3D'
        self.conv0 = getattr(module, stride_conv_name)(input_channel, base_channel, kernel_size=(1,3,3), pad=(0,1,1))
        self.conv1 = getattr(module, stride_conv_name)(base_channel, base_channel*2, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv2 = getattr(module, conv_name)(base_channel*2, base_channel*2)

        self.conv3 = getattr(module, stride_conv_name)(base_channel*2, base_channel*4, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv4 = getattr(module, conv_name)(base_channel*4, base_channel*4)

        self.conv5 = getattr(module, stride_conv_name)(base_channel*4, base_channel*8, kernel_size=(1,3,3), stride=(1,2,2), pad=(0,1,1))
        self.conv6 = getattr(module, conv_name)(base_channel*8, base_channel*8)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*8, base_channel*4, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*4, base_channel*2, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel*2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel*2, base_channel, kernel_size=(1,3,3), padding=(0,1,1), output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 1, stride=1, padding=0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x.squeeze(1)

class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
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
        return x.squeeze(1)  # B D H W

class PosEncSine(nn.Module):

    def __init__(self, temperature=1000):
        super(PosEncSine, self).__init__()
        self.temperature = temperature

    def forward(self, x, depth):
        # depth : B D H W
        with torch.no_grad():
            B,C,D,H,W = x.shape
            depth = depth.permute(0,2,3,1).reshape(B*H*W, D) / self.temperature  # BHW D
            pos = torch.stack([torch.sin(i * math.pi * depth) for i in range(C//2)] + [torch.cos(i * math.pi * depth) for i in range(C//2)], dim=-1)  # BHW,D,C
            pos = pos.reshape(B,H,W,D,C).permute(0,4,3,1,2)  # B C D H W
        x = x + pos
        return x

class PosEncLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, D, C):
        super().__init__()
        self.D = D
        self.C = C
        self.depth_embed = nn.Parameter(torch.Tensor(C, self.D))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.depth_embed)

    def forward(self, x, **kwargs):
        B,C,D,H,W = x.shape
        pos = self.depth_embed[None,:,:,None,None].repeat(B,1,1,H,W)  # B C D H W
        x = x + pos
        return x

class stagenet(nn.Module):
    def __init__(self, inverse_depth=False, mono=False, attn_fuse_d=True, vis_ETA=False, attn_temp=1):
        super(stagenet, self).__init__()
        self.inverse_depth = inverse_depth
        self.mono = mono
        self.attn_fuse_d = attn_fuse_d
        self.vis_ETA = vis_ETA
        self.attn_temp = attn_temp

    def forward(self, features, proj_matrices, depth_hypo, regnet, stage_idx, group_cor=False, group_cor_dim=8, split_itv=1, fn=None):

        # step 1. feature extraction
        proj_matrices = torch.unbind(proj_matrices, 1)
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B,D,H,W = depth_hypo.shape
        C = ref_feature.shape[1]

        ref_volume =  ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)
        cor_weight_sum = 1e-8
        cor_feats = 0
        # step 2. Epipolar Transformer Aggregation
        for src_idx, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)):
            if self.vis_ETA:
                scan_name = fn[0].split('/')[0]
                image_name = fn[0].split('/')[2][:-2]
                save_fn = './debug_figs/vis_ETA/{}_stage{}_src{}'.format(scan_name+'_'+image_name, stage_idx, src_idx)
            else:
                save_fn = None
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_src = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_hypo, self.vis_ETA, save_fn)  # B C D H W
            if group_cor:
                warped_src = warped_src.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                ref_volume = ref_volume.reshape(B, group_cor_dim, C//group_cor_dim, D, H, W)
                cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            else:
                cor_feat = (ref_volume - warped_src)**2 # B C D H W 
            del warped_src, src_proj, src_fea
            if self.vis_ETA:
                vis_weight = torch.softmax(cor_feat.sum(1), 1).detach().cpu().numpy()
                np.save(save_fn, vis_weight)

            if not self.attn_fuse_d:
                cor_weight = torch.softmax(cor_feat.sum(1), 1).max(1)[0]  # B H W
                cor_weight_sum += cor_weight  # B H W
                cor_feats += cor_weight.unsqueeze(1).unsqueeze(1) * cor_feat  # B C D H W
            else:
                cor_weight = torch.softmax(cor_feat.sum(1) / self.attn_temp, 1) / math.sqrt(C)  # B D H W
                cor_weight_sum += cor_weight  # B D H W
                cor_feats += cor_weight.unsqueeze(1) * cor_feat  # B C D H W
            del cor_weight, cor_feat
        if not self.attn_fuse_d:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1).unsqueeze(1)  # B C D H W
        else:
            cor_feats = cor_feats / cor_weight_sum.unsqueeze(1)  # B C D H W

        del cor_weight_sum, src_features
        
    
        # step 3. regularization
        attn_weight = regnet(cor_feats)  # B D H W
        del cor_feats
        attn_weight = F.softmax(attn_weight, dim=1)  # B D H W

        # step 4. depth argmax
        attn_max_indices = attn_weight.max(1, keepdim=True)[1]  # B 1 H W
        depth = torch.gather(depth_hypo, 1, attn_max_indices).squeeze(1)  # B H W

        if not self.training:
            with torch.no_grad():
                photometric_confidence = attn_weight.max(1)[0]  # B H W
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1), scale_factor=2**(3-stage_idx), mode='bilinear', align_corners=True).squeeze(1)
        else:
            photometric_confidence = torch.tensor(0.0, dtype=torch.float32, device=ref_feature.device, requires_grad=False)
        
        ret_dict = {"depth": depth,  "photometric_confidence": photometric_confidence, "hypo_depth": depth_hypo, "attn_weight": attn_weight}
        
        if self.inverse_depth:
            last_depth_itv = 1./depth_hypo[:,2,:,:] - 1./depth_hypo[:,1,:,:]
            inverse_min_depth = 1/depth + split_itv * last_depth_itv  # B H W
            inverse_max_depth = 1/depth - split_itv * last_depth_itv  # B H W
            ret_dict['inverse_min_depth'] = inverse_min_depth
            ret_dict['inverse_max_depth'] = inverse_max_depth

        # if self.mono and self.training:
        if self.mono:
            ret_dict['mono_feat'] = ref_feature  # B C H W
            
        return ret_dict
 
def sinkhorn(gt_depth, hypo_depth, attn_weight, mask, iters, eps=1, continuous=False):
    """
    gt_depth: B H W
    hypo_depth: B D H W
    attn_weight: B D H W
    mask: B H W
    """
    B,D,H,W = attn_weight.shape
    if not continuous:
        D_map = torch.stack([torch.arange(-i,D-i,1, dtype=torch.float32, device=gt_depth.device) for i in range(D)], dim=1).abs()
        D_map = D_map[None,None,:,:].repeat(B,H*W,1,1)  # B HW D D
        gt_indices = torch.abs(hypo_depth - gt_depth[:,None,:,:]).min(1)[1].squeeze(1).reshape(B*H*W, 1)  # BHW, 1
        gt_dist = torch.zeros_like(hypo_depth).permute(0,2,3,1).reshape(B*H*W, D)
        gt_dist.scatter_add_(1,gt_indices,torch.ones([gt_dist.shape[0],1], dtype=gt_dist.dtype, device=gt_dist.device))
        gt_dist = gt_dist.reshape(B,H*W,D)  # B HW D
    else:
        gt_dist = torch.zeros((B,H*W,D+1), dtype=torch.float32, device=gt_depth.device, requires_grad=False)  # B HW D+1
        gt_dist[:,:,-1] = 1
        D_map = torch.zeros((B,D,D+1), dtype=torch.float32, device=gt_depth.device, requires_grad=False)  # B D D+1
        D_map[:, :D, :D] = torch.stack([torch.arange(-i,D-i,1, dtype=torch.float32, device=gt_depth.device) for i in range(D)], dim=1).abs().unsqueeze(0)  # B D D+1
        D_map = D_map[:,None,None,:,:].repeat(1,H,W,1,1)  # B H W D D+1
        itv = 1/hypo_depth[:,2,:,:] - 1/hypo_depth[:,1,:,:]  # B H W
        gt_bin_distance_ = (1/gt_depth - 1/hypo_depth[:,0,:,:]) / itv  # B H W
        #FIXME hard code 100
        gt_bin_distance_[~mask] = 10

        gt_bin_distance = torch.stack([(gt_bin_distance_ - i).abs() for i in range(D)], dim=1).permute(0,2,3,1)  # B H W D
        D_map[:,:,:,:,-1] = gt_bin_distance
        D_map = D_map.reshape(B,H*W,D,1+D)  # B HW D D+1

    pred_dist = attn_weight.permute(0,2,3,1).reshape(B,H*W,D)  # B HW D

    # map to log space for stability
    log_mu = (gt_dist+1e-12).log()
    log_nu = (pred_dist+1e-12).log()  # B HW D or D+1

    u, v = torch.zeros_like(log_nu), torch.zeros_like(log_mu)
    for _ in range(iters):
        # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
        v = log_mu - torch.logsumexp(D_map/eps + u.unsqueeze(3), dim=2)  # log(sum(exp()))
        u = log_nu - torch.logsumexp(D_map/eps + v.unsqueeze(2), dim=3)

    # convert back from log space, recover probabilities by normalization 2W
    T_map = (D_map/eps + u.unsqueeze(3) + v.unsqueeze(2)).exp()  # B HW D D
    loss = (T_map * D_map).reshape(B*H*W,-1)[mask.reshape(-1)].sum(-1).mean()
    
    return T_map, loss