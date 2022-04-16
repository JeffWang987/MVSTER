import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.mvs4net_utils import stagenet, reg2d, reg3d, FPN4, FPN4_convnext, FPN4_convnext4, PosEncSine, PosEncLearned, \
        init_range, schedule_range, init_inverse_range, schedule_inverse_range, sinkhorn, mono_depth_decoder, ASFF


class MVS4net(nn.Module):
    def __init__(self, arch_mode="fpn", reg_net='reg2d', num_stage=4, fpn_base_channel=8, 
                reg_channel=8, stage_splits=[8,8,4,4], depth_interals_ratio=[0.5,0.5,0.5,1],
                group_cor=False, group_cor_dim=[8,8,8,8],
                inverse_depth=False,
                agg_type='ConvBnReLU3D',
                dcn=False,
                pos_enc=0,
                mono=False,
                asff=False,
                attn_temp=2,
                attn_fuse_d=True,
                vis_ETA=False,
                vis_mono=False
                ):
        # pos_enc: 0 no pos enc; 1 depth sine; 2 learnable pos enc
        super(MVS4net, self).__init__()
        self.arch_mode = arch_mode
        self.num_stage = num_stage
        self.depth_interals_ratio = depth_interals_ratio
        self.group_cor = group_cor
        self.group_cor_dim = group_cor_dim
        self.inverse_depth = inverse_depth
        self.asff = asff
        if self.asff:
            self.asff = nn.ModuleList([ASFF(i) for i in range(num_stage)])
        self.attn_ob = nn.ModuleList()
        if arch_mode == "fpn":
            self.feature = FPN4(base_channels=fpn_base_channel, gn=False, dcn=dcn)
        self.vis_mono = vis_mono
        self.stagenet = stagenet(inverse_depth, mono, attn_fuse_d, vis_ETA, attn_temp)
        self.stage_splits = stage_splits
        self.reg = nn.ModuleList()
        self.pos_enc = pos_enc
        self.pos_enc_func = nn.ModuleList()
        self.mono = mono
        if self.mono:
            self.mono_depth_decoder = mono_depth_decoder()
        if reg_net == 'reg3d':
            self.down_size = [3,3,2,2]
        for idx in range(num_stage):
            if self.group_cor:
                in_dim = group_cor_dim[idx]
            else:
                in_dim = self.feature.out_channels[idx]
            if reg_net == 'reg2d':
                self.reg.append(reg2d(input_channel=in_dim, base_channel=reg_channel, conv_name=agg_type))
            elif reg_net == 'reg3d':
                self.reg.append(reg3d(in_channels=in_dim, base_channels=reg_channel, down_size=self.down_size[idx]))


    def forward(self, imgs, proj_matrices, depth_values, filename=None):
        depth_min = depth_values[:, 0].cpu().numpy()
        depth_max = depth_values[:, -1].cpu().numpy()
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(len(imgs)):  #imgs shape (B, N, C, H, W)
            img = imgs[nview_idx]
            features.append(self.feature(img))
        if self.vis_mono:
            scan_name = filename[0].split('/')[0]
            image_name = filename[0].split('/')[2][:-2]
            save_fn = './debug_figs/vis_mono/feat_{}'.format(scan_name+'_'+image_name)
            feat_ = features[-1]['stage4'].detach().cpu().numpy()
            np.save(save_fn, feat_)
        # step 2. iter (multi-scale)
        outputs = {}
        for stage_idx in range(self.num_stage):
            if not self.asff:
                features_stage = [feat["stage{}".format(stage_idx+1)] for feat in features]
            else:
                features_stage = [self.asff[stage_idx](feat['stage1'],feat['stage2'],feat['stage3'],feat['stage4']) for feat in features]

            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            B,C,H,W = features[0]['stage{}'.format(stage_idx+1)].shape

            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_hypo = init_inverse_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W)
                else:
                    depth_hypo = init_range(depth_values, self.stage_splits[stage_idx], img[0].device, img[0].dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_hypo = schedule_inverse_range(outputs_stage['inverse_min_depth'].detach(), outputs_stage['inverse_max_depth'].detach(), self.stage_splits[stage_idx], H, W)  # B D H W
                else:
                    depth_hypo = schedule_range(outputs_stage['depth'].detach(), self.stage_splits[stage_idx], self.depth_interals_ratio[stage_idx] * depth_interval, H, W)

            outputs_stage = self.stagenet(features_stage, proj_matrices_stage, depth_hypo=depth_hypo, regnet=self.reg[stage_idx], stage_idx=stage_idx,
                                        group_cor=self.group_cor, group_cor_dim=self.group_cor_dim[stage_idx],
                                        split_itv=self.depth_interals_ratio[stage_idx],
                                        fn=filename)

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)
        
        if self.mono and self.training:
        # if self.mono:
            outputs = self.mono_depth_decoder(outputs, depth_values[:,0], depth_values[:,1])

        return outputs

def MVS4net_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        B,H,W = depth_pred.shape
        D = hypo_depth.shape[1]
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]

        if mono and stage_idx!=0:
            this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
        else:
            this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        # mask range
        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_l1_loss.append(this_stage_l1_loss)
        stage_ot_loss.append(this_stage_ot_loss)
        total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[0] * this_stage_l1_loss + l1ot_lw[1] * this_stage_ot_loss)

    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio


def Blend_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    stage_lw = kwargs.get("stage_lw", [1,1,1,1])
    l1ot_lw = kwargs.get("l1ot_lw", [0,1])
    inverse = kwargs.get("inverse_depth", False)
    ot_iter = kwargs.get("ot_iter", 3)
    ot_eps = kwargs.get("ot_eps", 1)
    ot_continous = kwargs.get("ot_continous", False)
    depth_max = kwargs.get("depth_max", 100)
    depth_min = kwargs.get("depth_min", 1)
    mono = kwargs.get("mono", False)
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    stage_ot_loss = []
    stage_l1_loss = []
    range_err_ratio = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([(inputs[k], k) for k in inputs.keys() if "stage" in k]):
        depth_pred = stage_inputs['depth']
        hypo_depth = stage_inputs['hypo_depth']
        attn_weight = stage_inputs['attn_weight']
        B,H,W = depth_pred.shape
        mask = mask_ms[stage_key]
        mask = mask > 0.5
        depth_gt = depth_gt_ms[stage_key]
        depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:,None,None]  # B H W
        depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:,None,None]  # B H W

        if mono and stage_idx!=0:
            this_stage_l1_loss = F.l1_loss(stage_inputs['mono_depth'][mask], depth_gt[mask], reduction='mean')
        else:
            this_stage_l1_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

        if inverse:
            depth_itv = (1/hypo_depth[:,2,:,:]-1/hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((1/hypo_depth - 1/depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        else:
            depth_itv = (hypo_depth[:,2,:,:]-hypo_depth[:,1,:,:]).abs()  # B H W
            mask_out_of_range = ((hypo_depth - depth_gt.unsqueeze(1)).abs() <= depth_itv.unsqueeze(1)).sum(1) == 0 # B H W
        range_err_ratio.append(mask_out_of_range[mask].float().mean())

        this_stage_ot_loss = sinkhorn(depth_gt, hypo_depth, attn_weight, mask, iters=ot_iter, eps=ot_eps, continuous=ot_continous)[1]

        stage_l1_loss.append(this_stage_l1_loss)
        stage_ot_loss.append(this_stage_ot_loss)
        total_loss = total_loss + stage_lw[stage_idx] * (l1ot_lw[0] * this_stage_l1_loss + l1ot_lw[1] * this_stage_ot_loss)

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err3 = (abs_err<=3).float().mean()*100
    err1= (abs_err<=1).float().mean()*100
    return total_loss, stage_l1_loss, stage_ot_loss, range_err_ratio, epe, err3, err1