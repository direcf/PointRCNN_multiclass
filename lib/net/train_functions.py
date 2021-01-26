import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from collections import namedtuple


def model_joint_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn(model, data):
        if cfg.RPN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']

            if not cfg.RPN.FIXED:
                rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
                rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
                rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking=True).float()

            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True).float()
            input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}
        else:
            input_data = {}
            for key, val in data.items():
                if key != 'sample_id':
                    input_data[key] = torch.from_numpy(val).contiguous().cuda(non_blocking=True).float()
            if not cfg.RCNN.ROI_SAMPLE_JIT:
                pts_input = torch.cat((input_data['pts_input'], input_data['pts_features']), dim=-1)
                input_data['pts_input'] = pts_input
        # print("input data dict key", input_data.keys()) # dict_keys(['pts_input', 'gt_boxes3d'])
        ret_dict = model(input_data)
        # print("ret_dict key", ret_dict)
        # print("rpn cls", ret_dict['rpn_cls'].size()) # size([4, 16384, 1])
        # print("rpn reg", ret_dict['rpn_reg'].size()) # size([4, 16384, 76])
        # print("backbone xyz", ret_dict['backbone_xyz'].size()) # size([4, 16384, 3])
        # print("backbone features", ret_dict['backbone_features'].size()) # size([4, 128, 16384])
        # print("rois", ret_dict['rois'].size()) # size([4, 512, 7])
        # print("roi scores raw", ret_dict['roi_scores_raw'].size()) # size([4, 512])
        # print("seg_result", ret_dict['seg_result'].size()) # size([4, 16384])
        # print("rcnn cls", ret_dict['rcnn_cls'].size()) # size([256, 4])
        # print("rcnn reg", ret_dict['rcnn_reg'].size()) # size([256, 46])
        # print("sampled pts", ret_dict['sampled_pts'].size()) # size([256, 512, 3])
        # print("pts feature", ret_dict['pts_feature'].size()) # size([256, 512, 130])
        # print("cls label", ret_dict['cls_label'].size()) # size([256])
        # print("reg valid mask", ret_dict['reg_valid_mask'].size()) # size([256])
        # print("gt of rois", ret_dict['gt_of_rois'].size()) # size([256, 7])
        # print("gt iou", ret_dict['gt_iou'].size()) # size([256])
        # print("roi boxes3d", ret_dict['roi_boxes3d'].size()) # size([256, 7])
        # print("pts input", ret_dict['pts_input'].size()) # size([256, 512, 133])
        tb_dict = {}
        disp_dict = {}
        loss = 0
        if cfg.RPN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            rpn_loss = get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()

        if cfg.RCNN.ENABLED:
            rcnn_loss = get_rcnn_loss(model, ret_dict, tb_dict)
            disp_dict['reg_fg_sum'] = tb_dict['rcnn_reg_fg']
            loss += rcnn_loss

        disp_dict['loss'] = loss.item()

        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=MEAN_SIZE,
                                        get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                        'rpn_loss': rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                        'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item()})

        return rpn_loss

    def get_rcnn_loss(model, ret_dict, tb_dict):
        rcnn_cls, rcnn_reg = ret_dict['rcnn_cls'], ret_dict['rcnn_reg']

        cls_label = ret_dict['cls_label'].float() #### cls_label process
        reg_valid_mask = ret_dict['reg_valid_mask']
        roi_boxes3d = ret_dict['roi_boxes3d']
        roi_size = roi_boxes3d[:, 3:6]
        gt_boxes3d_ct = ret_dict['gt_of_rois']
        pts_input = ret_dict['pts_input']

        # rcnn classification loss
        if isinstance(model, nn.DataParallel):
            cls_loss_func = model.module.rcnn_net.cls_loss_func
        else:
            cls_loss_func = model.rcnn_net.cls_loss_func
        # print("cls_label",cls_label) #### -1, 0, 1로 이루어진 tensor 256개
        # print("cls_label_size",cls_label.size()) #### torch.size([256])
        cls_label_flat = cls_label.view(-1)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            rcnn_cls_flat = rcnn_cls.view(-1)

            cls_target = (cls_label_flat > 0).float()
            pos = (cls_label_flat > 0).float()
            neg = (cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

            rcnn_loss_cls = cls_loss_func(rcnn_cls_flat, cls_target, cls_weights)
            rcnn_loss_cls_pos = (rcnn_loss_cls * pos).sum()
            rcnn_loss_cls_neg = (rcnn_loss_cls * neg).sum()
            rcnn_loss_cls = rcnn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rcnn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rcnn_loss_cls_neg.item()

        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), cls_label, reduction='none')
            cls_valid_mask = (cls_label_flat >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy': #### TRAIN -> RCNN
        # elif cfg.TRAIN.LOSS_CLS == 'CrossEntropy':
            # print(rcnn_cls.size()) #### torch.size([256,4])
            # tensor([[ 0.0186, -0.0566, -0.0374, -0.0273],
            #         [-0.0119, -0.0458, -0.0206, -0.0464],
            #         [-0.0035, -0.0503, -0.0334, -0.0135],
            #         ...,
            #         [ 0.0098, -0.0219, -0.0139, -0.0330],
            #         [ 0.0182, -0.0153,  0.0086, -0.0376],
            #         [ 0.0071, -0.0278,  0.0146, -0.0302]], device='cuda:0',
            # print(rcnn_cls_reshape.size()) #### torch.size([256,4])
            # rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1).sum(dim=1) #### choose sum / mean
            # rcnn_cls_reshape = rcnn_cls.view(-1)

            rcnn_cls_reshape = rcnn_cls.view(rcnn_cls.shape[0], -1)
            cls_target = cls_label_flat.long()

            # print("cls_target", cls_target) # print(cls_target.size()) #### torch.size([256])
            # tensor([ 1, -1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,
            #         1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  0,  0,  0,  0,
            #         0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0,  0, -1, -1, -1,  0, -1, -1, -1, -1, -1,  1,  1, -1,  1,  1, -1,
            #         1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1,
            #         1, -1,  1, -1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1, -1,
            #         1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  0,  0,
            #         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
            #         -1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,
            #         -1,  1, -1,  1, -1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            #         0,  0,  0,  0], device='cuda:0')       
            cls_valid_mask = (cls_target >= 0).float() #### -1 : invalid ???????

            cls_target_final = torch.zeros(rcnn_cls_reshape.shape[0],rcnn_cls_reshape.shape[1])
            # print(cls_target_final.size()) # size([256,4])
            for i in range(cls_target_final.shape[0]):
                if cls_target[i] == -1:
                    cls_target[i] = 0
            #    cls_target_final[i, cls_target[i].cpu().numpy()] = 1 #### class가 0,1,2,3,.. 한 줄 이어야 한다...
            # print("cls_target_final", cls_target) # size([256,4]) #### too many value 1...
            cls_target_final = cls_target_final.cuda()
            cls_target_final = cls_target_final.long()
            # print("cls_target_final", cls_target_final) # size([256,4])

            #### cls_target = cls_target.unsqueeze(1) #### size([256,1])
            #### cls_target = torch.cat((cls_target, cls_target, cls_target, cls_target), 1) #### size([256,4])
            #### cls_target = cls_target.view(-1)
            
            # print("rcnn_cls_reshape", rcnn_cls_reshape) #### size([256,4])
            # print(cls_target.size()) #### size([256])
            batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target) #### loss calculation
            ## batch_loss_cls = cls_loss_func(rcnn_cls_reshape, cls_target) #### loss calculation
            # print(batch_loss_cls.size()) #### size([256])
            normalizer = torch.clamp(cls_valid_mask.sum(), min=1.0)
            rcnn_loss_cls = (batch_loss_cls.mean(dim=0) * cls_valid_mask).sum() / normalizer
            # rcnn_loss_cls = (batch_loss_cls.mean(dim=1) * cls_valid_mask).sum() / normalizer
            #### why the writer misunderstand dimension 0 and 1?

        else:
            raise NotImplementedError

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            all_anchor_size = roi_size
            anchor_size = all_anchor_size[fg_mask] if cfg.RCNN.SIZE_RES_ON_ROI else MEAN_SIZE

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rcnn_reg.view(batch_size, -1)[fg_mask],
                                        gt_boxes3d_ct.view(batch_size, 8)[fg_mask],
                                        loc_scope=cfg.RCNN.LOC_SCOPE,
                                        loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                        loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                        get_ry_fine=True)

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
            tb_dict.update(reg_loss_dict)
        else:
            loss_loc = loss_angle = loss_size = rcnn_loss_reg = rcnn_loss_cls * 0

        rcnn_loss = rcnn_loss_cls + rcnn_loss_reg
        tb_dict['rcnn_loss_cls'] = rcnn_loss_cls.item()
        tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
        tb_dict['rcnn_loss'] = rcnn_loss.item()

        tb_dict['rcnn_loss_loc'] = loss_loc.item()
        tb_dict['rcnn_loss_angle'] = loss_angle.item()
        tb_dict['rcnn_loss_size'] = loss_size.item()
        # fg : foreground, bg : background
        tb_dict['rcnn_cls_fg'] = (cls_label > 0).sum().item()
        tb_dict['rcnn_cls_bg'] = (cls_label == 0).sum().item()
        tb_dict['rcnn_reg_fg'] = reg_valid_mask.sum().item()

        return rcnn_loss

    return model_fn
