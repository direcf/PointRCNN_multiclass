import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            print("6 : point_rcnn.py & rpn.py")
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.RPN.ENABLED: # RPN.ENABLED=TRUE, RPN.FIXED=FALSE
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                # print("10 : rpn input keys when RPN.FIXED ", input_data.keys()) #### ['pts_input', 'gt_boxes3d']
                # print(" input_data : pts_input", input_data['pts_input'].size()) # size([16,16384,3])
                # print("input_data : gt_boxes3d", input_data['gt_boxes3d'].size()) # size([16,23(22,19),7]) 
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
                # print("11 : rpn output keys when RPN.FIXED ", rpn_output.keys()) #### ['rpn_cls', 'rpn_reg', 'backbone_xyz', 'backbone_features']
            
            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0] # rpn scoring when class = 1
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)
                    # print("depth", pts_depth.size()) #### size([4,16384])

                    # proposal layer
                    # print("12 : rois, roi_scores_raw, seg_result by using proposal_layer")
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                    #### rois size([4,512,7]), roi_scores_raw size([4,512])
                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training: # Use it
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info) #### core point!
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED: # Don't use this
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError
        # print("rcnn_input_info", rcnn_input_info.keys())
        # dict_keys(['rpn_xyz', 'rpn_features', 'seg_mask', 'roi_boxes3d', 'pts_depth', 'gt_boxes3d'])
        # print("1", rcnn_input_info['rpn_xyz'].size())
        # print("2", rcnn_input_info['rpn_features'].size())
        # print("3", rcnn_input_info['seg_mask'].size())
        # print("4", rcnn_input_info['roi_boxes3d'].size())
        # print("5", rcnn_input_info['gt_boxes3d'].size())



        # print(output.keys()) #### ['rpn_cls', 'rpn_reg', 'backbone_xyz', 'backbone_features', 'rois', 
        #'roi_scores_raw', 'seg_result', 'rcnn_cls', 'rcnn_reg', 'sampled_pts', 'pts_feature', 'cls_label', 
        # 'reg_valid_mask', 'gt_of_rois', 'gt_iou', 'roi_boxes3d', 'pts_input']

        # print("1", output['rcnn_cls'].size())
        # print("2", output['rcnn_reg'].size())
        # print("3", output['sampled_pts'].size())
        # print("4", output['cls_label'].size())
        # print("5", output['pts_feature'].size())
        # print("6", output['reg_valid_mask'].size())
        # print("7", output['gt_of_rois'].size())
        # print("8", output['gt_iou'].size())
        # print("9", output['roi_boxes3d'].size())
        # print("10", output['pts_input'].size())

        # print("rpn_cls", output['rpn_cls'].size()) # size([16,16384,1])
        # print("rpn_reg", output['rpn_reg'].size()) # size([16,16384,76])
        # print("backbone_xyz", output['backbone_xyz'].size()) # size([16,16384,3])
        # print("backbone_features", output['backbone_features'].size()) # size([16,128,16384])
        # print("rois", output['rois'].size()) # size([4,512,7])
        # print("roi_scores_raw", output['roi_scores_raw'].size()) # size([4,512])
        # print("seg_result", output['seg_result'].size()) # size([4,16384])
    #     output check {'rpn_cls': tensor([[[-5.1735],
    #      [-3.7231],
    #      [-2.9883],
    #      ...,
    #      [-3.7360],
    #      [-3.6476],
    #      [-6.0314]],

    #     [[-2.3807],
    #      [-3.1805],
    #      [-2.7425],
    #      ...,
    #      [-3.3026],
    #      [-0.8901],
    #      [-2.9785]],

    #     [[-3.9357],
    #      [-4.6314],
    #      [-2.8741],
    #      ...,
    #      [-4.5779],
    #      [-5.5495],
    #      [-3.5850]],

    #     ...,

    #     [[-5.1414],
    #      [-3.7547],
    #      [-2.9199],
    #      ...,
    #      [-2.7652],
    #      [-4.4252],
    #      [-3.0268]],

    #     [[-4.8873],
    #      [-4.7900],
    #      [-2.2165],
    #      ...,
    #      [-5.0135],
    #      [-3.7314],
    #      [-2.1040]],

    #     [[-4.6535],
    #      [-4.5354],
    #      [-2.1991],
    #      ...,
    #      [-3.9017],
    #      [-5.5090],
    #      [-3.3456]]], device='cuda:0', grad_fn=<TransposeBackward0>), 
    
    #      'rpn_reg': tensor([[[-0.0030, -0.0025, -0.0270,  ...,  0.0106, -0.0045,  0.0100],
    #      [-0.0069, -0.0056,  0.0013,  ..., -0.0135, -0.0121, -0.0205],
    #      [-0.0073, -0.0028, -0.0037,  ..., -0.0119,  0.0050, -0.0087],
    #      ...,
    #      [-0.0160, -0.0036,  0.0042,  ...,  0.0078, -0.0131, -0.0103],
    #      [-0.0097, -0.0009,  0.0159,  ...,  0.0069,  0.0044, -0.0213],
    #      [-0.0197, -0.0046,  0.0093,  ..., -0.0091, -0.0036, -0.0064]],

    #     [[-0.0044, -0.0001,  0.0101,  ..., -0.0037, -0.0018, -0.0181],
    #      [-0.0141, -0.0067,  0.0206,  ...,  0.0149, -0.0037, -0.0043],
    #      [ 0.0045, -0.0007,  0.0050,  ...,  0.0048,  0.0027, -0.0060],
    #      ...,
    #      [-0.0080, -0.0132,  0.0084,  ..., -0.0030, -0.0059, -0.0044],
    #      [-0.0191, -0.0097,  0.0027,  ..., -0.0109, -0.0052,  0.0017],
    #      [-0.0431, -0.0046,  0.0214,  ..., -0.0081,  0.0418, -0.0160]],

    #     [[-0.0224,  0.0354,  0.0456,  ...,  0.0470,  0.0364,  0.0129],
    #      [-0.0003, -0.0068,  0.0088,  ..., -0.0124,  0.0023, -0.0108],
    #      [-0.0140,  0.0025, -0.0173,  ..., -0.0129, -0.0042, -0.0022],
    #      ...,
    #      [-0.0142,  0.0074, -0.0136,  ...,  0.0014, -0.0230, -0.0038],
    #      [ 0.0053, -0.0010,  0.0010,  ...,  0.0003, -0.0128, -0.0053],
    #      [-0.0144,  0.0066, -0.0089,  ..., -0.0130, -0.0001,  0.0043]],

    #     ...,

    #     [[-0.0073,  0.0144,  0.0047,  ..., -0.0046, -0.0127, -0.0053],
    #      [-0.0073,  0.0143,  0.0134,  ..., -0.0121, -0.0172, -0.0087],
    #      [-0.0012,  0.0008,  0.0075,  ..., -0.0115,  0.0037,  0.0113],
    #      ...,
    #      [-0.0160, -0.0008,  0.0014,  ..., -0.0096,  0.0009,  0.0168],
    #      [-0.0077, -0.0204,  0.0016,  ...,  0.0069, -0.0034, -0.0109],
    #      [-0.0118, -0.0002, -0.0119,  ...,  0.0076, -0.0003, -0.0028]],

    #     [[-0.0058, -0.0029, -0.0242,  ...,  0.0193, -0.0070, -0.0189],
    #      [-0.0080,  0.0025,  0.0093,  ...,  0.0026, -0.0154,  0.0006],
    #      [-0.0163, -0.0162,  0.0073,  ...,  0.0118, -0.0049,  0.0109],
    #      ...,
    #      [-0.0152, -0.0099,  0.0081,  ...,  0.0083, -0.0066, -0.0163],
    #      [-0.0061, -0.0071,  0.0113,  ...,  0.0025,  0.0034,  0.0019],
    #      [-0.0297, -0.0089,  0.0032,  ...,  0.0131, -0.0165, -0.0016]],

    #     [[-0.0328, -0.0086,  0.0042,  ...,  0.0095, -0.0195, -0.0202],
    #      [-0.0044, -0.0077, -0.0021,  ..., -0.0024,  0.0027, -0.0147],
    #      [-0.0057, -0.0080, -0.0051,  ..., -0.0027, -0.0138, -0.0278],
    #      ...,
    #      [ 0.0021, -0.0028,  0.0149,  ...,  0.0022, -0.0052, -0.0128],
    #      [-0.0211, -0.0081,  0.0074,  ..., -0.0160, -0.0039, -0.0115],
    #      [-0.0288, -0.0105,  0.0017,  ..., -0.0049, -0.0030,  0.0006]]],
    #    device='cuda:0', grad_fn=<CloneBackward>), 
    
    #      'backbone_xyz': tensor([[[-3.0450e+00,  1.7477e+00,  7.3536e+00],
    #      [-2.8207e+00,  1.8023e+00,  1.5552e+01],
    #      [-1.4563e+00,  1.7466e+00,  9.0663e+00],
    #      ...,
    #      [-3.2104e+00,  7.5752e-01,  1.7776e+01],
    #      [-4.7241e+00,  1.6880e+00,  5.2166e+01],
    #      [-3.2135e+00,  1.7506e+00,  9.0108e+00]],

    #     [[ 1.2817e+00,  1.5900e+00,  7.0689e+00],
    #      [ 4.3956e+00,  1.5504e+00,  7.4190e+00],
    #      [-2.1143e+00,  1.0660e+00,  2.4041e+01],
    #      ...,
    #      [-1.4653e+00,  4.8930e-01,  2.4777e+01],
    #      [ 4.5721e+00,  6.7786e-01,  3.2771e+01],
    #      [-2.9633e+00,  7.6913e-01,  4.0196e+00]],

    #     [[ 1.8854e+00,  5.2625e-01,  9.7074e+00],
    #      [ 2.4766e+00,  1.6312e+00,  7.0634e+00],
    #      [-3.9135e+00,  9.0114e-01,  6.5021e+00],
    #      ...,
    #      [ 2.6864e+00,  1.5195e+00,  7.9244e+00],
    #      [-2.2287e+01,  5.8842e-01,  3.9904e+01],
    #      [ 1.3065e-02,  1.7124e+00,  8.1955e+00]],

    #     ...,

    #     [[ 7.3371e+00,  1.5727e+00,  8.6214e+00],
    #      [ 2.9540e+00,  1.7296e+00,  8.6262e+00],
    #      [ 1.0131e+00,  1.7175e+00,  6.2225e+00],
    #      ...,
    #      [ 6.7662e+00,  1.7713e+00,  1.2571e+01],
    #      [-1.4129e+01,  1.0701e+00,  3.4923e+01],
    #      [-4.6669e+00,  1.5269e+00,  1.7717e+01]],

    #     [[ 5.8294e+00,  1.4071e+00,  1.2038e+01],
    #      [ 5.7820e+00,  1.4821e+00,  7.5465e+00],
    #      [ 7.6756e+00,  1.5017e-01,  1.3766e+01],
    #      ...,
    #      [-3.9948e+00,  1.0493e+00,  8.4988e+00],
    #      [ 7.1555e+00, -2.5194e-01,  1.3790e+01],
    #      [ 3.2816e+00,  4.8616e-01,  1.2042e+01]],

    #     [[ 7.5882e+00,  4.0229e-01,  1.5010e+01],
    #      [ 1.1825e+01, -1.9168e-01,  3.3753e+01],
    #      [-2.4167e+00,  1.0656e+00,  1.1514e+01],
    #      ...,
    #      [ 7.3970e+00,  2.1952e+00,  2.0686e+01],
    #      [-1.3037e-01,  1.7683e+00,  8.4944e+00],
    #      [ 1.6258e-01,  9.2454e-01,  1.1520e+01]]], device='cuda:0'), 
    
    #      'backbone_features': tensor([[[1.0240, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #      [0.8422, 0.0000, 0.2286,  ..., 0.0000, 0.0000, 0.0000],
    #      [0.6216, 0.3254, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #      ...,
    #      [0.0000, 0.7835, 0.3120,  ..., 0.0247, 1.8844, 0.0000],
    #      [1.7096, 0.7817, 0.7809,  ..., 0.2252, 0.0000, 0.4022],
    #      [0.0000, 0.6971, 0.0171,  ..., 0.7339, 1.2453, 0.4075]],

    #     [[0.0000, 0.0485, 0.0000,  ..., 0.0000, 0.0000, 1.0796],
    #      [0.3091, 0.0000, 0.0242,  ..., 0.0000, 0.0000, 2.4629],
    #      [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2058],
    #      ...,
    #      [0.7341, 0.1059, 0.7387,  ..., 2.2113, 0.4392, 0.0000],
    #      [0.0000, 0.7646, 0.5568,  ..., 1.0912, 1.4438, 0.0000],
    #      [0.3786, 0.0000, 0.9027,  ..., 0.2513, 0.0612, 0.0000]],

    #     [[0.0000, 0.8502, 1.0199,  ..., 1.9348, 0.2216, 0.0000],
    #      [2.1404, 0.4681, 0.8735,  ..., 1.2392, 0.0000, 0.0000],
    #      [0.0000, 0.0992, 0.6485,  ..., 0.6603, 0.0000, 0.0000],
    #      ...,
    #      [0.0000, 0.0000, 0.0000,  ..., 0.6013, 0.0000, 0.0000],
    #      [0.0000, 0.3969, 0.0000,  ..., 0.7706, 1.1524, 1.2900],
    #      [0.0000, 0.4175, 0.0000,  ..., 0.2451, 0.0000, 0.8827]],

    #     ...,

    #     [[0.0000, 0.0000, 0.4496,  ..., 0.0000, 0.0000, 0.0000],
    #      [0.0000, 1.2469, 2.0409,  ..., 0.8542, 0.0000, 0.0000],
    #      [0.0000, 0.0000, 0.3108,  ..., 0.0000, 0.0000, 0.0000],
    #      ...,
    #      [2.1831, 0.3466, 1.2687,  ..., 0.2440, 0.9746, 0.8589],
    #      [0.0000, 0.0000, 0.0000,  ..., 0.5533, 0.0664, 0.0000],
    #      [0.7207, 0.0000, 0.5719,  ..., 0.0000, 0.3842, 0.7616]],

    #     [[0.0000, 1.5484, 0.0000,  ..., 0.0000, 0.0000, 0.3902],
    #      [0.0000, 0.0000, 0.0000,  ..., 1.0491, 0.0000, 1.9582],
    #      [0.3808, 0.4163, 0.0116,  ..., 0.8172, 1.2071, 0.4874],
    #      ...,
    #      [0.1024, 0.2911, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #      [0.0000, 1.0773, 0.3174,  ..., 0.9181, 0.6023, 0.0000],
    #      [0.0000, 0.0000, 0.6266,  ..., 0.0000, 0.7492, 0.0000]],

    #     [[0.0000, 0.2948, 0.0000,  ..., 0.0000, 0.0000, 1.1005],
    #      [0.0000, 0.0000, 2.1901,  ..., 0.0000, 0.7972, 0.5424],
    #      [0.7746, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #      ...,
    #      [0.0000, 0.1079, 0.0000,  ..., 1.5799, 0.7394, 0.0000],
    #      [0.0000, 0.7342, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
    #      [0.0000, 0.0000, 0.0000,  ..., 1.2627, 1.7921, 0.7948]]],
    #    device='cuda:0', grad_fn=<SqueezeBackward1>)}
        return output
