from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class HeatmapMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(HeatmapMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True, reduce=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, 1, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, 1, -1)).split(1, 1)
        loss = 0

        heatmap_pred = heatmaps_pred[0].squeeze()
        heatmap_gt = heatmaps_gt[0].squeeze()
        if self.use_target_weight:
            loss += 0.5 * self.criterion(
                heatmap_pred.mul(target_weight),
                heatmap_gt.mul(target_weight)
            )
        else:
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss


class HybridLoss(nn.Module):
    def __init__(self, roi_weight, regress_weight, use_target_weight, hrnet_only=False, co_atten=False):
        super(HybridLoss, self).__init__()
        self.heatmap_mse = HeatmapMSELoss(use_target_weight)
        self.smooth_l1 = nn.SmoothL1Loss(size_average=True, reduce=True)
        # self.smooth_l2 = nn.HuberLoss(reduction='mean', delta=1.0)
        self.roi_weight = roi_weight
        self.regress_weight = regress_weight
        self.hrnet_only = hrnet_only
        self.co_atten = co_atten

    def get_L1_L2_distance(self, output_heatmaps, target_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, 1, height, width])
        '''
        output_heatmaps_np = output_heatmaps.cpu().detach().numpy()
        target_heatmaps_np = target_heatmaps.cpu().detach().numpy()
        assert isinstance(output_heatmaps_np, np.ndarray), \
            'output_heatmaps_np should be numpy.ndarray'
        assert output_heatmaps_np.ndim == 4, 'output_heatmaps_np should be 4-ndim'
        assert isinstance(target_heatmaps_np, np.ndarray), \
            'target_heatmaps_np should be numpy.ndarray'
        assert target_heatmaps_np.ndim == 4, 'target_heatmaps_np should be 4-ndim'

        batch_size = output_heatmaps_np.shape[0]
        num_joints = output_heatmaps_np.shape[1]
        width = output_heatmaps_np.shape[3]
        heatmaps_reshaped = output_heatmaps_np.reshape((batch_size, 1, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, 1, 1))
        idx = idx.reshape((batch_size, 1, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        # get H, W of the maximum value point
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))  # pred_mask: [0.26957142]->[1, 1]
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        # return preds, maxvals


        # calculae for target heatmaps
        batch_size = target_heatmaps_np.shape[0]
        width = target_heatmaps_np.shape[3]
        preds_target_heatmaps_reshaped = target_heatmaps_np.reshape((batch_size, 1, -1))
        idx = np.argmax(preds_target_heatmaps_reshaped, 2)
        preds_target_maxvals = np.amax(preds_target_heatmaps_reshaped, 2)

        preds_target_maxvals = preds_target_maxvals.reshape((batch_size, 1, 1))
        idx = idx.reshape((batch_size, 1, 1))

        preds_target = np.tile(idx, (1, 1, 2)).astype(np.float32)

        # get H, W of the maximum value point
        preds_target[:, :, 0] = (preds_target[:, :, 0]) % width
        preds_target[:, :, 1] = np.floor((preds_target[:, :, 1]) / width)

        preds_target_pred_mask = np.tile(np.greater(preds_target_maxvals, 0.0), (1, 1, 2))  # pred_mask: [0.26957142]->[1, 1]
        preds_target_pred_mask = preds_target_pred_mask.astype(np.float32)

        preds_target *= preds_target_pred_mask

        # calculae for L1 and L2 distance
        # finding sum of squares
        l2_dist_sum = 0.0
        l1_dist_sum = 0.0
        # import pdb; pdb.set_trace()
        for _ in range(batch_size):
            item_dist = np.sqrt(np.sum((preds[_, :] - preds_target[_, :]) ** 2))
            l2_dist_sum += item_dist
            l1_dist_sum = np.sum(np.abs(preds[_, :] - preds_target[_, :]))
        l2_dist_avg = l2_dist_sum / batch_size
        l1_dist_avg = l1_dist_sum / batch_size

        # print('L1 Distance: {} -- L2 Distance: {}'.format(l1_dist_avg, l2_dist_avg))
        return l1_dist_avg, l2_dist_avg

    # TODO: add ROI loss here -- re-use pred_offset, target_offset !!!
    def forward(self, pred_ds, target_ds, pred_roi, target_roi, pred_offset, target_offset, target_weight):
        heatmap_ds_loss = self.heatmap_mse(pred_ds, target_ds, target_weight)
        Loss_with_L1_L2 = False
        if self.hrnet_only:
            heatmap_roi_loss = torch.zeros(1, dtype=torch.float)
            # B = pred_roi.shape[0]
            # np.zeros((B, 1), dtype=np.float32)
            regress_loss = torch.zeros(1, dtype=torch.float)
            hybrid_loss = heatmap_ds_loss
        else:
            heatmap_roi_loss = self.heatmap_mse(pred_roi, target_roi, target_weight)
            heatmap_roi_loss = heatmap_roi_loss * self.roi_weight

            if self.co_atten:

                if Loss_with_L1_L2:
                    L1_loss, L2_loss = self.get_L1_L2_distance(pred_ds, target_ds)
                    heatmap_ds_loss += L1_loss + L2_loss

                if Loss_with_L1_L2:
                    L1_loss, L2_loss = self.get_L1_L2_distance(pred_roi, target_roi)
                    heatmap_roi_loss += (L1_loss + L2_loss) * self.roi_weight

                heatmap_subroi1_loss = self.heatmap_mse(pred_offset[0], target_roi, target_weight)
                heatmap_subroi2_loss = self.heatmap_mse(pred_offset[1], target_roi, target_weight)
                regress_loss = (self.roi_weight / 4) * (heatmap_subroi1_loss + heatmap_subroi2_loss)

                if Loss_with_L1_L2:
                    L1_loss, L2_loss = self.get_L1_L2_distance(pred_offset[0], target_roi)
                    regress_loss += (self.roi_weight/10) * (L1_loss + L2_loss)
                    L1_loss, L2_loss = self.get_L1_L2_distance(pred_offset[1], target_roi)
                    regress_loss += (self.roi_weight/10) * (L1_loss + L2_loss)
            else:
                pred_offset = pred_offset.mul(target_weight)
                target_offset = target_offset.mul(target_weight)
                regress_loss = self.smooth_l1(pred_offset, target_offset)
                regress_loss = regress_loss * self.regress_weight   # self.regress_weight fix to 0 here

            hybrid_loss = heatmap_ds_loss + heatmap_roi_loss + regress_loss

        return hybrid_loss, {'heatmap_ds': heatmap_ds_loss,
                             'heatmap_roi': heatmap_roi_loss,
                             'offset': regress_loss}



