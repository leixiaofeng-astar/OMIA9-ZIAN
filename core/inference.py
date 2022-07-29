# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import cv2
import copy
import torch.nn.functional as F

from utils.transforms import transform_preds

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, 1, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    # add for potential align issue
    batch_heatmaps = np.ascontiguousarray(batch_heatmaps)

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, 1, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, 1, 1))
    idx = idx.reshape((batch_size, 1, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # get H, W of the maximum value point
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))   # pred_mask: [0.26957142]->[1, 1]
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# xiaofeng add for MV processing to get circle
tmp_img_idx = 0
def get_heatmap_center_preds(batch_heatmaps, debug=False):
    global tmp_img_idx
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, 1, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_heatmaps = np.ascontiguousarray(batch_heatmaps)
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    # heatmaps_reshaped = batch_heatmaps.reshape((batch_size, 1, -1))
    batch_heatmaps = batch_heatmaps.transpose((0, 2, 3, 1))
    preds = np.zeros((batch_size, 1, 2), dtype = int)

    for idx in range(batch_size):
        tmp_img_idx += 1
        center_x = width // 2
        center_y = height // 2
        preds[idx, 0, 0] = center_x
        preds[idx, 0, 1] = center_y

        # handle it as one image
        batch_heatmaps[batch_heatmaps < 0] = 0
        img = np.uint8( batch_heatmaps[idx, :, :, :] * 255)
        # img = cv2.medianBlur(img, 5)
        img = cv2.medianBlur(img, 7)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # param1：使用HOUGH_GRADIENT方法检测圆形时，传递给Canny边缘检测器的两个阈值的较大值。
        # param2：使用HOUGH_GRADIENT方法检测圆形时，检测圆形的累加器阈值，阈值越大检测的圆形越精确
        # https://blog.csdn.net/weixin_42904405/article/details/82814768
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
        # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
        #                            param1=50, param2=30, minRadius=10, maxRadius=30)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=30, param2=30, minRadius=18, maxRadius=50)
        max_rad = 0.0
        # import pdb
        # pdb.set_trace()
        if circles is not None:
            sub_idx = 0
            preds[idx, 0, 0] = 0
            preds[idx, 0, 1] = 0
            for i in circles[0, :]:
                sub_idx += 1
                # get the maximum outer circle
                current_rad = i[2]
                if current_rad > max_rad:
                    max_rad = current_rad
                    preds[idx, 0, 0] += i[0]
                    preds[idx, 0, 1] += i[1]

                if debug:
                    cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 255, 0), 2)
                    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

                if debug:
                    print("MV heatmap %d-%d-%d: center: (%d, %d), current_rad: %.02f" %(tmp_img_idx, idx, sub_idx, i[0], i[1], current_rad))

            preds[idx, 0, 0] /= sub_idx
            preds[idx, 0, 1] /= sub_idx
            if debug:
                roi_heatmap_file = "./debug/test_roi_heatmap_%d_%d_%d.png" %(tmp_img_idx%400, int(preds[idx, 0, 0]), int(preds[idx, 0, 1]))
                cv2.imwrite(roi_heatmap_file, cimg)

    return preds

def get_final_preds(config, batch_heatmap_ds, batch_heatmap_roi, offsets_in_roi, meta, debug=False):
    coords_ds, maxvals_ds = get_max_preds(batch_heatmap_ds)
    if config.TRAIN.MV_IDEA:
        coords_roi = get_heatmap_center_preds(batch_heatmap_roi, debug=debug)
    else:
        coords_roi, maxvals_roi = get_max_preds(batch_heatmap_roi)

    # remove offsets_in_roi
    # region_size = 2 * config.MODEL.REGION_RADIUS
    # offsets_in_roi = offsets_in_roi * region_size
    # coords: [N, 1, 2] -> [N, 2]
    coords_ds = coords_ds[:, 0, :]
    coords_roi = coords_roi[:, 0, :]
    # import pdb
    # pdb.set_trace()
    coords_lr = coords_ds * config.MODEL.DS_FACTOR  # coords in lr x scaling
    coords_hr = coords_roi + meta['roi_center'].cpu().numpy() - config.MODEL.REGION_RADIUS

    # remove offsets_in_roi
    coords_final = copy.deepcopy(coords_hr)
    coords_roi_final = copy.deepcopy(coords_roi)
    # coords_final = coords_hr + offsets_in_roi
    # coords_roi_final = coords_roi + offsets_in_roi

    return coords_lr, coords_hr, coords_final, coords_roi, coords_roi_final


