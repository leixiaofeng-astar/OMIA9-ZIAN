from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import torch.nn.functional as F
import cv2

from core.inference import get_max_preds, get_heatmap_center_preds


def save_batch_image_with_fovea(batch_image, batch_fovea, file_name, nrow=8, padding=2, ds_factor=1):
    # visualize only the first sample
    batch_image = batch_image[:1, :, :, :]
    batch_fovea = batch_fovea[:1, :]

    if ds_factor > 1:
        _, _, ih, iw = batch_image.size()
        nh = int(ih * 1.0 / ds_factor)
        nw = int(iw * 1.0 / ds_factor)
        batch_image = F.upsample(batch_image, size=(nh, nw), mode='bilinear', align_corners=True)
        batch_fovea /= ds_factor

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # xiaofeng debug
            try:
                fovea = batch_fovea[k]
                fovea_x = int(x * width + padding + fovea[0] + 0.5)
                fovea_y = int(y * height + padding + fovea[1] + 0.5)

                cv2.circle(ndarr, (int(fovea_x), int(fovea_y)), 2, [255, 0, 0], 2)
            except:
                print("Exception: running error fovea_x: %d, fovea_y: %d" % (int(fovea_x), int(fovea_y)))
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True, MV_IDEA=False):
    # visualize only the first sample
    batch_image = batch_image[:1, :, :, :]
    batch_heatmaps = batch_heatmaps[:1, :]

    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           2*heatmap_width,
                           3),
                          dtype=np.uint8)

    if MV_IDEA:
        preds = get_heatmap_center_preds(batch_heatmaps.detach().cpu().numpy())
    else:
        preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        cv2.circle(resized_image,
                   (int(preds[i][0][0]), int(preds[i][0][1])),
                   1, [255, 0, 255], 2)
        heatmap = heatmaps[0, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image = colored_heatmap*0.7 + resized_image*0.3
        cv2.circle(masked_image,
                   (int(preds[i][0][0]), int(preds[i][0][1])),
                   1, [255, 0, 255], 2)

        width_begin = heatmap_width
        width_end = heatmap_width * 2
        grid_image[height_begin:height_end, width_begin:width_end, :] = \
            masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config,
                      input, input_roi,
                      heatmap_ds, heatmap_roi, meta,
                      heatmap_ds_pred, heatmap_roi_pred,
                      fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_FOVEA_FINAL_GT:
        save_batch_image_with_fovea(
            input, meta['fovea'], '{}_fovea_gt.jpg'.format(prefix),
        )
    if config.DEBUG.SAVE_FOVEA_FINAL_PRED:
        save_batch_image_with_fovea(
            input, fovea_final_pred, '{}_fovea_final_pred.jpg'.format(prefix),
        )
    if config.DEBUG.SAVE_HEATMAP_DS_GT and heatmap_ds is not None:
        save_batch_heatmaps(
            input, heatmap_ds, '{}_hm_ds_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAP_DS_PRED:
        save_batch_heatmaps(
            input, heatmap_ds_pred, '{}_hm_pred.jpg'.format(prefix)
        )

    # HRNET has not ROI heatmap and region
    if not config.MODEL.HRNET_ONLY:
        if config.TRAIN.MV_IDEA:
            MV_Center = True
        else:
            MV_Center = False
        if config.DEBUG.SAVE_HEATMAP_ROI_GT and heatmap_roi is not None:
            save_batch_heatmaps(
                input_roi, heatmap_roi, '{}_hm_roi_gt.jpg'.format(prefix), MV_IDEA=MV_Center
            )
        if config.DEBUG.SAVE_HEATMAP_ROI_PRED:
            save_batch_heatmaps(
                input_roi, heatmap_roi_pred, '{}_hm_roi_pred.jpg'.format(prefix), MV_IDEA=MV_Center
            )
        if config.DEBUG.SAVE_FOVEA_ROI_GT and heatmap_roi is not None:
            save_batch_image_with_fovea(
                input_roi, meta['fovea_in_roi'], '{}_fovea_roi_gt.jpg'.format(prefix)
            )
        if config.DEBUG.SAVE_FOVEA_ROI_INIT_PRED:
            save_batch_image_with_fovea(
                input_roi, fovea_roi_init_pred, '{}_fovea_roi_init_pred.jpg'.format(prefix)
            )
        if config.DEBUG.SAVE_FOVEA_ROI_FINAL_PRED:
            save_batch_image_with_fovea(
                input_roi, fovea_roi_final_pred, '{}_fovea_roi_final_pred.jpg'.format(prefix)
            )

