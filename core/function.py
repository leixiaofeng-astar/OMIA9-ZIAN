
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
from tqdm import tqdm

import numpy as np
import torch

from core.evaluate import calc_batch_l2_dist
from core.inference import get_final_preds
from utils.transforms import flip_torch
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    heatmap_ds_losses = AverageMeter()
    heatmap_roi_losses = AverageMeter()
    offset_losses = AverageMeter()

    # switch to train mode
    model.train()
    cuda = torch.device('cuda')

    end = time.time()
    # enumerate(tqdm(train_loader)) if necessary
    for i, (input, input_roi, heatmap_ds, heatmap_roi, offset_in_roi, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        model = model.to(device=cuda)
        input = input.to(device=cuda)

        # compute output
        #TODO -- offset_in_roi_pred change to heatmap
        heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta = \
            model(input, meta, input_roi=input_roi)

        heatmap_ds = heatmap_ds.cuda(non_blocking=True)
        heatmap_roi = heatmap_roi.cuda(non_blocking=True)
        offset_in_roi = offset_in_roi.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        #TODO -- loss is different now
        loss, loss_dict = criterion(heatmap_ds_pred, heatmap_ds,
                                    heatmap_roi_pred, heatmap_roi,
                                    offset_in_roi_pred, offset_in_roi,
                                    target_weight)
        heatmap_ds_loss = loss_dict['heatmap_ds']
        heatmap_roi_loss = loss_dict['heatmap_roi']
        offset_loss = loss_dict['offset']

        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        heatmap_ds_losses.update(heatmap_ds_loss.item(), input.size(0))
        heatmap_roi_losses.update(heatmap_roi_loss.item(), input.size(0))
        offset_losses.update(offset_loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # Epoch: [29][20/80]      Time 1.109s (1.171s)    Speed 7.2 samples/s
            # Data 0.000s (0.066s)    Loss 0.0003522 (0.0003718)
            # -- Losses 0.0000344 (0.0000428), 0.0003178 (0.0003291), 0.0000000 (0.000000)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.7f} ({loss.avg:.7f}) -- ' \
                  'Losses {heatmap_ds_loss.val:.7f} ({heatmap_ds_loss.avg:.7f}), ' \
                  '{heatmap_roi_loss.val:.7f} ({heatmap_roi_loss.avg:.7f}), ' \
                  '{offset_loss.val:.7f} ({offset_loss.avg:.6f})' \
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val, data_time=data_time,
                      loss=losses, heatmap_ds_loss=heatmap_ds_losses,
                      heatmap_roi_loss=heatmap_roi_losses, offset_loss=offset_losses)
            logger.info(msg)

            # comment it for speeding
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('heatmap_ds_loss', heatmap_ds_losses.val, global_steps)
            writer.add_scalar('heatmap_roi_loss', heatmap_roi_losses.val, global_steps)
            writer.add_scalar('offset_loss', offset_losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            _, _, fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred = \
                get_final_preds(config,
                                heatmap_ds_pred.detach().cpu().numpy(),
                                heatmap_roi_pred.detach().cpu().numpy(),
                                None,
                                # offset_in_roi_pred.detach().cpu().numpy(),
                                meta)
            # change input_roi to input_roi[0] for 3 ROI
            if config.MODEL.TRIP_ROI is True:
                save_debug_images(config,
                                  input, input_roi[0],
                                  heatmap_ds, heatmap_roi, meta,
                                  heatmap_ds_pred, heatmap_roi_pred,
                                  fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred,
                                  prefix)
            else:
                save_debug_images(config,
                                  input, input_roi,
                                  heatmap_ds, heatmap_roi, meta,
                                  heatmap_ds_pred, heatmap_roi_pred,
                                  fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred,
                                  prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, db_vals=[], debug_all=False):
    batch_time = AverageMeter()
    lr_init_dists = AverageMeter()
    hr_init_dists = AverageMeter()
    final_dists = AverageMeter()

    # switch to evaluate mode
    # import pdb
    # pdb.set_trace()
    # model = model.cpu()
    model.eval()
    cuda = torch.device('cuda')

    num_samples = len(val_dataset)
    all_fovea_final_preds = np.zeros((num_samples, 2), dtype=np.float32)
    all_fovea_lr_init_preds = np.zeros((num_samples, 2), dtype=np.float32)
    all_fovea_hr_init_preds = np.zeros((num_samples, 2), dtype=np.float32)
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        # xiaofeng comment -- (input, meta) --> 'image': image_file / 'fovea': np.array([fx, fy])
        for i, (input, meta) in enumerate(val_loader):
            # compute output
            model = model.to(device=cuda)
            input = input.to(device=cuda)
            heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta = \
                model(input, meta, input_roi=None)

            # xiaofeng add for one sample
            # array([[992., 812.]], dtype=float32),
            # array([[992., 811.]], dtype=float32),
            # array([[991.55975, 810.6652 ]], dtype=float32),
            # array([[128., 127.]], dtype=float32),
            # array([[127.55977, 126.66521]], dtype=float32))
            batch = input.size()[0]
            fovea_lr_init_pred, fovea_hr_init_pred, fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred = \
                get_final_preds(config,
                                heatmap_ds_pred.cpu().numpy(),
                                heatmap_roi_pred.cpu().numpy(),
                                None, meta)
                                # offset_in_roi_pred.cpu().numpy(), meta)

            if config.TEST.FLIP_TEST:
                # import pdb; pdb.set_trace()
                input_flipped = flip_torch(input, dim=3)
                meta_flip = copy.deepcopy(meta)
                heatmap_ds_pred_flip, heatmap_roi_pred_flip, offset_in_roi_pred_flip, meta_flip = model(input_flipped, meta_flip, input_roi=None)

                # coords_lr, coords_hr, coords_final, coords_roi, coords_roi_final
                fovea_lr_init_pred_flip, fovea_hr_init_pred_flip, fovea_final_pred_flip, fovea_roi_init_pred_flip, fovea_roi_final_pred_flip = \
                    get_final_preds(config,
                                    heatmap_ds_pred_flip.cpu().numpy(),
                                    heatmap_roi_pred_flip.cpu().numpy(),
                                    None, meta_flip, debug=debug_all)
                                    # offset_in_roi_pred_flip.cpu().numpy(), meta_flip, debug=debug_all)

                width = input.size()[3]
                fovea_lr_init_pred_flip[:, 0] = width - fovea_lr_init_pred_flip[:, 0]
                fovea_hr_init_pred_flip[:, 0] = width - fovea_hr_init_pred_flip[:, 0]
                fovea_final_pred_flip[:, 0] = width - fovea_final_pred_flip[:, 0]

                if debug_all:
                    gt_location = meta['fovea'].cpu().numpy()
                    hr_loc_orig = fovea_hr_init_pred
                    hr_loc_flip = fovea_hr_init_pred_flip
                    logger.info('Image ID %d: GT:(%.2f, %.2f), HR:(%.2f, %.2f), HR_Flip:(%.2f, %.2f)' \
                                %(i*batch, gt_location(0, 0), gt_location(0, 1), hr_loc_orig(0, 0), \
                                  hr_loc_orig(0, 1), hr_loc_flip(0, 0), hr_loc_flip(0, 1)))

                fovea_lr_init_pred = (fovea_lr_init_pred + fovea_lr_init_pred_flip) * 0.5
                fovea_hr_init_pred = (fovea_hr_init_pred + fovea_hr_init_pred_flip) * 0.5
                fovea_final_pred = (fovea_final_pred + fovea_final_pred_flip) * 0.5

            num_images = input.size(0)
            fovea_target = meta['fovea'].cpu().numpy()
            lr_init_dist = calc_batch_l2_dist(fovea_lr_init_pred, fovea_target, reduce=True)
            hr_init_dist = calc_batch_l2_dist(fovea_hr_init_pred, fovea_target, reduce=True)
            final_dist = calc_batch_l2_dist(fovea_final_pred, fovea_target, reduce=True)

            lr_init_dists.update(lr_init_dist, num_images)
            hr_init_dists.update(hr_init_dist, num_images)
            final_dists.update(final_dist, num_images)

            all_fovea_final_preds[idx:idx+num_images, :] = fovea_final_pred
            all_fovea_lr_init_preds[idx:idx+num_images, :] = fovea_lr_init_pred
            all_fovea_hr_init_preds[idx:idx+num_images, :] = fovea_hr_init_pred
            idx += num_images

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            image_path.extend(meta['image'])
            # Test: [0/100]  Time 0.251 (0.275) LRInit 23.7397 (18.6406) HRInit 22.5420 (18.3990) Final 22.5497 (18.3919)
            if (i % config.PRINT_FREQ == 0) or (debug_all is True):
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'LRInit {lr_init_dist.val:.4f} ({lr_init_dist.avg:.4f})\t' \
                      'HRInit {hr_init_dist.val:.4f} ({hr_init_dist.avg:.4f})\t' \
                      'Final {final_dist.val:.4f} ({final_dist.avg:.4f})\t' \
                      .format(
                          i+1, len(val_loader), batch_time=batch_time,
                          lr_init_dist=lr_init_dists,
                          hr_init_dist=hr_init_dists,
                          final_dist=final_dists)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)

                save_debug_images(config,
                                  input, meta['input_roi'],
                                  None, None, meta,
                                  heatmap_ds_pred, heatmap_roi_pred,
                                  fovea_final_pred, fovea_roi_init_pred, fovea_roi_final_pred,
                                  prefix)

            if debug_all:
                logger.info("Compare target {}: target:{} VS HR prediction:{} VS LR prediction:{} " \
                            .format(meta['image'], fovea_target, fovea_hr_init_pred, fovea_lr_init_pred))

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_lr_init_dist',
                lr_init_dists.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_hr_init_dist',
                hr_init_dists.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_final_dist',
                final_dists.avg,
                global_steps
            )
            writer_dict['valid_global_steps'] = global_steps + 1

    # TODO - show results by different set in one val_dataset
    if db_vals:
        lr_init_avg_l2_dist = hr_init_avg_l2_dist = final_avg_l2_dist = 0
        sum_db = 0

        individual_db_images = [400]
        for idx, dataset_iter in enumerate(db_vals):
            each_db_images = individual_db_images[idx]
            lr_tmp_avg_l2_dist = dataset_iter.evaluate(all_fovea_lr_init_preds[sum_db*each_db_images:(sum_db+1)*each_db_images], output_dir='./')
            lr_init_avg_l2_dist += lr_tmp_avg_l2_dist
            hr_tmp_avg_l2_dist = dataset_iter.evaluate(all_fovea_hr_init_preds[sum_db*each_db_images:(sum_db+1)*each_db_images], output_dir=output_dir)
            hr_init_avg_l2_dist += hr_tmp_avg_l2_dist
            final_tmp_avg_l2_dist = dataset_iter.evaluate(all_fovea_final_preds[sum_db*each_db_images:(sum_db+1)*each_db_images], output_dir=None)
            final_avg_l2_dist += final_tmp_avg_l2_dist

            sum_db += 1
            logger.info('Dataset %d Average L2 Distance on test set: lr_L2 = %.2f, hr_L2 = %.2f' % (
                    sum_db, lr_tmp_avg_l2_dist, hr_tmp_avg_l2_dist))

        if sum_db == 0:
            sum_db = 1
        lr_init_avg_l2_dist /= sum_db
        hr_init_avg_l2_dist /= sum_db
        final_avg_l2_dist /= sum_db
    else:
        lr_init_avg_l2_dist = val_dataset.evaluate(all_fovea_lr_init_preds, output_dir='./', debug_enable=debug_all)
        hr_init_avg_l2_dist = val_dataset.evaluate(all_fovea_hr_init_preds, output_dir=output_dir, debug_enable=debug_all)
        final_avg_l2_dist = val_dataset.evaluate(all_fovea_final_preds, output_dir=None)

    # logger.info('Average L2 Distance on test set: lr_init = %.2f, hr_init = %.2f, final = %.2f' %(
    #     lr_init_avg_l2_dist, hr_init_avg_l2_dist, final_avg_l2_dist))
    logger.info('Average L2 Distance on evaluation: lr_L2 = %.2f, hr_L2 = %.2f' %(lr_init_avg_l2_dist, hr_init_avg_l2_dist))

    return lr_init_avg_l2_dist, hr_init_avg_l2_dist, 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
