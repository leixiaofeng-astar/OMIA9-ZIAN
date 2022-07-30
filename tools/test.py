'''
Instructions to use:
     python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L12dot8_2stage_finetune_AGE.pth

     python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best.pth
     python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best.pth TEST.RELEASE_TEST True

    python3 tools/test.py --cfg experiments/refuge.yaml MODEL.SELF_ATTEN False MODEL.TRIP_ROI False MODEL.HRNET_ONLY True TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best.pth TEST.RELEASE_TEST True

    # add or GAMMA test: TEST.TRIAL_RUN True
    python3 tools/test.py --cfg experiments/refuge.yaml TEST.TRIAL_RUN True TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best.pth
description:
    test the model

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import importlib

import torch
# from torchsummary import summary
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config.common import _C as cfg
from config.common import update_config
from core.loss import HybridLoss
from core.function import validate
from utils.utils import create_logger

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

"""
        Zhang Yu original -- mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
        before gray conv
        'mean': { 'train':  [0.449, 0.260, 0.130],
          'test':   [0.654, 0.458, 0.364],
          'valid':  [0.662, 0.459, 0.356],
          'valid2': [0.684, 0.384, 0.163] },
        'std':  { 'train':  [0.240, 0.151, 0.086],
          'test':   [0.215, 0.185, 0.144],
          'valid':  [0.217, 0.185, 0.144],
          'valid2': [0.210, 0.157, 0.127] },

        after gray conv
        'mean': { 'train':  [0.400, 0.298, 0.229],
                  'test':   [0.590, 0.490, 0.442],
                  'valid':  [0.596, 0.493, 0.440],
                  'valid2': [0.566, 0.416, 0.306] },
        'std':  { 'train':  [0.176, 0.140, 0.108],
                  'test':   [0.184, 0.174, 0.153],
                  'valid':  [0.184, 0.174, 0.152],
                  'valid2': [0.184, 0.159, 0.140] },

    before CLAHE
    train:
        mean = [0.084, 0.168, 0.282], std = [0.062, 0.110, 0.189]
        mean = [0.282, 0.168, 0.084], std = [0.189, 0.110, 0.062]
    val:
        mean = [0.215, 0.270, 0.409], std = [0.160, 0.203, 0.288]
        mean = [0.409, 0.270, 0.215], std = [0.288, 0.203, 0.160]
    test:
        mean = [0.222, 0.271, 0.404], std = [0.163, 0.202, 0.284]
        mean = [0.404, 0.271, 0.222], std = [0.284, 0.202, 0.163]
    val2:
        mean = [0.059, 0.208, 0.417], std = [0.079, 0.152, 0.273]
        mean = [0.417, 0.208, 0.059], std = [0.273, 0.152, 0.079]

    after CLAHE
        train: 
        mean = [0.134, 0.227, 0.316], std = [0.089, 0.142, 0.197]

        val:
        mean = [0.257, 0.303, 0.390], std = [0.186, 0.219, 0.268]

        test:
        mean = [0.262, 0.303, 0.386], std = [0.188, 0.218, 0.264]

        val2:
        mean = [0.128, 0.283, 0.422], std = [0.135, 0.187, 0.262]

"""

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model_builder = importlib.import_module("models." + cfg.MODEL.NAME).get_fovea_net
    model = model_builder(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        (filepath, tempfilename) = os.path.split(cfg.TEST.MODEL_FILE)
        if "checkpoint" in tempfilename:
            # workaround to load python2 model -- Note: the final result could be different
            if 'P2' in tempfilename:
                checkpoint = torch.load(cfg.TEST.MODEL_FILE, encoding='latin1')
            else:
                checkpoint = torch.load(cfg.TEST.MODEL_FILE)
            model.load_state_dict(checkpoint['best_state_dict'])
        else:
            model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        checkpoint_file = os.path.join(
            final_output_dir, 'checkpoint_final_state.pth'
        )

        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['best_state_dict'])

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # get the model parameters with summary
    # summary(model, input_size=(3, 224, 224))
    count_parameters(model)

    # define loss function (criterion) and optimizer
    criterion = HybridLoss(
        roi_weight=cfg.LOSS.ROI_WEIGHT,
        regress_weight=cfg.LOSS.REGRESS_WEIGHT,
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    TRAIN_AGE_MODEL = cfg.TRAIN.AGE_DATA
    TRAIN_GAMMA_MODEL = cfg.TRAIN.GAMMA_DATA

    if TRAIN_AGE_MODEL:
        normalize = transforms.Normalize(
            # mean=[0.070, 0.071, 0.152], std=[0.127, 0.127, 0.138]
            mean=[0.152, 0.071, 0.070], std=[0.138, 0.127, 0.127]
        )
        # TEST_SET is AGE test dataset with dummy GT
        valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
            cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif TRAIN_GAMMA_MODEL:
        gamma_val_release = True
        gamma_final_challenge = True
        if gamma_val_release:
            if not gamma_final_challenge:
                # GAMMA validation dataset (100 images)
                normalize = transforms.Normalize(
                    mean=[0.265, 0.130, 0.036], std=[0.293, 0.154, 0.069]
                )
                valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                    cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            else:
                # TODO: replace it with final release dataset
                # GAMMA final test dataset (X images)
                # python3 xf_get_mean_std.py --dirs ../data/train/test
                # add dummy file with correct filename in /data/xiaofeng/miccai-refuge/data/GAMMA/fovea_localization_test_GT.xlsx
                normalize = transforms.Normalize(
                    mean=[0.265, 0.130, 0.036], std=[0.293, 0.154, 0.069]
                )
                valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                    cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )

        else:
            # 3: partial of GAMMA train dataset (15 out of 100 images)
            normalize = transforms.Normalize(
                # it is 100 gamma images mean/std
                mean=[0.259, 0.129, 0.038], std=[0.291, 0.156, 0.072]
            )
            valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET_2, False,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )
    else:
        # TRAIN_FOVEA_MODEL or GAMMA_DATA
        release_result = cfg.TEST.RELEASE_TEST
        if release_result:
            # it is for result submission
            trial_enable = cfg.TEST.TRIAL_RUN
            trial_db_index = cfg.TEST.TRIAL_DATASET
            if trial_enable is False:   # it is normal processing for Refuge and Paper
                normalize = transforms.Normalize(
                    # mean=[0.134, 0.207, 0.330], std=[0.127, 0.160, 0.239]
                    # mean=[0.417, 0.208, 0.059], std=[0.273, 0.152, 0.079]
                    # mean=[0.350, 0.224, 0.158], std=[0.255, 0.175, 0.143]
                    # mean=[0.369, 0.220, 0.130], std=[0.262, 0.169, 0.136]    # get from 1600 images
                    mean=[0.158, 0.224, 0.350], std=[0.143, 0.175, 0.255]
                )
                # TEST_SET comes from refuge2 semi-final test 400 images
                valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                    cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                    transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                )
            else:
                if trial_db_index == 1:  # fovea ext2 dataset
                    normalize = transforms.Normalize(
                        # prepare for refuge2 final submission - 'Refuge2-Ext'
                        mean=[0.435, 0.211, 0.070], std=[0.310, 0.166, 0.085]
                    )
                    # VAL_SET is 400 images from refuge2 external 400 images (Refuge2-Ext-GT)
                    # Note: it was automatically decided by cfg.TEST.TRIAL_RUN function in refuge.py file
                    valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                        cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
                        transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ])
                    )

                elif trial_db_index == 2 or trial_db_index == 3:
                    # 2: GAMMA train dataset
                    # 3: partial of GAMMA train dataset (15 out of 100 images)
                    normalize = transforms.Normalize(
                        # it is 100 gamma images
                        mean=[0.259, 0.129, 0.038], std=[0.291, 0.156, 0.072]
                    )
                    valid_dataset = importlib.import_module('dataset.' + cfg.DATASET.DATASET).Dataset(
                        cfg, cfg.DATASET.ROOT, cfg.DATASET.GAMMA_TRAIN, False,
                        transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                        ])
                    )
                else:
                    raise ("Wait for GAMMA data release!!!")
        else:
            normalize = transforms.Normalize(
                # 20% images from refuge1 800 images
                # mean=[0.350, 0.224, 0.158], std=[0.255, 0.175, 0.143]
                # mean=[0.369, 0.220, 0.130], std=[0.262, 0.169, 0.136]
                mean=[0.158, 0.224, 0.350], std=[0.143, 0.175, 0.255]
            )
            # VAL_SET is 240 images from refuge1 Validation 400 images (REFUGE-Val2Val)
            valid_dataset = importlib.import_module('dataset.'+cfg.DATASET.DATASET).Dataset(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.VAL_SET, False,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
            )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    debug_enable = cfg.TEST.DEBUG
    # evaluate on validation set
    if debug_enable:
        debug_dir = './debug'
        if not os.path.isdir(str(debug_dir)):
            os.mkdir(debug_dir)
        validate(cfg, valid_loader, valid_dataset, model, criterion, debug_dir, tb_log_dir, debug_all=debug_enable)
    else:
        validate(cfg, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir)



if __name__ == '__main__':
    main()
    print("SATA Test Program Exit ... \n")
