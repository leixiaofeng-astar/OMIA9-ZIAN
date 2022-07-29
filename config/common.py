from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'fovera_net'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [1634, 1634]  # width * height
_C.MODEL.CROP_SIZE = [1536, 1536]  # width * height
_C.MODEL.PATCH_SIZE = [1024, 1024] # width * height
_C.MODEL.DS_FACTOR = 4
_C.MODEL.SIGMA = 2
_C.MODEL.SIGMA_ROI = 2
_C.MODEL.MAX_DS_OFFSET = 8 # in number of pixels at downsampled scale
_C.MODEL.MAX_OFFSET = 8 # in number of pixels at original scale
_C.MODEL.REGION_RADIUS = 128 # in number of pixels at image scale
_C.MODEL.ROI_SCALE = 1 # fine stage ROI size selection by scale
_C.MODEL.SELF_ATTEN = False  # enalbe or disable self-attention module
_C.MODEL.CO_ATTEN = False    # enalbe or disable co-attention module
_C.MODEL.TRIP_ROI = False    # enalbe or disable 3 ROI crop module
_C.MODEL.ROI_NUM = 2         # N of the multi-ROI crop module,it works when TRIP_ROI enabled
_C.MODEL.HRNET_ONLY = False  # only run coarse network
_C.MODEL.HRNET_TYPE = 0      # different single network in coarse stage
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.ROI_WEIGHT = 1.0
_C.LOSS.REGRESS_WEIGHT = 1.0

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = '/data/xiaofeng/miccai-refuge/data/'
_C.DATASET.DATASET = 'refuge'
_C.DATASET.TRAIN_SET = 'train+val'
_C.DATASET.TRAIN_SET_1 = 'train'
_C.DATASET.GAMMA_TRAIN = 'gamma_train'
_C.DATASET.TRAIN_SET_2 = 'val'
_C.DATASET.TRAIN_FOLD = 0 # 0 denotes using all the dataset
_C.DATASET.TEST_SET = 'test'
_C.DATASET.VAL_SET = 'val2'

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.2
_C.DATASET.ROT_FACTOR = 15.
_C.DATASET.SHIFT_FACTOR = 0.1

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_EXP = False
_C.TRAIN.AGE_DATA = False
_C.TRAIN.GAMMA_DATA = False

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.FULL_DATA = False
_C.TRAIN.EFF_NET = False
_C.TRAIN.DATA_CLAHE = False
_C.TRAIN.ROI_CLAHE = False
_C.TRAIN.MV_IDEA = False
_C.TRAIN.MV_IDEA_HM1 = False

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 8
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.FLIP_TEST = True
_C.TEST.RELEASE_TEST = False
_C.TEST.MODEL_FILE = ''
_C.TEST.DEBUG = False
_C.TEST.TRIAL_RUN = False
_C.TEST.TRIAL_DATASET = 1   # 1 is refuge-ext2, 2 is gamma-train, 3 is fixed part from gamma-train


# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.SAVE_HEATMAP_DS_GT = True
_C.DEBUG.SAVE_HEATMAP_DS_PRED = True
_C.DEBUG.SAVE_HEATMAP_ROI_GT = True
_C.DEBUG.SAVE_HEATMAP_ROI_PRED = True
_C.DEBUG.SAVE_FOVEA_FINAL_GT = True
_C.DEBUG.SAVE_FOVEA_FINAL_PRED = True
_C.DEBUG.SAVE_FOVEA_ROI_GT = True
_C.DEBUG.SAVE_FOVEA_ROI_INIT_PRED = True
_C.DEBUG.SAVE_FOVEA_ROI_FINAL_PRED = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

