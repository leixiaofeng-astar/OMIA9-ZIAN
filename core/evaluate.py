from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


def calc_batch_l2_dist(preds, target, reduce=False):
    '''
    preds, target: N * 2 numpy array
    '''
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    l2_dist = np.sqrt((preds[:, 0] - target[:, 0])**2 + (preds[:, 1] - target[:, 1])**2)
    if reduce:
        l2_dist = l2_dist.mean()
    return l2_dist

