# ------------------------------------------------------------------------------
# Image/label transform.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch
import torch.nn.functional as F

def flip_torch(image, dim):
    image_np = image.cpu().numpy()
    image_np = np.flip(image_np, axis=dim).copy()
    image = torch.FloatTensor(image_np).to(image.device)
    return image

def fliplr_coord(coord, width):
    """
    flip coords
    """
    # Flip horizontal
    coord = coord.copy()
    coord[0] = width - coord[0] - 1

    return coord

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(
        center, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    src_w = output_size[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + src_w * shift
    src[1, :] = center + src_dir + src_w * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_matrix_indices(h, w):
    w_idx_mat, h_idx_mat = np.meshgrid(np.arange(0, h), np.arange(0, w))
    w_idx, h_idx = w_idx_mat.flatten().astype(np.int32), h_idx_mat.flatten().astype(np.int32)
    h_idx = h_idx.reshape(w, h).transpose()
    w_idx = w_idx.reshape(w, h).transpose()
    return h_idx, w_idx


'''
Resembling to Tensorflow's crop_and_resize implemented with grid_sample
'''
def crop_and_resize(image, center, output_size, scale=1):
    '''
    image: NCHW
    center: N * 2
    '''
    n, ch, h, w = image.size()

    c_idx, r_idx = get_matrix_indices(output_size, output_size)
    r_idx_zero_centered = torch.FloatTensor(r_idx) - output_size // 2
    c_idx_zero_centered = torch.FloatTensor(c_idx) - output_size // 2
    r_idx_zero_centered = r_idx_zero_centered.unsqueeze(0).expand(n, -1, -1) * scale
    c_idx_zero_centered = c_idx_zero_centered.unsqueeze(0).expand(n, -1, -1) * scale
    r_idx_zero_centered = r_idx_zero_centered.to(image.device)
    c_idx_zero_centered = c_idx_zero_centered.to(image.device)
    r_idx_centered = center[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, output_size, output_size).float() + r_idx_zero_centered
    c_idx_centered = center[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, output_size, output_size).float() + c_idx_zero_centered
    # to flows in [-1, 1]
    r_idx_centered = (r_idx_centered - h // 2) / (h // 2)
    c_idx_centered = (c_idx_centered - w // 2) / (w // 2)
    grid = torch.stack([c_idx_centered, r_idx_centered], dim=-1)

    crop = F.grid_sample(image, grid)

    return crop


def img_crop_with_pad(img, top, left, bottom, right):
    height, width = bottom - top, right - left
    if len(img.shape) == 3:
        ch, ih, iw = img.size()
    elif len(img.shape) == 4:
        _, ch, ih, iw = img.size()
    else:
        assert False, 'Unknown image type'
    nt, nl = max(top, 0), max(left, 0)
    nb, nr = min(ih, bottom), min(iw, right)
    ct = 0 if top >= 0 else -top
    cb = ct + height if bottom <= ih else ih - top
    cl = 0 if left >= 0 else -left
    cr = cl + width if right <= iw else iw - left

    if len(img.shape) == 3:
        cropped = torch.zeros((ch, height, width), dtype=img.dtype).to(img.device)
        cropped[:, ct:cb, cl:cr] = img[:, nt:nb, nl:nr]
    elif len(img.shape) == 4:
        cropped = torch.zeros((n, ch, height, width), dtype=img.dtype).to(img.device)
        cropped[:, :, ct:cb, cl:cr] = img[:, :, nt:nb, nl:nr]
    else:
        assert False, 'Unsupported image type.'

    return cropped

