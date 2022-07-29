import math
import numpy as np
import re
from easydict import EasyDict as edict
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from models.segtran_shared import CrossAttFeatTrans, SegtranInitWeights
torch.set_printoptions(sci_mode=False)

# Assume the normalization is along the last dimension.
def extract_std_mean(x, x_normed):
    # 1st element in the last dim.
    x1_normed = torch.index_select(x_normed, -1, torch.tensor(1, device='cuda')).squeeze(-1)
    # 0th element in the last dim.
    x0_normed = torch.index_select(x_normed, -1, torch.tensor(0, device='cuda')).squeeze(-1)
    x1        = torch.index_select(x,        -1, torch.tensor(1, device='cuda')).squeeze(-1)
    x0        = torch.index_select(x,        -1, torch.tensor(0, device='cuda')).squeeze(-1)
    norm_invstd = (x1_normed - x0_normed) / (x1 - x0 + 1e-6)
    if torch.isnan(norm_invstd).sum() > 0:
        pdb.set_trace()
    # norm_mean: [B, 3, 50]. x0_normed = (x0 * norm_invstd) - norm_mean
    norm_mean = x0 * norm_invstd - x0_normed
    return norm_invstd, norm_mean

def manual_normalize(x, invstd, mean):
    if invstd.dim() < x.dim():
        invstd = invstd.unsqueeze(-1)
        mean   = mean.unsqueeze(-1)
        
    x_normed = x * invstd - mean
    return x_normed

class PolyformerLayer(SegtranInitWeights):
    def __init__(self, feat_dim, name='poly', chan_axis=1, num_attractors=256, use_residual=True, poly_do_layernorm=False):
        config = edict()
        config.in_feat_dim  = feat_dim
        config.feat_dim     = feat_dim
        config.min_feat_dim = feat_dim
        config.num_modes    = 2
        config.attn_clip    = 500
        config.cross_attn_score_scale       = 1.
        config.base_initializer_range       = 0.02
        config.hidden_dropout_prob          = 0.2
        config.attention_probs_dropout_prob = 0.2
        config.tie_qk_scheme    = 'loose'           # shared, loose, or none.
        config.ablate_multihead = False
        config.eval_robustness  = False
        config.pool_modes_feat  = 'softmax'     # softmax, max, mean, or none. With [] means keepdim=True.
        config.pool_modes_basis = 'feat'        # attn or feat
        config.mid_type         = 'shared'      # shared, private, or none.
        config.trans_output_type    = 'private' # shared or private.
        config.act_fun          = F.gelu
        config.apply_attn_early = True
        config.feattrans_lin1_idbias_scale  = 10
        config.query_idbias_scale   = 10
        
        super(PolyformerLayer, self).__init__(config)
        self.name           = name
        self.chan_axis      = chan_axis
        self.feat_dim       = feat_dim
        self.num_attractors = num_attractors
        self.use_residual   = use_residual
        # If disabling multi-mode expansion in in_ator_trans, performance will drop 1-2%.
        #config1 = copy.copy(config)
        #config1.num_modes = 1
        self.in_ator_trans  = CrossAttFeatTrans(config, name + '-in-squeeze')
        self.ator_out_trans = CrossAttFeatTrans(config, name + '-squeeze-out')
        self.attractors     = Parameter(torch.randn(1, self.num_attractors, self.feat_dim))
        self.infeat_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        self.poly_do_layernorm = poly_do_layernorm
        print("Polyformer layer: {}, {} channels, {} layernorm".format(name, feat_dim, 'with' if poly_do_layernorm else 'no'))
        
        self.apply(self.init_weights)
        # tie_qk() has to be executed after weight initialization.
        self.apply(self.tie_qk)
        self.apply(self.add_identity_bias)
                
    def forward(self, in_feat):
        B           = in_feat.shape[0]
        in_feat2    = in_feat.transpose(self.chan_axis, -1)
        # Using layernorm reduces performance by 1-2%. Maybe because in_feat has just passed through ReLU(),
        # and a naive layernorm makes zero activations non-zero.
        if self.poly_do_layernorm:
            in_feat2    = self.infeat_norm_layer(in_feat2)
        vfeat       = in_feat2.reshape((B, -1, self.feat_dim))
        
        batch_attractors = self.attractors.expand(B, -1, -1)
        new_batch_attractors = self.in_ator_trans(batch_attractors, vfeat)
        vfeat_out = self.ator_out_trans(vfeat, new_batch_attractors)
        vfeat_out = vfeat_out.transpose(self.chan_axis, -1)
        out_feat  = vfeat_out.reshape(in_feat.shape)
        
        if self.use_residual:
            mix_feat = in_feat2.transpose(self.chan_axis, -1) + out_feat
            out_feat = mix_feat
            
        return out_feat
