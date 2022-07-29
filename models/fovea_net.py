
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import cv2
import copy
import numpy as np

from core.inference import get_max_preds, get_heatmap_center_preds
from utils.transforms import crop_and_resize
from models.polyformer import PolyformerLayer
from models.unet_parts import *
from models.models import AG_Net
from models.efficientunet import *


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

bb2feat_dims = { 'resnet34':  [64, 64,  128, 256,  512],
                 'resnet50':  [64, 256, 512, 1024, 2048],
                 'resnet101': [64, 256, 512, 1024, 2048],
                 'eff-b0':    [16, 24,  40,  112,  1280],   # input: 224
                 'eff-b1':    [16, 24,  40,  112,  1280],   # input: 240
                 'eff-b2':    [16, 24,  48,  120,  1408],   # input: 260
                 'eff-b3':    [24, 32,  48,  136,  1536],   # input: 300
                 'eff-b4':    [24, 32,  56,  160,  1792],   # input: 380
               }

class ResNet18(nn.Module):
    def __init__(self, num_class=2):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.features(x).view(x.shape[0], -1)
        # out = torch.sigmoid(self.fc(features))
        # out1 = torch.softmax(self.fc(features), dim=1)
        return features
        # return features, out, out1
'''
U-Net
'''
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # return last layer feature for feature confusion
        return x, logits


"""
ResNet34 + U-Net
"""
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_Unet(nn.Module):

    def __init__(self, in_channel, out_channel, pretrained=False):
        super(Resnet34_Unet, self).__init__()

        self.resnet = models.resnet34(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            # Note: disable layer0 pool to keep size same
            self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decode
        self.conv_decode4 = expansive_block(1024+512, 512, 512)
        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    def forward(self, x):
        x = self.layer0(x)
        # Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        final_layer = self.final_layer(decode_block0)

        return final_layer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            try:
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
            except:
                import pdb
                pdb.set_trace()
                print(y.shape, (self.fuse_layers[i][j](x[j])).shape)
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=256,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)


        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_hrnet(cfg, is_train, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model

def get_all_indices(shape):
    indices = torch.arange(shape.numel()).view(shape)
    # indices = indices.cuda()

    out = []
    for dim in reversed(shape):
        out.append(indices % dim)
        indices = indices // dim
    return torch.stack(tuple(reversed(out)), len(shape))


class CoattentionModel(nn.Module):
    def __init__(self, block, layers, num_classes, all_channel=256, all_dim=60 * 60):  # 473./8=60
        super(CoattentionModel, self).__init__()
        self.encoder = ResNet(block, layers, num_classes)
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.bn2 = nn.BatchNorm2d(all_channel)
        self.prelu = nn.ReLU(inplace=True)
        self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        self.softmax = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input1, input2):  # 注意input2 可以是多帧图像

        # input1_att, input2_att = self.coattention(input1, input2)
        input_size = input1.size()[2:]
        exemplar, temp = self.encoder(input1)
        query, temp = self.encoder(input2)
        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        A1 = F.softmax(A.clone(), dim=1)  #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        input2_att = input2_att * input2_mask
        input1_att = torch.cat([input1_att, exemplar], 1)
        input2_att = torch.cat([input2_att, query], 1)
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        input1_att = self.bn1(input1_att)
        input2_att = self.bn2(input2_att)
        input1_att = self.prelu(input1_att)
        input2_att = self.prelu(input2_att)
        x1 = self.main_classifier1(input1_att)
        x2 = self.main_classifier2(input2_att)
        x1 = F.upsample(x1, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        x2 = F.upsample(x2, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        # print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)
        x2 = self.softmax(x2)

        #        x1 = self.softmax(x1)
        #        x2 = self.softmax(x2)
        return x1, x2, temp  # shape: NxCx


class FoveaNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FoveaNet, self).__init__()
        self.cfg = cfg

        self.trip_roi = cfg.MODEL.TRIP_ROI
        self.roi_num = cfg.MODEL.ROI_NUM
        self.hrnet_only = cfg.MODEL.HRNET_ONLY
        self.hrnet_type = cfg.MODEL.HRNET_TYPE
        self.feature_ch = 16
        # TODO later
        self.debug_iteration = 0
        # self.roi_feature_ch = 64

        if self.hrnet_type == 0:
            # hrnet
            self.feature_ch = 16
            logger.info('=> We use HRNET in coarse stage network')
            self.hrnet = get_hrnet(cfg, is_train=False, **kwargs)
        elif self.hrnet_type == 1:
            #unet
            self.feature_ch = 64
            logger.info('=> We use UNET in coarse stage network')
            self.Unet = UNet(n_channels=3, n_classes=1)
        elif self.hrnet_type == 2:
            #efficientunet B0
            self.feature_ch = 32
            logger.info('=> We use efficientunet B0 in coarse stage network')
            self.Eff_Unet = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True)
        elif self.hrnet_type == 3:
            # TODO
            logger.info('=> We use AG_Net in coarse stage network')
            self.AG_Net = AG_Net(n_classes=1)
        elif self.hrnet_type == 4:
            # efficientunet B5
            self.feature_ch = 32
            logger.info('=> We use efficientunet B5 in coarse stage network')
            self.Eff_Unet = get_efficientunet_b5(out_channels=1, concat_input=True, pretrained=True)

        if not self.hrnet_only:
            # we apply one model for 1 ROI model first
            if self.hrnet_type == 0:
                self.hrnet_roi = get_hrnet(cfg, is_train=False, **kwargs)
            elif self.hrnet_type == 1:
                # unet
                self.Unet_1 = UNet(n_channels=3, n_classes=1)
            elif self.hrnet_type == 2:
                # efficientunet
                self.Eff_Unet_1 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)
            elif self.hrnet_type == 3:
                # TODO AG_Net
                self.AG_Net_1 = AG_Net(n_classes=1)
            elif self.hrnet_type == 4:
                # efficientunet B0 instead B5 in fine network
                self.Eff_Unet_1 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)

        self.roi_scale = cfg.MODEL.ROI_SCALE
        # self.subpixel_up_by8 = nn.PixelShuffle(8)
        self.subpixel_up_by4 = nn.PixelShuffle(4)
        self.subpixel_up_by2 = nn.PixelShuffle(2)
        self.heatmap_ds = nn.Sequential(
            nn.Conv2d(self.feature_ch, 1, kernel_size=1, padding=0)
        )

        # we apply the other 2 models for 3 ROI model
        if self.trip_roi is True:
            if self.hrnet_type == 0:
                # unet
                if self.roi_num != 2:
                    self.hrnet_roi_2 = get_hrnet(cfg, is_train=False, **kwargs)
                self.hrnet_roi_3 = get_hrnet(cfg, is_train=False, **kwargs)
            elif self.hrnet_type == 1:
                # unet
                if self.roi_num != 2:
                    self.Unet_2 = UNet(n_channels=3, n_classes=1)
                self.Unet_3 = UNet(n_channels=3, n_classes=1)
            elif self.hrnet_type == 2:
                # efficientunet B0
                if self.roi_num != 2:
                    self.Eff_Unet_2 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)
                self.Eff_Unet_3 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)
            elif self.hrnet_type == 3:
                # TODO
                self.AG_Net_2 = AG_Net(n_classes=1)
                self.AG_Net_3 = AG_Net(n_classes=1)
            elif self.hrnet_type == 4:
                # efficientunet B5, fine network use B0 network
                if self.roi_num != 2:
                    self.Eff_Unet_2 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)
                self.Eff_Unet_3 = get_efficientunet_b0(out_channels=1, concat_input=False, pretrained=True)

        self.isAtten = cfg.MODEL.SELF_ATTEN
        self.coAtten = cfg.MODEL.CO_ATTEN
        self.simpleCA = False  # Note: Fovea by default setting
        self.coAtten_scale = 4.0
        self.add_heatmap_channel = False
        self.image_channel = 3

        # if self.add_heatmap_channel:
        #     self.image_channel += 1

        # the self-attention module
        if self.coAtten:
            if self.simpleCA:
                self.coatten_conv = nn.Conv2d(self.feature_ch * 2, self.feature_ch, kernel_size=3, padding=1,
                                               bias=False)
                # self.coatten_conv2 = nn.Conv2d(self.feature_ch * 2, self.feature_ch, kernel_size=3, padding=1,
                #                                bias=False)
                self.coatten_bn = nn.BatchNorm2d(self.feature_ch)
                # self.coatten_bn2 = nn.BatchNorm2d(self.feature_ch)
                self.coatten_prelu = nn.ReLU(inplace=True)
                self.softmax = nn.Sigmoid()
            else:
                self.linear_e = nn.Linear(self.feature_ch, self.feature_ch, bias=False)
                self.coatten_gate = nn.Conv2d(self.feature_ch, 1, kernel_size=1, bias=False)
                self.coatten_gate_s = nn.Sigmoid()
                self.coatten_conv1 = nn.Conv2d(self.feature_ch * 2, self.feature_ch, kernel_size=3, padding=1, bias=False)
                self.coatten_conv2 = nn.Conv2d(self.feature_ch * 2, self.feature_ch, kernel_size=3, padding=1, bias=False)
                self.coatten_bn1 = nn.BatchNorm2d(self.feature_ch)
                self.coatten_bn2 = nn.BatchNorm2d(self.feature_ch)
                self.coatten_prelu = nn.ReLU(inplace=True)
                self.main_classifier1 = nn.Conv2d(self.feature_ch, 1, kernel_size=1, bias=True)
                self.main_classifier2 = nn.Conv2d(self.feature_ch, 1, kernel_size=1, bias=True)
                self.softmax = nn.Sigmoid()

        # the self-attention module
        if self.isAtten:
            # self.attention = PolyformerLayer(self.feature_ch * 4)
            if self.trip_roi is True:
                # TODO: depends on the multi-ROI number
                self.attention = PolyformerLayer(self.feature_ch * (self.roi_num+1))
            else:
                self.attention = PolyformerLayer(self.feature_ch * 2)

        self.relu = nn.ReLU(inplace=True)
        if self.trip_roi is True:
            # fusion layer
            self.convf = nn.Conv2d(self.feature_ch*(self.roi_num+1), self.feature_ch*(self.roi_num+1), kernel_size=3, stride=1, padding=1, bias=False)
            self.bnf = nn.BatchNorm2d(self.feature_ch*(self.roi_num+1), momentum=BN_MOMENTUM)
            # heatmap layer
            self.heatmap_roi = nn.Sequential(
                nn.Conv2d(self.feature_ch*(self.roi_num+1), self.feature_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.feature_ch, momentum=BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2d(self.feature_ch, 1, kernel_size=1, stride=1, padding=0),
            )
        elif not self.hrnet_only:
            # one ROI case here
            # fusion layer
            self.convf = nn.Conv2d(self.feature_ch * 2, self.feature_ch * 2, kernel_size=3, stride=1, padding=1, bias=False)
            self.bnf = nn.BatchNorm2d(self.feature_ch * 2, momentum=BN_MOMENTUM)
            # heatmap layer
            self.heatmap_roi = nn.Sequential(
                nn.Conv2d(self.feature_ch * 2, self.feature_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.feature_ch, momentum=BN_MOMENTUM),
                nn.ReLU(),
                nn.Conv2d(self.feature_ch, 1, kernel_size=1, stride=1, padding=0),
            )

    def init_weights(self, pretrained):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if self.hrnet_type == 0:
            if os.path.isfile(pretrained):
                # Initialize low-resolution branch
                self.hrnet.init_weights(pretrained)

                # Initialize high-resolution branch
                if not self.hrnet_only:
                    self.hrnet_roi.init_weights(pretrained)
                # 3 ROI crop module
                if self.trip_roi is True and self.hrnet_type == 0:
                    if self.roi_num != 2:
                        self.hrnet_roi_2.init_weights(pretrained)
                    self.hrnet_roi_3.init_weights(pretrained)

                # Initialize high-resolution branch
                need_init_state_dict = {}
                pretrained_state_dict = torch.load(pretrained)
                for name, m in pretrained_state_dict.items():
                    cond1 = 'conv1' in name or 'conv2' in name or 'bn1' in name or 'bn2' in name
                    cond2 = 'stage' in name or 'layer' in name or 'head' in name or 'transition' in name
                    if cond1 and not cond2:
                        need_init_state_dict[name] = m
                self.load_state_dict(need_init_state_dict, strict=False)
            elif pretrained:
                logger.error('=> please download pre-trained models first!')
                raise ValueError('{} is not exist!'.format(pretrained))

    def co_atten_forward(self, exemplar, query):
        if self.simpleCA:
            # import pdb
            # pdb.set_trace()
            fea_size = query.size()[2:]
            all_dim = fea_size[0] * fea_size[1]
            exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
            query_flat = query.view(-1, query.size()[1], all_dim)
            exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
            A = torch.bmm(exemplar_t, query_flat)
            A1 = F.softmax(A.clone(), dim=1)  # WH x WH
            B = F.softmax(torch.transpose(A, 1, 2), dim=1)
            query_att = torch.bmm(exemplar_flat, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
            exemplar_att = torch.bmm(query_flat, B).contiguous()

            input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
            input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])

            input1_att = torch.cat([input1_att, exemplar], 1)
            input2_att = torch.cat([input2_att, query], 1)
            input1_att = self.coatten_prelu(self.coatten_bn(self.coatten_conv(input1_att)))
            input2_att = self.coatten_prelu(self.coatten_bn(self.coatten_conv(input2_att)))
        else:
            fea_size = query.size()[2:]
            all_dim = fea_size[0] * fea_size[1]
            exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
            query_flat = query.view(-1, query.size()[1], all_dim)
            exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
            exemplar_corr = self.linear_e(exemplar_t)  #
            A = torch.bmm(exemplar_corr, query_flat)
            A1 = F.softmax(A.clone(), dim=1)  #
            B = F.softmax(torch.transpose(A, 1, 2), dim=1)
            query_att = torch.bmm(exemplar_flat, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
            exemplar_att = torch.bmm(query_flat, B).contiguous()

            input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
            input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
            input1_mask = self.coatten_gate(input1_att)
            input2_mask = self.coatten_gate(input2_att)
            input1_mask = self.coatten_gate_s(input1_mask)
            input2_mask = self.coatten_gate_s(input2_mask)
            input1_att = input1_att * input1_mask
            input2_att = input2_att * input2_mask
            input1_att = torch.cat([input1_att, exemplar], 1)
            input2_att = torch.cat([input2_att, query], 1)
            input1_att = self.coatten_conv1(input1_att)
            input2_att = self.coatten_conv2(input2_att)
            input1_att = self.coatten_bn1(input1_att)
            input2_att = self.coatten_bn2(input2_att)
            input1_att = self.coatten_prelu(input1_att)
            input2_att = self.coatten_prelu(input2_att)

        return input1_att, input2_att  # shape: NxCx


    def forward(self, input, meta, input_roi=None):
        infer_roi = input_roi is None
        ds_factor = self.cfg.MODEL.DS_FACTOR
        ds_scale_feature_enable = True

        # Low-resolution branch
        batch, _, ih, iw = input.size()   # train input is CROP_SIZExCROP_SIZE / AGE 768x768
        nh = int(ih * 1.0 / ds_factor)
        nw = int(iw * 1.0 / ds_factor)
        input_ds = F.upsample(input, size=(nh, nw), mode='bilinear', align_corners=True)

        if self.hrnet_type == 0:
            # HRNET
            input_ds_feats_orig = self.hrnet(input_ds)  # (batch, 256, 64, 64)
            # 256 channel --> 16 channel, HRNET resolution: (batch, 256, 64, 64) --> (20, 16, 256, 256)
            input_ds_feats = self.subpixel_up_by4(input_ds_feats_orig)
            # heatmap_ds: 16 channel --> 1 channel / (batch, 16, 256, 256) --> (batch, 1, 256, 256)
            heatmap_ds_pred = self.heatmap_ds(input_ds_feats)
        elif self.hrnet_type == 1:
            # use 64 channel last layer feature here
            input_ds_feats_orig, input_ds_feats = self.Unet(input_ds)  # (batch, 64, H, W):[4, 64, 224, 224]/[4, 1, 224, 224]
            # 1 channel
            heatmap_ds_pred = input_ds_feats
        elif self.hrnet_type == 2:
            # use 32 channel last layer feature here
            input_ds_feats_orig, input_ds_feats = self.Eff_Unet(input_ds)  # (batch, 64, H, W):[4, 64, 224, 224]/[4, 1, 224, 224]
            # 1 channel
            heatmap_ds_pred = input_ds_feats
        elif self.hrnet_type == 3:
            # TODO, it still need to debug
            raise ("AG_Net not supported yet")
            input_ds_feats = self.AG_Net(input_ds)  # (batch, 64, H/2, W/2)
            # 1 channel
            heatmap_ds_pred = input_ds_feats
        elif self.hrnet_type == 4:
            # use 32 channel last layer feature here
            input_ds_feats_orig, input_ds_feats = self.Eff_Unet(input_ds)
            # 1 channel
            heatmap_ds_pred = input_ds_feats

        # One-stage network returns directly
        if self.hrnet_only:
            # Fill in the dummy data
            region_size = 2 * self.cfg.MODEL.REGION_RADIUS
            # B = heatmap_ds_pred.shape[0]
            heatmap_roi_pred = torch.FloatTensor(np.zeros((batch, 1, region_size, region_size), dtype=np.float32))
            offset_in_roi_pred = torch.from_numpy(np.tile(np.array([-1, -1], np.float32), (batch, 1))).cuda()
            meta.update(
                {'roi_center': offset_in_roi_pred.cpu(),
                 'input_roi': heatmap_roi_pred.cpu()
                 })
            return heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta
        # One-stage network returns completely

        if ds_scale_feature_enable:
            roi_scale_1 = 1.0
            roi_scale_2 = 1.5
            roi_scale_3 = 2.0
            roi_feature_scale_1 = 1.0
            roi_feature_scale_2 = 1.0/1.5
            roi_feature_scale_3 = 1.0/2.0
            ds_feature_scale = 1.0/ds_factor
        else:
            roi_scale_1 = 1.0
            roi_scale_2 = 1.5
            roi_scale_3 = 2.0
            roi_feature_scale_1 = 1.0
            roi_feature_scale_2 = 1.0
            roi_feature_scale_3 = 1.0
            ds_feature_scale = 1.0

        # ********************** Note: fine stage network here **********************
        region_size = 2 * self.cfg.MODEL.REGION_RADIUS
        if infer_roi:
            # Get the predicted ROI
            roi_center = get_max_preds(heatmap_ds_pred.cpu().numpy())[0][:, 0, :]
            roi_center = torch.FloatTensor(roi_center)
            roi_center *= ds_factor
            roi_center = roi_center.cuda(non_blocking=True)
            # print("roi_center: {} with scale {}" .format(roi_center, ds_factor))

            # 3 ROI crop module
            if self.trip_roi is True:
                input_roi = []

                input_roi.append(crop_and_resize(input, roi_center, region_size, scale=roi_scale_1))
                if self.roi_num != 2:
                    input_roi.append(crop_and_resize(input, roi_center, region_size, scale=roi_scale_2))
                input_roi.append(crop_and_resize(input, roi_center, region_size, scale=roi_scale_3))

                # for i in range(len(input_roi)):
                #     input_roi[i] = input_roi[i].cuda(non_blocking=True)
                    # cuda = torch.device('cuda')
                    # input_roi[i] = input_roi[i].to(device=cuda)
                    # input_roi[i] = input_roi[i].cpu()
                # meta.update(
                #     {'roi_center': roi_center.cpu(),
                #      'input_roi': input_roi[0].cpu()
                #      })
                meta.update(
                    {'roi_center': roi_center.cpu(),
                     'input_roi': input_roi[0].cpu()
                     })
            else:
                input_roi = crop_and_resize(input, roi_center, region_size, scale=roi_scale_1)

                meta.update(
                    {'roi_center': roi_center.cpu(),
                     'input_roi': input_roi.cpu()
                     })

        else:
            # Note: train stage, it has GT for ROI
            assert 'roi_center' in meta.keys()
            roi_center = meta['roi_center'].cuda(non_blocking=True)   # roi center with random offset

        # if self.add_heatmap_channel:
        #     heatmap_2_roilayer = crop_and_resize(heatmap_ds_pred, roi_center/ds_factor, region_size, scale=1. / ds_factor)
        #     input_roi = torch.stack([input_roi, heatmap_2_roilayer], dim=1)

        # (batch, 16, 256, 256)
        # 3 channel --> 64 channel (batch, 3, 256, 256) --> (batch, 64, 128, 128)
        # 3 ROI crop module
        if self.trip_roi is True:

            multi_roi_center = np.empty(shape=(batch, 2))
            multi_roi_center.fill(region_size/2-1)
            multi_roi_center = torch.from_numpy(multi_roi_center)
            multi_roi_center = multi_roi_center.cuda(non_blocking=True)

            if self.hrnet_type == 0:
                # HRNET
                # 1st ROI 1:1
                roi_feats_hr_1 = self.hrnet_roi(input_roi[0])  # (batch, 256, 64, 64)
                # roi_feats_hr_1 = self.subpixel_up_by2(roi_feats_hr_1)  # 256 -> 64 channel
                # roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                # 256 channel --> 16 channel
                roi_feats_hr_1 = self.subpixel_up_by4(roi_feats_hr_1)
                roi_feats_hr_1 = crop_and_resize(roi_feats_hr_1, multi_roi_center, region_size, scale=roi_feature_scale_1)

                # 2nd ROI 1.5:1
                if self.roi_num == 2:
                    # 3rd ROI 2:1
                    roi_feats_hr_3 = self.hrnet_roi_3(input_roi[1])  # (batch, 256, 64, 64)
                    # 256 channel --> 16 channel
                    roi_feats_hr_3 = self.subpixel_up_by4(roi_feats_hr_3)
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size, scale=roi_feature_scale_3)
                else:
                    roi_feats_hr_2 = self.hrnet_roi_2(input_roi[1])  # (batch, 256, 64, 64)
                    # 256 channel --> 16 channel
                    roi_feats_hr_2 = self.subpixel_up_by4(roi_feats_hr_2)
                    roi_feats_hr_2 = crop_and_resize(roi_feats_hr_2, multi_roi_center, region_size, scale=roi_feature_scale_2)

                    # 3rd ROI 2:1
                    roi_feats_hr_3 = self.hrnet_roi_3(input_roi[2])  # (batch, 256, 64, 64)
                    # 256 channel --> 16 channel
                    roi_feats_hr_3 = self.subpixel_up_by4(roi_feats_hr_3)
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size, scale=roi_feature_scale_3)

                # Note: concat features from 3 ROIs
                if self.roi_num == 2:
                    # TODO : add 2 more heatmap for ROI roi_feats_lr
                    # the co-attention module
                    if self.coAtten:
                        # down-sample to save memory
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, int(region_size/self.coAtten_scale), mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, int(region_size/self.coAtten_scale), mode="bilinear")
                        roi_feats_hr_1, roi_feats_hr_3 = self.co_atten_forward(roi_feats_hr_1, roi_feats_hr_3)
                        # up-sample to align
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")

                        # TODO: add 2 new heatmap for sub-ROI loss calculation
                        heatmap_subroi_pred = []
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_1))
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_3))
                    # end -- co-attention

                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_3], dim=1)
                else:
                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_2, roi_feats_hr_3], dim=1)

            elif self.hrnet_type == 1:
                # U-Net
                # 1st ROI 1:1
                roi_feats_hr_1, _ = self.Unet_1(input_roi[0])  # (batch, 64, H, W)
                roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                # roi_feats_hr_1 = crop_and_resize(roi_feats_hr_1, multi_roi_center, region_size, scale=roi_feature_scale_1)

                # 2nd ROI 1.5:1
                if self.roi_num == 2:
                    # 2nd ROI 2:1
                    roi_feats_hr_3, _ = self.Unet_3(input_roi[1])  # (batch, 64, H, W)
                    roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size, scale=roi_feature_scale_3)
                else:
                    roi_feats_hr_2, _ = self.Unet_2(input_roi[1])  # (batch, 64, H, W)
                    roi_feats_hr_2 = F.interpolate(roi_feats_hr_2, region_size, mode="bilinear")
                    roi_feats_hr_2 = crop_and_resize(roi_feats_hr_2, multi_roi_center, region_size, scale=roi_feature_scale_2)

                    # 3rd ROI 2:1
                    roi_feats_hr_3, _ = self.Unet_3(input_roi[2])  # (batch, 64, H, W)
                    roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size, scale=roi_feature_scale_3)

                # Note: concat features from N ROIs
                if self.roi_num == 2:
                    # TODO : add 2 more heatmap for ROI roi_feats_lr
                    # the co-attention module
                    if self.coAtten:
                        # down-sample to save memory
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, int(region_size//self.coAtten_scale), mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, int(region_size//self.coAtten_scale), mode="bilinear")
                        roi_feats_hr_1, roi_feats_hr_3 = self.co_atten_forward(roi_feats_hr_1, roi_feats_hr_3)
                        # up-sample to align
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")

                        # TODO: add 2 new heatmap for sub-ROI loss calculation
                        heatmap_subroi_pred = []
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_1))
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_3))
                    # end -- co-attention

                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_3], dim=1)
                else:
                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_2, roi_feats_hr_3], dim=1)


            elif self.hrnet_type == 2 or self.hrnet_type == 4:
                # efficientunet B0 or B5
                # 1st ROI 1:1
                roi_feats_hr_1, _ = self.Eff_Unet_1(input_roi[0])  # (batch, 32, H, W)
                roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                # roi_feats_hr_1 = crop_and_resize(roi_feats_hr_1, multi_roi_center, region_size, scale=roi_feature_scale_1)

                # 2nd ROI 1.5:1
                if self.roi_num == 2:
                    # 2nd ROI 2:1
                    roi_feats_hr_3, _ = self.Eff_Unet_3(input_roi[1])  # (batch, 32, H, W)
                    roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size,
                                                     scale=roi_feature_scale_3)
                else:
                    roi_feats_hr_2, _ = self.Eff_Unet_2(input_roi[1])  # (batch, 32, H, W)
                    roi_feats_hr_2 = F.interpolate(roi_feats_hr_2, region_size, mode="bilinear")
                    roi_feats_hr_2 = crop_and_resize(roi_feats_hr_2, multi_roi_center, region_size,
                                                     scale=roi_feature_scale_2)

                    # 3rd ROI 2:1
                    roi_feats_hr_3, _ = self.Eff_Unet_3(input_roi[2])  # (batch, 32, H, W)
                    roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")
                    roi_feats_hr_3 = crop_and_resize(roi_feats_hr_3, multi_roi_center, region_size,
                                                     scale=roi_feature_scale_3)

                # Note: concat features from N ROIs
                if self.roi_num == 2:
                    # TODO : add 2 more heatmap for ROI roi_feats_lr
                    # the co-attention module
                    self.debug_iteration += 1
                    if self.coAtten:
                        # down-sample to save memory
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, int(region_size // self.coAtten_scale),
                                                       mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, int(region_size // self.coAtten_scale),
                                                       mode="bilinear")
                        roi_feats_hr_1, roi_feats_hr_3 = self.co_atten_forward(roi_feats_hr_1, roi_feats_hr_3)
                        # up-sample to align
                        roi_feats_hr_1 = F.interpolate(roi_feats_hr_1, region_size, mode="bilinear")
                        roi_feats_hr_3 = F.interpolate(roi_feats_hr_3, region_size, mode="bilinear")

                        # TODO: add 2 new heatmap for sub-ROI loss calculation
                        heatmap_subroi_pred = []
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_1))
                        heatmap_subroi_pred.append(self.heatmap_ds(roi_feats_hr_3))

                        if self.debug_iteration % 1000 == 0:
                            # import pdb
                            # pdb.set_trace()
                            # tmp_file = "xxx.png"
                            # cv2.imwrite(tmp_file, roi_feats_hr_1)

                            input_roi_tmpimg = heatmap_subroi_pred[0][0].permute(1, 2, 0)
                            tmpimg = input_roi_tmpimg.detach().cpu().numpy()
                            tmpimg -= np.min(tmpimg)
                            tmpimg /= np.max(tmpimg)  # Normalize between 0-1
                            tmpimg = np.uint8(tmpimg * 255.0)
                            tmp_file = "co_atten_roi1.png"
                            cv2.imwrite(tmp_file, tmpimg)

                            tmpimg = heatmap_subroi_pred[1][0].permute(1, 2, 0).detach().cpu().numpy()
                            tmpimg -= np.min(tmpimg)
                            tmpimg /= np.max(tmpimg)  # Normalize between 0-1
                            tmpimg = np.uint8(tmpimg * 255.0)
                            tmp_file = "co_atten_roi2.png"
                            cv2.imwrite(tmp_file, tmpimg)

                    # end -- co-attention

                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_3], dim=1)
                else:
                    roi_feats_hr = torch.cat([roi_feats_hr_1, roi_feats_hr_2, roi_feats_hr_3], dim=1)

            elif self.hrnet_type == 3:
                # TODO, it still need to debug
                import pdb
                pdb.set_trace()
                input_ds_feats = self.AG_Net(input_ds)  # (batch, 64, H/2, W/2)
                # 1 channel
                heatmap_ds_pred = input_ds_feats
            else:
                print("Warning: Not support backbone type: {}" .format(self.hrnet_type))

        else:
            # *** 1 ROI in fine stage network ***
            if self.hrnet_type == 0:
                roi_feats_hr = self.hrnet_roi(input_roi)  # (batch, 256, 64, 64)
                # roi_feats_hr = self.subpixel_up_by2(roi_feats_hr)  # 256 -> 64 channel
                # roi_feats_hr = F.interpolate(roi_feats_hr, region_size, mode="bilinear")
                # 256 channel --> 16 channel
                roi_feats_hr = self.subpixel_up_by4(roi_feats_hr)
                # TODO: note the ROI size and feature size is same here, skip the step if if ratio is 1:1
                # roi_feats_hr = crop_and_resize(roi_feats_hr, multi_roi_center, region_size, scale=1.0)

                # roi_feats_hr = self.hrnet_roi(input_roi)  # (batch, 256, 64, 64)
                # roi_feats_hr = F.interpolate(roi_feats_hr, region_size,
                #                              mode="bilinear")  # (batch, 16/256, 256, 256)

            elif self.hrnet_type == 1:
                roi_feats_hr, _ = self.Unet_1(input_roi)  # (batch, 64, H, W)
                roi_feats_hr = F.interpolate(roi_feats_hr, region_size, mode="bilinear")
                # TODO: note the ROI size and feature size is same here, skip the step if if ratio is 1:1
                # roi_feats_hr = crop_and_resize(roi_feats_hr, multi_roi_center, region_size, scale=1.0)

            elif self.hrnet_type == 2 or self.hrnet_type == 4:
                roi_feats_hr, _ = self.Eff_Unet_1(input_roi)  # (batch, 32, H, W)
                roi_feats_hr = F.interpolate(roi_feats_hr, region_size, mode="bilinear")
            else:
                # TODO
                raise("we don't support it now, will support it later")

            # TODO: test unet result
            # import pdb
            # pdb.set_trace()
            # input_roi_tmpimg = input_roi[0, :, :, :]
            # if len(input_roi_tmpimg.shape) > 2: input_roi_tmpimg = input_roi_tmpimg.permute(1, 2, 0)
            # tmpimg = input_roi_tmpimg.detach().cpu().numpy()
            #
            # tmpimg -= np.min(tmpimg)
            # tmpimg /= np.max(tmpimg)  # Normalize between 0-1
            # tmpimg = np.uint8(tmpimg * 255.0)
            # tmp_file = "tmp.png"
            # cv2.imwrite(tmp_file, tmpimg)
            # # tmp_c = roi_feats_hr.shape[0]
            # # for i in range(tmp_c):
            # # sample_roi_image = roi_feats_hr[0, :, :, :]
            # # sample_roi_image = torch.mean(sample_roi_image, 0)
            # # sample_roi_image = self.subpixel_up_by4(roi_feats_hr)
            # sample_roi_image = roi_feats_hr[0, :, :, :]
            # # sample_roi_image = torch.mean(sample_roi_image, 0)
            # sample_roi_image = sample_roi_image[0, :, :]
            # sample_roi_image = sample_roi_image.detach().cpu().numpy()
            # sample_roi_image -= np.min(sample_roi_image)
            # sample_roi_image /= np.max(sample_roi_image)  # Normalize between 0-1
            # sample_roi_image = np.uint8(sample_roi_image * 255.0)
            # tmp_file = "tmp_unet.png"
            # cv2.imwrite(tmp_file, sample_roi_image)

        # Handle coarse feature map here
        if self.hrnet_type == 0:
            # HRNET
            # use 64 channel for saving memory
            # input_ds_feats = self.subpixel_up_by2(input_ds_feats_orig)  # 256 -> 64 channel
            # input_ds_feats = F.interpolate(input_ds_feats, region_size, mode="bilinear")

            # use 16 channel input_ds_feats dirctly for hrnet
            roi_feats_lr = crop_and_resize(input_ds_feats, roi_center / ds_factor, region_size, scale=ds_feature_scale)

        elif self.hrnet_type == 1 or self.hrnet_type == 2:
            input_ds_feats = input_ds_feats_orig  # 64 or 32 channel
            input_ds_feats = F.interpolate(input_ds_feats, region_size, mode="bilinear")
            roi_feats_lr = crop_and_resize(input_ds_feats, roi_center / ds_factor, region_size, scale=ds_feature_scale)

        elif self.hrnet_type == 3:
            pass
        else:
            print("Warning: Not support backbone type: {} in 2 stage network".format(self.hrnet_type))

        # 16 + 16 channel --> (batch, 32, 256, 256) --> (336, 336)
        # fusion layer

        # 3 ROI roi_feats_hr with ds feats -- 256 x region_size x region_size
        # 1 ROI roi_feats_hr with ds feats -- 64 x region_size x region_size
        roi_feats = torch.cat([roi_feats_lr, roi_feats_hr], dim=1)  # (batch, 512/32, 256, 256)

        # the self-attention module
        if self.isAtten:
            roi_feats = self.attention(roi_feats)
        # end -- self-attention

        # fuse layer
        roi_feats = self.relu(self.bnf(self.convf(roi_feats)))

        # N channel --> 1 channel  (batch, 1, 256, 256)
        heatmap_roi_pred = self.heatmap_roi(roi_feats)

        # if infer_roi:
        #     # Get initial location from heatmap
        #     if self.cfg.TRAIN.MV_IDEA:
        #         loc_pred_init = get_heatmap_center_preds(heatmap_roi_pred.cpu().numpy())[:, 0, :]
        #     else:
        #         loc_pred_init = get_max_preds(heatmap_roi_pred.cpu().numpy())[0][:, 0, :]
        #
        #     loc_pred_init = torch.FloatTensor(loc_pred_init).cuda(non_blocking=True)
        #     meta.update({'pixel_in_roi': loc_pred_init.cpu()})
        # else:
        #     loc_pred_init = meta['pixel_in_roi'].cuda(non_blocking=True)

        # roi_feats: (batch, 32, 256, 256) --> (batch, 32, 1, 1)
        # loc_init_feat = crop_and_resize(roi_feats, loc_pred_init, output_size=1)
        # (batch, 32, 1, 1) --> (batch, 32) / [B, 1792, 1, 1] --> [B, 1792]
        # loc_init_feat = loc_init_feat[:, :, 0, 0]

        # xiaofeng: disable regression network
        # TODO: change it to ROI heatmap list []
        if self.coAtten:
            offset_in_roi_pred = heatmap_subroi_pred
        else:
            offset_in_roi_pred = torch.from_numpy(np.tile(np.array([-1, -1], np.float32), (batch, 1))).cuda()
        # if self.cfg.TRAIN.EFF_NET:
        #     # (batch, 1792) --> (batch, 2)
        #     offset_in_roi_pred = self.eff_regress(loc_init_feat)
        # else:
        #     # (batch, 32) --> (batch, 2)
        #     offset_in_roi_pred = self.regress(loc_init_feat)

        return heatmap_ds_pred, heatmap_roi_pred, offset_in_roi_pred, meta


def get_fovea_net(cfg, is_train, **kwargs):
    model = FoveaNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
