# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" This code file defines two typical network structures proposed by [1],
namely R3D and R(2+1)D.
R3D means that all 2D-conv ops are replaced by 3D-conv ops.
R(2+1)D means that the 3D-conv is decoupled into a 1D temporal conv and a
2D-spatial conv. The hidden channels between 1D-Conv and 2D-Conv are designed
to make number of parameters are close to the pure 3D case.

[1] A Closer Look at Spatiotemporal Convolutions for Action Recognition, CVPR18
"""

import numpy as np
import logging

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
from typing import List
from mmcv.cnn import kaiming_init, constant_init

from .base_backbone import BaseBackbone
from ...builder import BACKBONES


def build_3d_conv(block_type,
                  in_channels,
                  out_channels,
                  kernel_size,
                  stride=1,
                  padding=0,
                  dilation=1,
                  groups=1,
                  with_bn=True) -> nn.Module:
    """ build a 3D convolution block. The structure is
        conv -> (optional) bn -> relu
    Args:
        block_type (str): the block type. '3d' means the pure 3D conv
            and '2.5d' means the R(2+1)D conv.
        in_channels (int): input channels
        out_channels (int): output channels
        kernel_size (int|list|tuple): convolution kernel size
        stride (int|list|tuple): convolution stride
        padding (int|list|tuple): padding size
        dilation (int|list|tuple): dilation size
        groups (int): number of the groups
        with_bn (bool): whether apply BatchNorm.
     """
    assert block_type in ('3d', '2.5d'), "Support 3DConv and (2+1)D conv only."

    def check_dim(x):
        return [x]*3 if isinstance(x, int) else x

    kernel_size = check_dim(kernel_size)
    stride = check_dim(stride)
    padding = check_dim(padding)
    dilation = check_dim(dilation)

    _dict = OrderedDict()
    if block_type == '2.5d':
        # building block for R(2+1)D conv.
        mid_channels = 3 * in_channels * out_channels * \
                       kernel_size[1] * kernel_size[2]
        mid_channels /= (in_channels * kernel_size[1] *
                         kernel_size[2] + 3 * out_channels)
        mid_channels = int(mid_channels)

        # build spatial convolution
        _dict['conv_s'] = nn.Conv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=(1, kernel_size[1], kernel_size[2]),
            stride=[1, stride[1], stride[2]],
            padding=[0, padding[1], padding[2]],
            dilation=[1, dilation[1], dilation[2]],
            groups=groups,
            bias=not with_bn)
        if with_bn:
            _dict['bn_s'] = nn.BatchNorm3d(mid_channels, eps=1e-3)
        _dict['relu_s'] = nn.ReLU(inplace=True)
        _dict['conv_t'] = nn.Conv3d(in_channels=mid_channels,
                                    out_channels=out_channels,
                                    kernel_size=(kernel_size[0], 1, 1),
                                    stride=[stride[0], 1, 1],
                                    padding=[padding[0], 0, 0],
                                    dilation=[dilation[0], 1, 1],
                                    groups=groups,
                                    bias=False)
    else:
        # build spatial convolution
        _dict['conv'] = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=not with_bn)
    block = nn.Sequential(_dict)

    return block


class BasicBlock(nn.Module):

    def __init__(self,
                 block_type: str,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_channels: int,
                 down_sampling: bool = False,
                 down_sampling_temporal: bool = None,
                 with_bn: bool = True):
        """ Basic residual block proposed in ResNet. The only difference is
        we replace the 2d-conv with 3d or 2.5d conv.

        Args:
            block_type (str): conv type, one of ['3d', '2.5d']
            in_channels (int): input channels
            out_channels (int): output channels
            bottleneck_channels (int): placeholder, no sense in BasicBlock
            down_sampling (bool): whether apply spatial downsampling
            down_sampling_temporal (bool): whether apply temporal
                down-sampling, if None, it will be same as down_sampling.
            with_bn (bool): whether use batch normalization.

        """
        super(BasicBlock, self).__init__()

        self.with_bn = with_bn

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        spatial_stride = (2, 2) if down_sampling else (1, 1)
        temporal_stride = (2, ) if down_sampling_temporal else (1, )
        stride = temporal_stride + spatial_stride

        self.conv1 = build_3d_conv(block_type=block_type,
                                   in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=[3, 3, 3],
                                   stride=stride,
                                   padding=[1, 1, 1],
                                   with_bn=with_bn)
        if self.with_bn:
            self.bn1 = nn.BatchNorm3d(out_channels, eps=1e-3)
        self.conv2 = build_3d_conv(block_type=block_type,
                                   in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=[3, 3, 3],
                                   stride=[1, 1, 1],
                                   padding=[1, 1, 1],
                                   with_bn=with_bn)
        if self.with_bn:
            self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-3)
        if down_sampling or in_channels != out_channels:
            self.downsample = build_3d_conv(block_type='3d',
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=[1, 1, 1],
                                            stride=stride,
                                            padding=[0, 0, 0],
                                            with_bn=with_bn)
            if self.with_bn:
                self.downsample_bn = nn.BatchNorm3d(out_channels, eps=1e-3)
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.with_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
            if self.with_bn:
                identity = self.downsample_bn(identity)
        out = out + identity
        out = self.relu(out)
        return out


class BaseResNet3D(BaseBackbone):

    BLOCK_CONFIG = {
        10: (1, 1, 1, 1),
        16: (2, 2, 2, 1),
        18: (2, 2, 2, 2),
        26: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    SHALLOW_FILTER_CONFIG = [
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512]
    ]
    DEEP_FILTER_CONFIG = [
        [256, 64],
        [512, 128],
        [1024, 256],
        [2048, 512]
    ]

    def __init__(self,
                 block_type: str,
                 depth: int,
                 num_stages: int,
                 stem: dict,
                 down_sampling: List[bool],
                 channel_multiplier: int,
                 bottleneck_multiplier: int,
                 down_sampling_temporal: List[bool] = None,
                 with_bn: bool = True,
                 bn_eval: bool = False,
                 return_indices: List = None,
                 zero_init_residual: bool = False,
                 bn_momentum: float = None,
                 frozen: bool = False,
                 pretrained: str = None):
        """ A base class for R3D and R(2+1)D.
        Args:
            block_type (str): the conv block type, one element
                in ['3d', '2.5d'], which denotes R3D backbone and R(2+1)D
                backbone respectively.
            depth (int): the backbone depth, support Res18 only.
            num_stages (int): how many residual stages are used
            stem (dict): stem conv configuration.
            down_sampling (list): whether apply spatial down-sampling
                to the specific stage.
            down_sampling_temporal (list): whether apply temporal
                down-sampling to the specific stage.
            channel_multiplier (int): extend the number of res channels.
            bottleneck_multiplier (int): extend the number of res channels.
            with_bn (bool): apply batch normalization
            bn_eval (bool): if true, it will use the cached mean & var rather
                than online calculating BN statistics.
            return_indices (list): which inner features (after each stage)
                will be returned
            zero_init_residual (bool): set the bn weight to 0 when
                initialization. this is trick commonly used in image
                classification task. However, it does not show benefits
                in action recognition task.
            bn_momentum (float): the momentum value in batch norm op.
            frozen (bool|int): if true, all parameters won't be updated in
                the training. we can also partially freeze the backbone.
                for example, frozen == 2 means that stem_conv and res_1
                will be frozen.
            pretrained (str): pretrained model path.
        """
        super(BaseResNet3D, self).__init__()
        self.pretrained = pretrained
        self.return_indices = return_indices
        self.zero_init_residual = zero_init_residual
        self.bn_momentum = bn_momentum
        self.block_type = block_type
        self.depth = depth
        self.num_stages = num_stages
        self.bn_eval = bn_eval

        self.stem = self.build_stem_block(stem_type=block_type, with_bn=with_bn, **stem)

        stage_blocks = self.BLOCK_CONFIG[depth]
        if self.depth <= 18 or self.depth == 34:
            block_constructor = BasicBlock
        else:
            raise NotImplementedError("BottleNeckBlock is not supported yet.")

        if self.depth <= 34:
            filter_config = self.SHALLOW_FILTER_CONFIG
        else:
            filter_config = self.DEEP_FILTER_CONFIG
        filter_config = np.multiply(filter_config,
                                    channel_multiplier).astype(np.int)

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        in_channels = 64
        for i in range(num_stages):
            layer = self.build_res_layer(
                block_constructor, block_type,
                num_blocks=stage_blocks[i],
                in_channels=in_channels,
                out_channels=int(filter_config[i][0]),
                bottleneck_channels=int(filter_config[i][1] *
                                        bottleneck_multiplier),
                down_sampling=down_sampling[i],
                down_sampling_temporal=down_sampling_temporal[i],
                with_bn=with_bn)
            self.add_module('layer{}'.format(i+1), layer)
            in_channels = int(filter_config[i][0])

        self.frozen = frozen
        if isinstance(self.frozen, bool):
            if self.frozen:
                for p in self.parameters():
                    p.requires_grad_(False)
        elif isinstance(self.frozen, int):
            if self.frozen >= 1:
                for p in self.stem.parameters():
                    p.requires_grad_(False)
                for i in range(1, self.frozen):
                    for p in getattr(self, 'layer{}'.format(i)).parameters():
                        p.requires_grad_(False)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.stem(x)
        feats = [x]
        for i in range(self.num_stages):
            feats.append(getattr(self, 'layer{}'.format(i+1))(feats[-1]))
        if self.return_indices is None:
            return feats[-1]
        else:
            return [feats[k] for k in self.return_indices]

    def init_weights(self):
        logger = logging.getLogger()
        if isinstance(self.pretrained, str):
            self.init_from_pretrained(self.pretrained, logger)
        elif self.pretrained is None:
            logger.info("Random init backbone network")
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        constant_init(m.bn2, 0)
        else:
            raise TypeError('pretrained must be a str or None')
        # https://github.com/facebookresearch/VMZ/blob/master/pt/vmz/models/r2plus1d.py
        if self.bn_momentum is not None:
            logger.info("Set BN momentum to {}".format(self.bn_momentum))
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.momentum = self.bn_momentum

    @staticmethod
    def build_res_layer(block,
                        block_type: str,
                        num_blocks: int,
                        in_channels: int,
                        out_channels: int,
                        bottleneck_channels: int,
                        down_sampling: bool = False,
                        down_sampling_temporal: bool = None,
                        with_bn: bool = True):
        layers = list()
        layers.append(block(
            block_type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            down_sampling=down_sampling,
            down_sampling_temporal=down_sampling_temporal,
            with_bn=with_bn
        ))
        for i in range(num_blocks-1):
            layers.append(block(
                block_type=block_type,
                in_channels=out_channels,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels,
                down_sampling=False,
                down_sampling_temporal=None,
                with_bn=with_bn
            ))
        return nn.Sequential(*layers)

    @staticmethod
    def build_stem_block(stem_type: str,
                         temporal_kernel_size: int,
                         temporal_stride: int,
                         in_channels: int = 3,
                         with_bn: bool = True,
                         with_pool: bool = True) -> nn.Sequential:
        _dict = OrderedDict()
        if stem_type == '2.5d':
            _dict['conv_s'] = nn.Conv3d(
                in_channels=in_channels,
                out_channels=45,
                kernel_size=(1, 7, 7),
                stride=[1, 2, 2],
                padding=[0, 3, 3],
                bias=not with_bn)
            if with_bn:
                _dict['bn_s'] = nn.BatchNorm3d(45, eps=1e-3)
            _dict['relu_s'] = nn.ReLU(inplace=True)
            _dict['conv_t'] = nn.Conv3d(
                in_channels=45,
                out_channels=64,
                kernel_size=(temporal_kernel_size, 1, 1),
                stride=[temporal_stride, 1, 1],
                padding=[(temporal_kernel_size-1)//2, 0, 0],
                bias=not with_bn)
            if with_bn:
                _dict['bn_t'] = nn.BatchNorm3d(64, eps=1e-3)
            _dict['relu_t'] = nn.ReLU(inplace=True)
        elif stem_type == '3d':
            _dict['conv'] = nn.Conv3d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(temporal_kernel_size, 7, 7),
                stride=[temporal_stride, 2, 2],
                padding=[(temporal_kernel_size-1)//2, 3, 3],
                bias=not with_bn)
            if with_bn:
                _dict['bn'] = nn.BatchNorm3d(64, eps=1e-3)
            _dict['relu'] = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

        if with_pool:
            _dict['pool'] = nn.MaxPool3d(
                kernel_size=[1, 3, 3],
                stride=[1, 2, 2],
                padding=[0, 1, 1])

        return nn.Sequential(_dict)

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class R2Plus1D(BaseResNet3D):
    def __init__(self, *args, **kwargs):
        super(R2Plus1D, self).__init__(block_type='2.5d', *args, **kwargs)


@BACKBONES.register_module()
class R3D(BaseResNet3D):
    def __init__(self, *args, **kwargs):
        super(R3D, self).__init__(block_type='3d', *args, **kwargs)
