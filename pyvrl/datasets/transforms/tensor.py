# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from typing import List, Iterable
from .base_transform import BaseTransform
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class GroupToTensor(BaseTransform):

    def __init__(self,
                 switch_rgb_channels: bool = True,
                 div255: bool = True,
                 mean: Iterable[float] = (0.485, 0.456, 0.406),
                 std: Iterable[float] = (0.229, 0.224, 0.225)):
        self.switch_rgb_channels = switch_rgb_channels
        self.div255 = div255
        self.mean = mean
        self.std = std

    def apply_image(self,
                    data: List[np.ndarray],
                    transform_param=None,
                    return_transform_param: bool = False):
        # concat the numpy array, result in a array in shape [N, H, W, 3]
        joint_array = np.concatenate([np.expand_dims(d, axis=0) for d in data], axis=0)
        img_tensor = torch.FloatTensor(joint_array)
        # dimension permutation
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # [N, 3, H, W]
        if self.switch_rgb_channels:
            img_tensor = img_tensor[:, [2, 1, 0], :, :].contiguous()
        if self.div255:
            img_tensor.div_(255)
        if self.mean is not None and self.std is not None:
            for i, (m, s) in enumerate(zip(self.mean, self.std)):
                img_tensor[:, i, :, :].sub_(m).div_(s)
        if return_transform_param:
            return img_tensor, transform_param
        else:
            return img_tensor

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        joint_array = np.concatenate([np.expand_dims(d, axis=0) for d in flows], axis=0)
        flow_tensor = torch.FloatTensor(joint_array)
        flow_tensor = flow_tensor.permute(0, 3, 1, 2)  # [N, 2, H, W]
        return flow_tensor
