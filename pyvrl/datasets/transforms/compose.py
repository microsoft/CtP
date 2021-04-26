# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from typing import List
from mmcv import build_from_cfg

from .base_transform import BaseTransform
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class Compose(BaseTransform):

    def __init__(self, transform_cfgs: List[dict]):
        self.transforms = []  # type: List[BaseTransform]
        for transform_cfg in transform_cfgs:
            if isinstance(transform_cfg, BaseTransform):
                self.transforms.append(transform_cfg)
            else:
                self.transforms.append(build_from_cfg(transform_cfg, TRANSFORMS))

    def get_transform_param(self, *args, **kwargs):
        raise NotImplementedError

    def index(self, trans_type: str):
        index_id = -1
        for i, t in enumerate(self.transforms):
            t_name = str(t.__class__.__name__)
            if t_name.lower() == trans_type.lower():
                index_id = i
        return index_id

    def apply_image(self,
                    data: List[np.ndarray],
                    transform_param: List = None,
                    return_transform_param: bool = False):
        has_transform_param = transform_param is not None
        if not has_transform_param:
            transform_param = []
        for i, trans in enumerate(self.transforms):
            if has_transform_param:
                data = trans.apply_image(data, transform_param[i])
            else:
                if return_transform_param:
                    data, p = trans.apply_image(data, return_transform_param=True)
                    transform_param.append(p)
                else:
                    data = trans.apply_image(data)

        if not return_transform_param:
            return data
        else:
            return data, transform_param

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param):
        for i, trans in enumerate(self.transforms):
            boxes = trans.apply_boxes(boxes, transform_param[i])
        return boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param):
        for i, trans in enumerate(self.transforms):
            flows = trans.apply_flow(flows, transform_param[i])
        return flows
