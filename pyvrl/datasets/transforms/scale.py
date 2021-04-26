# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import cv2
import numpy as np
from typing import Union, Tuple, List

from .base_transform import BaseTransform
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class GroupScale(BaseTransform):

    def __init__(self, scales: Union[List[Tuple[int, int]], Tuple[int, int]]):
        if not isinstance(scales, list):
            scales = [scales]
        self.num_scales = len(scales)
        self.scales = scales

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        scale_idx = int(np.random.choice(self.num_scales))
        scale = self.scales[scale_idx]
        h, w = data[0].shape[0:2]
        if isinstance(scale, tuple):
            assert len(scale) == 2
            assert scale[0] >= scale[1]
            if h > w:
                out_size = (scale[1], scale[0])
            else:
                out_size = scale

        elif isinstance(scale, int):
            short_edge = min(h, w)
            assert short_edge > 0
            scale_factor = float(scale) / short_edge
            out_size = (int(round(w * scale_factor)), int(round(h * scale_factor)))
        else:
            raise NotImplementedError
        return dict(img_shape=data[0].shape, out_size=out_size)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        out_size = transform_param['out_size']
        data = [cv2.resize(img, dsize=out_size) for img in data]

        return data

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        out_size = transform_param['out_size']
        img_h, img_w = transform_param['img_shape'][0:2]
        fx = out_size[0] / float(img_w)
        fy = out_size[1] / float(img_h)

        transformed_boxes = []
        for i_boxes in boxes:
            i_trans_boxes = i_boxes.copy()
            i_trans_boxes[:, 0:4:2] *= fx
            i_trans_boxes[:, 1:4:2] *= fy
            transformed_boxes.append(i_trans_boxes)
        return transformed_boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        out_size = transform_param['out_size']
        img_h, img_w = transform_param['img_shape'][0:2]
        fx = out_size[0] / float(img_w)
        fy = out_size[1] / float(img_h)

        transformed_flows = []
        for i_flow in flows:
            i_trans_flow = i_flow.copy()
            i_trans_flow[..., 0] *= fx
            i_trans_flow[..., 1] *= fy
            i_trans_flow = cv2.resize(i_trans_flow, dsize=out_size)
            transformed_flows.append(i_trans_flow)
        return transformed_flows
