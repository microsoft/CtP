# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import cv2
from typing import List
from .base_transform import BaseTransform
from .dynamic_utils import sample_key_frames, extend_key_frame_to_all
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class RandomBrightness(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float):
        self.brightness_prob = prob
        self.brightness_delta = delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.brightness_prob)
        delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
        return dict(flag=flag, delta=delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            delta = transform_param['delta']
            data = [np.clip(img + delta, a_min=0, a_max=255) for img in data]
        return data


@TRANSFORMS.register_module()
class RandomContrast(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float):
        self.contrast_prob = prob
        self.contrast_delta = delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.contrast_prob)
        delta = np.exp(np.random.uniform(-self.contrast_delta, self.contrast_delta))
        return dict(flag=flag, delta=delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        delta = transform_param['delta']
        if transform_param['flag']:
            data = [np.clip(img * delta, 0, 255) for img in data]
        return data


@TRANSFORMS.register_module()
class RandomHueSaturation(BaseTransform):

    def __init__(self,
                 prob: float,
                 hue_delta: float,
                 saturation_delta: float):
        self.prob = prob
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta

    def get_transform_param(self, *args, **kwargs) -> dict:
        flag = (0 < np.random.rand() < self.prob)
        hue_delta = np.random.uniform(-self.hue_delta, self.hue_delta)
        saturation_delta = np.exp(np.random.uniform(-self.saturation_delta, self.saturation_delta))
        return dict(flag=flag,
                    hue_delta=hue_delta,
                    saturation_delta=saturation_delta)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            hue_delta = transform_param['hue_delta']
            saturation_delta = transform_param['saturation_delta']
            # convert to HSV color space
            data = [cv2.cvtColor(self.cvt_uint8(img), cv2.COLOR_BGR2HSV).astype(np.float32)
                    for img in data]
            for i in range(len(data)):
                data[i][:, :, 0] += hue_delta
                data[i][:, :, 1] *= saturation_delta
            data = [cv2.cvtColor(self.cvt_uint8(img, is_bgr=False), cv2.COLOR_HSV2BGR).astype(np.float32)
                    for img in data]
        return data

    @staticmethod
    def cvt_uint8(img, is_bgr=True):
        """ convert data type from numpy.float32 to numpy.uint8 """
        nimg = np.round(np.clip(img, 0, 255)).astype(np.uint8)
        if not is_bgr:
            nimg[:, :, 0] = np.clip(nimg[:, :, 0], 0, 179)
        return nimg


@TRANSFORMS.register_module()
class DynamicContrast(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float,
                 num_key_frame_probs: List[float]):
        self.prob = prob
        self.delta = delta
        self.num_key_frame_probs = num_key_frame_probs

    def get_transform_param(self, data):
        flag = np.random.rand() < self.prob
        if not flag:
            return dict(flag=flag)
        key_frame_inds = sample_key_frames(len(data), self.num_key_frame_probs)
        num_key_frames = len(key_frame_inds)
        key_frame_deltas = np.exp(np.random.uniform(-self.delta, self.delta, size=(num_key_frames, )))
        deltas = extend_key_frame_to_all(key_frame_deltas, key_frame_inds, 'random')
        return dict(flag=flag, deltas=deltas)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            deltas = transform_param['deltas']
            data = [np.clip(img * deltas[i], a_min=0, a_max=255) for i, img in enumerate(data)]
        return data


@TRANSFORMS.register_module()
class DynamicBrightness(BaseTransform):

    def __init__(self,
                 prob: float,
                 delta: float,
                 num_key_frame_probs: List[float]):
        self.prob = prob
        self.delta = delta
        self.num_key_frame_probs = num_key_frame_probs

    def get_transform_param(self, data):
        flag = np.random.rand() < self.prob
        if not flag:
            return dict(flag=flag)
        key_frame_inds = sample_key_frames(len(data), self.num_key_frame_probs)
        num_key_frames = len(key_frame_inds)
        key_frame_deltas = np.random.uniform(-self.delta, self.delta, size=(num_key_frames, ))
        deltas = extend_key_frame_to_all(key_frame_deltas, key_frame_inds, 'random')
        return dict(flag=flag, deltas=deltas)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            deltas = transform_param['deltas']
            data = [np.clip(img + deltas[i], a_min=0, a_max=255) for i, img in enumerate(data)]
        return data
