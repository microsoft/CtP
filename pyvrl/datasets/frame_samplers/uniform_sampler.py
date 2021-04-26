# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random
from typing import List, Union


class UniformFrameSampler(object):

    def __init__(self,
                 num_clips: int,
                 clip_len: int,
                 strides: Union[int, List[int]],
                 temporal_jitter: bool):
        self.num_clips = num_clips
        self.clip_len = clip_len
        if isinstance(strides, (tuple, list)):
            self.strides = strides
        else:
            self.strides = [strides]
        self.temporal_jitter = temporal_jitter

    def sample(self, num_frames: int):
        stride = random.choice(self.strides)
        base_length = self.clip_len * stride
        delta_length = num_frames - base_length + 1

        if delta_length > 0:
            tick = float(delta_length) / self.num_clips
            offsets = np.array([int(tick / 2.0 + tick * x)
                                for x in range(self.num_clips)], np.int)
        else:
            offsets = np.zeros((self.num_clips, ), np.int)

        inds = np.arange(0, base_length, stride, dtype=np.int).reshape(1, self.clip_len)
        if self.num_clips > 1:
            inds = np.tile(inds, (self.num_clips, 1))
        # apply for the init offset
        inds = inds + offsets.reshape(self.num_clips, 1)
        if self.temporal_jitter and stride > 1:
            skip_offsets = np.random.randint(stride, size=self.clip_len)
            inds = inds + skip_offsets.reshape(1, self.clip_len)
        inds = np.clip(inds, a_min=0, a_max=num_frames-1)
        inds = inds.astype(np.int)
        return inds
