# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random
from typing import List, Union


class RandomFrameSampler(object):

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

    def sample_single(self, num_frames: int):
        stride = random.choice(self.strides)
        if stride > 1 and self.temporal_jitter:
            index_jitter = [random.randint(0, stride-1) for i in range(self.clip_len)]
        else:
            index_jitter = [0 for i in range(self.clip_len)]

        total_len = stride * (self.clip_len - 1) + 1
        if total_len >= num_frames:
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - total_len)
        frame_inds = [min(start_index + i * stride + index_jitter[i], num_frames-1)
                      for i in range(self.clip_len)]
        return frame_inds

    def sample(self, num_frames: int):
        return np.array([self.sample_single(num_frames) for _ in range(self.num_clips)], np.int)
