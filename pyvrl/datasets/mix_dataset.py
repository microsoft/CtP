# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from torch.utils.data import Dataset

from ..builder import DATASETS, build_dataset


@DATASETS.register_module()
class MixDataset(Dataset):
    """ Mix muptiple datasets. All the datasets should have the same length
    (__len__). In each iteration, we will randomly sample from one dataset.
    """
    def __init__(self,
                 datasets,
                 probs=None,
                 **kwargs):
        if probs is None:
            probs = np.ones((len(datasets), ), np.float32)
        self.probs = np.array(probs, np.float32)
        self.probs = self.probs / self.probs.sum()

        self.datasets = [build_dataset(d, default_args=kwargs)
                         for d in datasets]
        assert all([len(d) == len(self.datasets[0]) for d in self.datasets])

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        select_idx = np.random.choice(len(self.datasets), p=self.probs)
        return self.datasets[select_idx][idx]
