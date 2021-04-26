# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from torch.utils.data import Dataset

from ..builder import DATASETS, build_dataset


@DATASETS.register_module()
class ConcatDataset(Dataset):
    """ Concat multiple datasets. """
    def __init__(self,
                 datasets,
                 **kwargs):
        self.datasets = [build_dataset(d, default_args=kwargs)
                         for d in datasets]
        self.data_inds = [(i, j) for i in range(len(self.datasets))
                          for j in range(len(self.datasets[i]))]

    def __len__(self):
        return len(self.data_inds)

    def __getitem__(self, idx):
        dataset_id, sample_id = self.data_inds[idx]
        return self.datasets[dataset_id][sample_id]
