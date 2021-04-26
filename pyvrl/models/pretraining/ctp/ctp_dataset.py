# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class CtPDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool = False):
        """ CTP dataset configurations.
        Args:
            data_source (dict): data source configuration dictionary
            data_dir (str): data root directory
            transform_cfg (list): data augmentation configuration list
            backend (dict): storage backend configuration
            test_mode (bool): placeholder, not available in CtP training.
        """
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source,
                                                     dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend,
                                             dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)

        try:
            self.mask_trans_idx = \
                next(i for i, trans in enumerate(self.img_transform.transforms)
                     if trans.__class__.__name__ == 'PatchMask')
        except Exception:
            raise ValueError("cannot find PatchMask transformation "
                             "in the data augmentation configurations.")
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor, trans_params = \
            self.img_transform.apply_image(img_list,
                                           return_transform_param=True)
        gt_trajs = trans_params[self.mask_trans_idx]['traj_rois']
        gt_trajs = torch.FloatTensor(gt_trajs)

        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()
        gt_weights = torch.ones((gt_trajs.size(0), gt_trajs.size(1))).float()
        data = dict(
            imgs=DC(img_tensor, stack=True, pad_dims=1, cpu_only=False),
            gt_trajs=DC(gt_trajs, stack=True, pad_dims=1, cpu_only=False),
            gt_weights=DC(gt_weights, stack=True, pad_dims=1, cpu_only=False)
        )
        storage_obj.close()

        return data
