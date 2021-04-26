# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from ....datasets.transforms import Compose
from ....datasets import builder
from ....builder import DATASETS


@DATASETS.register_module()
class TSNDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool,
                 name: str = None):
        if name is None:
            name = 'undefined_dataset'
        self.name = name
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source,
                                                     dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend,
                                             dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)

        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape

        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor_list = []
        for i in range(num_segs):
            raw_imgs = img_list[i*clip_len:(i+1)*clip_len]
            img_tensor = self.img_transform.apply_image(raw_imgs)
            img_tensor_list.append(img_tensor)

        img_tensor = torch.cat(img_tensor_list, dim=0)
        # img_tensor: (M, C, H, W) M = N_seg * L
        img_tensor = img_tensor.view((num_segs, clip_len) +
                                     img_tensor.shape[1:])
        img_tensor = img_tensor.permute(0, 2, 1, 3, 4).contiguous()
        # img_tensor: [N_seg, 3, L, H, W]
        data = dict(
            imgs=DataContainer(img_tensor, stack=True, cpu_only=False)
        )
        if not self.test_mode:
            gt_label = torch.LongTensor([video_info['label']]) - 1
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)

        return data

    def evaluate(self, results, logger=None):
        if isinstance(results, list):
            if results[0].ndim == 1:
                results = [r[np.newaxis, ...] for r in results]
            results = np.concatenate(results, axis=0)
        assert len(results) == len(self), \
            f'The results should have same size as gts. But' \
            f' got {len(results)} and {len(self)}'
        labels = np.array([int(self.data_source[_]['label']) - 1
                           for _ in range(len(self))], np.long)
        sort_inds = results.argsort(axis=1)[:, ::-1]

        acc_dict = dict()
        for k in [1, 5]:
            top_k_inds = sort_inds[:, :k]
            correct = (top_k_inds.astype(np.long) ==
                       labels.reshape(len(self), 1))
            correct_count = np.any(correct, axis=1).astype(np.float32).sum()
            acc = correct_count / len(self)
            acc_dict[f'top_{k}_acc'] = acc
            if logger is not None:
                logger.info(f'top_{k}_acc: {acc*100}%')

        # mean class accuracy
        per_class_acc = dict()
        for i in range(len(results)):
            class_id = int(labels[i])
            if class_id not in per_class_acc:
                per_class_acc[class_id] = []
            if sort_inds[i, 0] == class_id:
                per_class_acc[class_id].append(1.0)
            else:
                per_class_acc[class_id].append(0.0)
        per_class_acc_list = []
        for k, v in per_class_acc.items():
            per_class_acc_list.append(sum(v) / len(v))
        acc_dict[f'mean_class_acc'] = sum(per_class_acc_list) / \
                                      len(per_class_acc_list)

        return acc_dict
