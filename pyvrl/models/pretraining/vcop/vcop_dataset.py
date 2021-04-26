import numpy as np
import torch
import random
import itertools
from torch.utils.data import Dataset

from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class VCOPDataset(Dataset):
    """ pretext task: video clip order predction [1].

    Official github: https://github.com/xudejing/video-clip-order-prediction

    [1] Self-supervised Spatiotemporal Learning via Video Clip Order
        Prediction, CVPR'19

    """

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 transform_cfg: list,
                 clip_len: int,
                 min_interval: int,
                 max_interval: int,
                 tuple_len: int,
                 test_mode: bool):
        """ VCOP dataset configuration.
        Args:
            data_source (dict): data source configuration dictionary.
            data_dir (str): data root directory.
            transform_cfg (list): data augmentation configuration list.
            clip_len (int): the length of sampled clips
            min_interval (int): the min interval of neighbour clips.
            max_interval (int): the max interval of neighbour clips.
            tuple_len (int): how many clips are sampled in a batch
            test_mode (bool): placeholder, not available in VCOP training.
        """
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.img_transform = Compose(transform_cfg)

        self.clip_len = clip_len
        self.min_interval = min_interval
        self.max_interval = max_interval
        assert self.max_interval >= self.min_interval
        self.tuple_len = tuple_len
        permutations = np.array(list(itertools.permutations(list(range(self.tuple_len)))))
        self.permutations = torch.LongTensor(permutations)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)  # type: storage_backends.BaseStorageBackend
        num_frames = len(storage_obj)

        frame_inds = self._sample_indices(num_frames)

        # extract video frames from backend storage
        # get frame according to the frame indexes
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_list = [img.astype(np.float32) for img in img_list]
        img_tensor = self.img_transform.apply_image(img_list)  # type: torch.Tensor
        img_tensor = img_tensor.view((self.tuple_len, self.clip_len) + img_tensor.shape[1:])
        img_tensor = img_tensor.permute((0, 2, 1, 3, 4))  # to [N_tup, 3, T, H, W]

        # random shuffle
        shuffle_index = random.randint(0, self.permutations.size(0)-1)
        shuffle_order = self.permutations[shuffle_index]
        img_tensor = img_tensor[shuffle_order].contiguous()

        gt_label = torch.LongTensor([shuffle_index])
        data = dict(
            imgs=DataContainer(img_tensor, stack=True, pad_dims=2, cpu_only=False),
            gt_labels=DataContainer(gt_label, stack=True, pad_dims=None, cpu_only=False)
        )

        return data

    def __len__(self):
        return len(self.data_source)

    def _sample_indices(self, num_frames: int) -> np.ndarray:
        assert self.tuple_len > 1
        avg_interval = (num_frames - self.tuple_len * self.clip_len) // (self.tuple_len - 1)
        if avg_interval > 0:
            if avg_interval >= self.max_interval:
                interval = np.random.randint(self.min_interval, self.max_interval+1)
            elif avg_interval >= self.min_interval:
                interval = np.random.randint(self.min_interval, avg_interval + 1)
            else:
                interval = avg_interval
            tuple_start_max = num_frames - self.tuple_len * self.clip_len - (self.tuple_len - 1) * interval
            tuple_start = random.randint(0, tuple_start_max)
            offsets = np.array([tuple_start + (self.clip_len + interval) * i
                                for i in range(self.tuple_len)], dtype=np.int)
        else:
            # for special caes, we adopt the same sample setting in TSNDataset.
            # note that this won't happen in Kinetics training.
            delta_length = num_frames - self.clip_len + 1
            if delta_length > 0:
                tick = float(delta_length) / self.tuple_len
                offsets = np.array([int(tick / 2.0 + tick * x)
                                    for x in range(self.tuple_len)], np.int)
            else:
                offsets = np.zeros((self.tuple_len, ), np.int)
        inds = np.arange(0, self.clip_len, dtype=np.int).reshape(1, self.clip_len)
        inds = np.tile(inds, (self.tuple_len, 1))
        inds = inds + offsets.reshape(self.tuple_len, 1)
        inds = np.clip(inds, a_min=0, a_max=num_frames - 1)
        inds = inds.astype(np.int)
        return inds
