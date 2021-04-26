import numpy as np
from torch.utils.data import Dataset

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class MemDPCDataset(Dataset):

    def __init__(self,
                 data_dir,
                 data_source,
                 backend,
                 transform_cfg,
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 test_mode=False):
        self.test_mode = test_mode
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample

        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.img_transform = Compose(transform_cfg)

        self.available_inds = []
        for idx in range(len(self.data_source)):
            video_info = self.data_source[idx]
            storage_obj = self.backend.open(video_info)
            vlen = len(storage_obj)
            if vlen - self.num_seq * self.seq_len * self.downsample > 0:
                self.available_inds.append(idx)
            storage_obj.close()

    def idx_sampler(self, vlen):
        """ sample index from a video"""
        assert vlen >= self.num_seq * self.seq_len * self.downsample
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), 1)
        seq_idx = np.arange(self.num_seq * self.seq_len) * self.downsample + start_idx
        seq_idx = seq_idx.astype(np.long)
        return seq_idx

    def __getitem__(self, index):
        mapped_idx = self.available_inds[index]

        video_info = self.data_source[mapped_idx]
        storage_obj = self.backend.open(video_info)
        frame_index = self.idx_sampler(len(storage_obj))

        seq = storage_obj.get_frame(frame_index.reshape(-1))
        t_seq = self.img_transform.apply_image(seq)
        B, C, H, W = t_seq.size()
        # (C, H, W) = t_seq[0].size()
        # t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
        storage_obj.close()
        return dict(imgs=t_seq)

    def __len__(self):
        return len(self.available_inds)
