from torch.utils.data import Dataset


from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class MoCoDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 test_mode: bool = False):
        """ A dataset class to generate a pair of training examples
        for contrastive learning. Basically, the vanilla MoCo is traine on
        image dataset, like ImageNet-1M. To facilitate its pplication on video
        dataset, we random pick two video clip and discriminate whether
        these two clips are from same video or not.

        Args:
            data_source (dict): data source configuration dictionary
            data_dir (str): data root directory
            transform_cfg (list): data augmentation configuration list
            backend (dict): storage backend configuration
            test_mode (bool): placeholder, not available in MoCo training.
        """
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_source)

    def get_single_clip(self, video_info, storage_obj):
        """ Get single video clip according to the video_info query."""
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor = self.img_transform.apply_image(img_list)
        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()
        return img_tensor

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)
        imgs_q = self.get_single_clip(video_info, storage_obj)
        imgs_k = self.get_single_clip(video_info, storage_obj)
        data = dict(
            imgs_q=imgs_q,
            imgs_k=imgs_k
        )
        storage_obj.close()

        return data
