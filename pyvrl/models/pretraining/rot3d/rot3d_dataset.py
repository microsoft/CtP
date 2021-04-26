import torch
import logging
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ...classifiers.tsn import TSNDataset


@DATASETS.register_module()
class RotateDataset(TSNDataset):
    """ Construct training data for 3DRotNet.

    The data is almost same as TSNDataset, except for the ground-truth label.
    We use rotation flags as the ground-truth for classification:
    0: no rotate
    1: 90 rotate
    2: 180 rotate
    3: 270 rotate

    """
    def __init__(self, *args, **kwargs):
        super(RotateDataset, self).__init__(*args, **kwargs)

        # find the index of rotation transformation in the list
        # of image transformations, because we need to get the rotation
        # degree as the ground-truth.
        try:
            self.rot_trans_index = \
                next(i for i, trans in enumerate(self.img_transform.transforms)
                     if trans.__class__.__name__ == 'GroupRectRotate')
        except Exception:
            logger = logging.getLogger()
            logger.error("Cannot find 'GroupRectRotate' in "
                         "the image transformation configuration."
                         "It is necessary for Rot3D task.")
            raise ValueError

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        storage_obj = self.backend.open(video_info)
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1, f'support num_segs==1 only, got {num_segs}'

        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        img_tensor, trans_params = \
            self.img_transform.apply_image(img_list,
                                           return_transform_param=True)
        img_tensor = img_tensor.view((num_segs, clip_len) +
                                     img_tensor.shape[1:])
        img_tensor = img_tensor.permute(0, 2, 1, 3, 4).contiguous()

        data = dict(
            imgs=DataContainer(img_tensor,
                               stack=True,
                               pad_dims=2,
                               cpu_only=False),
        )

        if not self.test_mode:
            gt_label = \
                torch.LongTensor([trans_params[self.rot_trans_index]['flag']])
            data['gt_labels'] = DataContainer(gt_label,
                                              stack=True,
                                              pad_dims=None,
                                              cpu_only=False)

        return data

