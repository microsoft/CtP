# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import random
import cv2
from typing import List

from ....datasets import BaseTransform
from ....datasets.transforms.dynamic_utils import (extend_key_frame_to_all,
                                                   sample_key_frames)
from ....builder import TRANSFORMS


class RandomRegionSampler(object):

    def __init__(self,
                 num_rois: int,
                 scales: tuple,
                 ratios: tuple,
                 scale_jitter: float):
        """ Randomly sample several RoIs

        Args:
            num_rois (int): number of sampled RoIs per image
            scales (tuple): scales of candidate bounding boxes
            ratios (tuple): aspect ratios of candidate bounding boxes
            scale_jitter (float): scale jitter factor, positive number
        """

        self.num_rois = num_rois
        self.scale_jitter = scale_jitter

        scales = np.array(scales, np.float32)
        ratios = np.array(ratios, np.float32)
        widths = scales.reshape(1, -1) * np.sqrt(ratios).reshape(-1, 1)
        heights = scales.reshape(1, -1) / np.sqrt(ratios).reshape(-1, 1)
        self.anchors = np.concatenate((widths.reshape(-1, 1),
                                       heights.reshape(-1, 1)), axis=-1)

    def sample(self, data: List[np.ndarray]) -> np.ndarray:
        """ Sample boxes.

        Args:
            data (list): image list, each element is a numpy.ndarray
                in shape of [H, W, 3]

        Returns:
            boxes (np.ndarray): the sampled bounding boxes. in shape of
                [self.num_rois, 4], represented in (x1, y1, x2, y2).

        """
        h, w = data[0].shape[0:2]

        # random sample box shapes
        anchor_inds = np.random.randint(0, len(self.anchors),
                                        size=(self.num_rois, ))
        box_shapes = self.anchors[anchor_inds].copy()
        if self.scale_jitter is not None:
            scale_factors = np.random.uniform(-self.scale_jitter,
                                              self.scale_jitter,
                                              size=(self.num_rois, 2))
            box_shapes = box_shapes * np.exp(scale_factors)
        box_shapes[:, 0] = np.clip(box_shapes[:, 0], 1, w - 1)
        box_shapes[:, 1] = np.clip(box_shapes[:, 1], 1, h - 1)

        # random sample box x1, y1
        x1 = np.random.uniform(0, w - box_shapes[:, 0])
        y1 = np.random.uniform(0, h - box_shapes[:, 1])
        boxes = np.concatenate((x1.reshape(-1, 1),
                                y1.reshape(-1, 1),
                                (x1 + box_shapes[:, 0]).reshape(-1, 1),
                                (y1 + box_shapes[:, 1]).reshape(-1, 1)),
                               axis=1)

        return boxes


@TRANSFORMS.register_module()
class PatchMask(BaseTransform):

    def __init__(self,
                 region_sampler: dict,
                 key_frame_probs: list,
                 loc_velocity: float,
                 size_velocity: float,
                 label_prob: float):
        """ Core transformation in Catch-the-Patch.

        Args:
            region_sampler (dict): region sampler setting, it will be used to
                construct a RandomRegionSampler object.
            key_frame_probs (list): probabilities of sampling how many key
                frames. The sum of this list should be 1.
            loc_velocity (float): the maximum patch movement speed. (pix per
                frame).
            size_velocity (float): the maximum size change ratios between two
                neighbouring frames.
            label_prob (float): how many percentages of frames will be
                modified. Note that even the frame is not modified, we still
                force the model to infer the patch positions. (see MRM module
                in the paper).
        """
        self.region_sampler = RandomRegionSampler(**region_sampler)
        self.key_frame_probs = key_frame_probs
        self.loc_velocity = loc_velocity
        self.size_velocity = size_velocity
        self.label_prob = label_prob

    def get_transform_param(self, data: List[np.ndarray], *args, **kwargs):
        """ Generate the transformation parameters.

        Args:
            data (List[np.ndarray]): list of image array, each element is in
                a shape of [H, W, 3]

        Returns:
            params (dict): a dict that contains necessary transformation
                params, which include:
                'patches': list of image patches (np.ndarray)
                'alphas': list of alpha mask, same size and shape as patches.
                'traj_rois': the trajectory position, in shape of
                    [N_traj, N_frame, 4]
                'traj_labels': whether the patches have been pasted on some
                    specific frames, in shape of [N_traj, N_frame]
        """
        # Step 1, generate the trajectories.
        h, w = data[0].shape[0:2]
        boxes = self.region_sampler.sample(data)
        traj_rois = self.rectangle_movement(boxes, (w, h),
                                            self.loc_velocity,
                                            self.size_velocity,
                                            len(data),
                                            self.key_frame_probs)
        traj_rois = np.round(traj_rois).astype(int)
        traj_rois[..., 0::2] = np.clip(traj_rois[..., 0::2], 0, w)
        traj_rois[..., 1::2] = np.clip(traj_rois[..., 1::2], 0, h)

        # Step 2, crop the patches and prepare the alpha masks.
        patches_list = []
        alphas_list = []
        label_list = []
        for i in range(len(boxes)):
            patches, crop_index = self.pick_patches(data, traj_rois[i])
            alphas = self.pick_alphas(data, traj_rois[i], crop_index)
            labels = np.random.uniform(0, 1, size=(len(data), ))
            labels[crop_index] = 0.0
            labels[0] = 0.0
            labels = labels <= self.label_prob
            patches_list.append(patches)
            alphas_list.append(alphas)
            label_list.append(labels)

        return dict(
            patches=patches_list,
            alphas=alphas_list,
            traj_rois=traj_rois,
            traj_labels=label_list
        )

    def pick_patches(self,
                     data: List[np.ndarray],
                     traj_rois: np.ndarray) -> tuple:
        """ Pick image patches from the raw video frame.

        We just randomly select a frame index, and crop the frame according to
        the trajectory rois. This cropped patch will be resized into the
        suitable size specified by the traj_rois.

        Args:
            data (List[np.ndarray]): list of images, each element is in shape
                of [H, W, 3]
            traj_rois (np.ndarray): the generated trajectories, in shape of
                [N_frames, 4]. (x1, y1, x2, y2)

        Returns:
            patches (List[np.ndarray]): the cropped patches
            select_idx (int): the frame index which the source patch
                cropped from.
        """
        traj_sizes = traj_rois[..., 2:4] - traj_rois[..., 0:2]
        num = len(traj_sizes)
        select_idx = random.randint(0, num - 1)
        x1, y1, x2, y2 = traj_rois[select_idx]
        img = data[select_idx][y1:y2, x1:x2, :]
        patches = [cv2.resize(img, (traj_sizes[i, 0], traj_sizes[i, 1]))
                   for i in range(traj_rois.shape[0])]
        return patches, select_idx

    def pick_alphas(self,
                    data,
                    traj_rois: np.ndarray,
                    crop_index: int):
        """ Generate the alpha masks for merging the patches into the raw
        frames:
            out_frame = raw_frame * (1 - alpha) + patch * alpha.
        Despite the transparency, the alpha values are also used to mask the
        patches into some predefined shapes, like ellipse or rhombus.
        There are many strange constants in this function. But we do not
        conduct any ablation analysis on these constants. They should have
        little impact to the final performances.

         Args:
            data (List[np.ndarray]): list of images, each element is in shape
                of [H, W, 3]
            traj_rois (np.ndarray): the generated trajectories, in shape of
                [N_frames, 4]. (x1, y1, x2, y2)
            crop_index (int): the frame index which the source patch
                cropped from.

        Returns:
            alphas (List[np.ndarray]): the generated alpha values

        """
        traj_sizes = traj_rois[..., 2:4] - traj_rois[..., 0:2]
        num_frames = traj_sizes.shape[0]

        base_w, base_h = traj_sizes[crop_index]

        base_x_grids, base_y_grids = np.meshgrid(
            np.arange(base_w).astype(np.float32),
            np.arange(base_h).astype(np.float32)

        )
        ctr_w = (base_w - 1) // 2
        ctr_h = (base_h - 1) // 2

        dist_to_ctr_x = np.abs(base_x_grids - ctr_w) / base_w
        dist_to_ctr_y = np.abs(base_y_grids - ctr_h) / base_h

        mask_type = int(np.random.choice(3, p=[0.5, 0.35, 0.15]))
        if mask_type == 0:
            dist_to_ctr = np.maximum(dist_to_ctr_x, dist_to_ctr_y)
            base_alpha = np.ones((base_h, base_w), np.float32)
        elif mask_type == 1:
            dist_to_ctr = np.sqrt(dist_to_ctr_x ** 2 + dist_to_ctr_y ** 2)
            base_alpha = np.where(dist_to_ctr < 0.5,
                                  np.ones((base_h, base_w), np.float32),
                                  np.zeros((base_h, base_w), np.float32))
        elif mask_type == 2:
            dist_to_ctr = (dist_to_ctr_x + dist_to_ctr_y)
            base_alpha = np.where(dist_to_ctr < 0.5,
                                  np.ones((base_h, base_w), np.float32),
                                  np.zeros((base_h, base_w), np.float32))
        else:
            raise NotImplementedError

        use_smooth_edge = random.uniform(0, 1) < 0.5
        if use_smooth_edge:
            turning_point = random.uniform(0.30, 0.45)
            k = -1 / (0.5 - turning_point)
            alpha_mul = k * dist_to_ctr - 0.5 * k
            alpha_mul = np.clip(alpha_mul, 0, 1)
            base_alpha = base_alpha * alpha_mul

        # sample key frames
        key_inds = sample_key_frames(num_frames, self.key_frame_probs)
        frame_alphas = np.random.uniform(0.8, 1.0, size=(len(key_inds), 1))
        frame_alphas = extend_key_frame_to_all(frame_alphas, key_inds)

        alphas = []
        for frame_idx in range(num_frames):
            w, h = traj_sizes[frame_idx]
            i_alpha = cv2.resize(base_alpha, (w, h))
            i_alpha = i_alpha * frame_alphas[frame_idx]
            alphas.append(i_alpha)
        return alphas

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):

        transformed_data = []

        # we sort the size and firstly paste the large patch
        # this trick is because, if we paste the small patch first, it may
        # be totally covered by a large one.
        sizes = transform_param['traj_rois'][..., 2:4] - \
                transform_param['traj_rois'][..., 0:2]
        avg_sizes = np.prod(np.mean(sizes, axis=1), axis=1)
        arg_rank = np.argsort(avg_sizes)[::-1]

        for frame_idx in range(len(data)):
            i_rois = transform_param['traj_rois'][:, frame_idx, :]
            img = data[frame_idx].copy()
            for patch_idx in arg_rank:
                if not transform_param['traj_labels'][patch_idx][frame_idx]:
                    continue
                i_patch = transform_param['patches'][patch_idx][frame_idx]
                i_alpha = transform_param['alphas'][patch_idx][frame_idx]
                i_alpha = i_alpha[..., np.newaxis]
                x1, y1, x2, y2 = i_rois[patch_idx]
                img[y1:y2, x1:x2, :] = img[y1:y2, x1:x2, :] * (1 - i_alpha) + \
                                       i_patch * i_alpha
            transformed_data.append(img)
        return transformed_data

    @staticmethod
    def rectangle_movement(boxes: np.ndarray,
                           img_wh: tuple,
                           loc_velocity: float,
                           size_velocity: float,
                           num_frames: int,
                           key_frame_probs: List[float]) -> np.ndarray:
        """ Simulate the object movement.

        Args:
            boxes (np.ndarray): in shpae of [N_boxes, 4]
            img_wh (tuple): image width and image height
            loc_velocity (float): max speed of the center point movement
            size_velocity (float): max speed of size changes
            num_frames (int): number of frames
            key_frame_probs (float): probability distribution of how many key
                frames will be sampled.

        Returns
            all_boxes (np.ndarray): the generated box trajectory, in shpae
                of [N_traj, N_frame, 4].

        """
        # Step 1, sample key frames for location changes
        loc_key_inds = sample_key_frames(num_frames, key_frame_probs)
        # Step 2, decide box locations in key frames
        ctr_pts = (boxes[:, 0:2] + boxes[:, 2:4]) * 0.5
        box_sizes = (boxes[:, 2:4] - boxes[:, 0:2])

        min_ctr_pts = box_sizes * 0.5
        max_ctr_pts = np.array(img_wh[0:2]).reshape(1, 2) - box_sizes * 0.5

        ctr_pts_list = [np.expand_dims(ctr_pts, axis=0)]
        for i in range(len(loc_key_inds) - 1):
            if loc_velocity > 0:
                index_diff = loc_key_inds[i + 1] - loc_key_inds[i]
                shifts = np.random.uniform(low=-loc_velocity * index_diff,
                                           high=loc_velocity * index_diff,
                                           size=ctr_pts.shape)
                ctr_pts = ctr_pts + shifts
                ctr_pts = np.clip(ctr_pts, min_ctr_pts, max_ctr_pts)
            ctr_pts_list.append(np.expand_dims(ctr_pts, axis=0))
        ctr_pts = np.concatenate(ctr_pts_list, axis=0)
        ctr_pts = extend_key_frame_to_all(ctr_pts, loc_key_inds, 'random')

        # Step 3, sample key frames for shape changes
        size_key_inds = sample_key_frames(num_frames, key_frame_probs)

        # Step 4, setup shape in different key frames
        box_sizes_list = [np.expand_dims(box_sizes, axis=0)]
        for i in range(len(size_key_inds) - 1):
            if size_velocity > 0:
                index_diff = size_key_inds[i + 1] - size_key_inds[i]
                scales = np.random.uniform(low=-size_velocity * index_diff,
                                           high=size_velocity * index_diff,
                                           size=box_sizes.shape)
                scales = np.exp(scales)
                box_sizes = box_sizes * scales
            box_sizes_list.append(np.expand_dims(box_sizes, axis=0))
        box_sizes = np.concatenate(box_sizes_list, axis=0)
        box_sizes = extend_key_frame_to_all(box_sizes, size_key_inds, 'random')

        # Step 5, construct boxes in key frames
        all_boxes = np.concatenate((ctr_pts - box_sizes * 0.5,
                                    ctr_pts + box_sizes * 0.5), axis=2)
        all_boxes[..., 0::2] = np.clip(all_boxes[..., 0::2], 0, img_wh[0])
        all_boxes[..., 1::2] = np.clip(all_boxes[..., 1::2], 0, img_wh[1])
        all_boxes = all_boxes.transpose((1, 0, 2))
        return all_boxes
