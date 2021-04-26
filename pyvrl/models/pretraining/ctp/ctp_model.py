# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn
from typing import Tuple
from mmcv.ops.bbox import bbox_overlaps

from .ctp_head import CtPHead
from ...train_step_mixin import TrainStepMixin
from ....builder import build_backbone, MODELS


@MODELS.register_module()
class CtP(nn.Module, TrainStepMixin):

    def __init__(self,
                 backbone: dict,
                 head: dict):
        """ The main component of Catch-the-Patch pretraining task. """
        super(CtP, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = CtPHead(**head)
        self.init_weigths()

    def init_weigths(self):
        self.backbone.init_weights()
        self.head.init_weights()

    def _forward(self,
                 imgs: torch.Tensor,
                 gt_trajs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward the model and get the predicted results.

        Args:
            imgs (torch.Tensor): RGB img tensor in shape of [N, C, T, H, W]
            gt_trajs (torch.Tensor): ground-truth trajectories, in shape of
                [N, N_traj, T, 4]. The boxes are represented in format of
                (x1, y1, x2, y2).

        Returns:
            preds (torch.Tensor): prediction results,
                in shape of [N, N_traj, T, 4]
            rois (torch.Tensor): the query rois, in shape of [N * N_traj, 5]
        """
        # Step 1, extract spatial-temporal features
        feats = self.backbone(imgs)

        # Step 2, pick the bounding box on the starting frame as query
        # note tha the RoIAlign operation needs to specify image index.
        # therefore, we first generate roi_inds.
        bs, nt, nf, _ = gt_trajs.size()  # [B, N_trajectory, N_frames, 4]
        roi_inds = torch.arange(bs).view(bs, 1).repeat(1, nt)
        roi_inds = roi_inds.view(bs * nt, 1).type_as(imgs)
        boxes = gt_trajs[:, :, 0, :].contiguous().view(bs * nt, 4)
        rois = torch.cat((roi_inds, boxes), dim=1)  # [bs*nt, 5]

        # Step 3, feed into the prediction head
        preds = self.head(feats, rois)
        return preds, rois

    def forward_train(self,
                      imgs: torch.Tensor,
                      gt_trajs: torch.Tensor,
                      gt_weights: torch.Tensor = None) -> dict:
        """ Receive video clips and ground-truth trajectory as inputs.
            This function will return the loss values (as dictionary)

        Args:
            imgs (torch.Tensor): input video clips, in shape of
                [Num_imgs, 3, T, H, W]
            gt_trajs (torch.Tensor): ground-truth trajectories, in shape of
                [Num_imgs, Num_trajs, Num_frames, 4].
            gt_weights (torch.Tensor): control whether to calculate over
                all frames, in shape of [Num_imgs, Num_trajs, Num_frames],
                If None, it will be filled with 1.
        """
        preds, rois = self._forward(imgs, gt_trajs)
        losses = self.head.loss(preds, rois, gt_trajs, gt_weights)

        return losses

    def forward_test(self,
                     imgs: torch.Tensor,
                     gt_trajs: torch.Tensor,
                     gt_weights: torch.Tensor = None) -> np.ndarray:
        """ Evaluate how precise the predictions are. This function has
            nothing to do with the feature learning, just for debug and
            analysis.

        Args:
            imgs (torch.Tensor): input video clips, in shape of
                [Num_imgs, 3, T, H, W]
            gt_trajs (torch.Tensor): ground-truth trajectories, in shape of
                [Num_imgs, Num_trajs, Num_frames, 4].
            gt_weights (torch.Tensor): control whether to calculate over
                all frames, in shape of [Num_imgs, Num_trajs, Num_frames],
                If None, it will be filled with 1.

        Returns:
            ious (np.ndarray): the iou values between the predictions and the
                ground-truth trajectory.

        """
        bs, nt, nf, _ = gt_trajs.size()
        preds, rois = self._forward(imgs, gt_trajs)
        pred_trajs = self.head.predict(preds, rois).view(bs*nt*nf, 4)
        gt_trajs = gt_trajs.view(bs*nt*nf, 4)
        # calculate the RoIs
        ious = bbox_overlaps(pred_trajs, gt_trajs, aligned=True)
        ious = ious.view(bs, nt, nf)
        ious = ious.cpu().numpy()
        return ious

    def forward(self, return_loss=True, *args, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)
