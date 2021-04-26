import torch
from torch import nn
from mmcv.cnn import kaiming_init, normal_init
import math
import numpy as np
import torch.nn.functional as F

from ...train_step_mixin import TrainStepMixin
from ....builder import MODELS, build_backbone


@MODELS.register_module()
class VCOP(nn.Module, TrainStepMixin):
    """ pretext task: video clip order predction [1].

    Official github: https://github.com/xudejing/video-clip-order-prediction

    [1] Self-supervised Spatiotemporal Learning via Video Clip Order
        Prediction, CVPR'19

    """
    def __init__(self,
                 backbone,
                 vcop_head):
        super(VCOP, self).__init__()
        self.backbone = build_backbone(backbone)
        self.vcop_head = VCOPHead(**vcop_head)
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.vcop_head.init_weights()

    def forward(self,
                imgs: torch.Tensor,
                gt_labels: torch.Tensor):
        # imgs in shape of [B, N-seg, 3, T, H, W]
        batch_size, tuple_len, channels, clip_len, h, w = imgs.size()
        imgs = imgs.view(-1, channels, clip_len, h, w)
        feats = self.backbone(imgs)
        order_preds = self.vcop_head(feats)
        losses = self.vcop_head.loss(order_preds, gt_labels)
        return losses


class VCOPHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 tuple_len: int,
                 hidden_channels: int = 512,
                 dropout_ratio: float = 0.25):
        super(VCOPHead, self).__init__()
        self.tuple_len = tuple_len
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.class_num = int(math.factorial(tuple_len))
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.num_pairs = (tuple_len - 1) * tuple_len // 2
        pair_inds = [(i, j)
                     for i in range(tuple_len)
                     for j in range(i+1, tuple_len)]
        self.pair_inds = torch.LongTensor(np.array(pair_inds).reshape(-1))
        assert self.pair_inds.size(0) == self.num_pairs * 2
        self.fc2 = nn.Linear(hidden_channels * self.num_pairs, self.class_num)
        self.dropout = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        kaiming_init(self.fc1)
        normal_init(self.fc2, std=0.001)

    def forward(self, feats: torch.Tensor):
        batch_size = feats.size(0) // self.tuple_len
        assert feats.size(1) == self.in_channels
        feats = feats.view((batch_size, self.tuple_len, self.in_channels, -1))
        # apply average pooling
        # [batch_size, tuple_len, channels]
        feats = torch.mean(feats, dim=3, keepdim=False)
        pair_inds = self.pair_inds.to(feats.device)
        feats = torch.index_select(feats, dim=1, index=pair_inds).contiguous()
        feats = feats.view(batch_size * self.num_pairs, self.in_channels * 2)
        feats = self.relu(self.fc1(feats))
        feats = feats.view(batch_size, self.num_pairs * self.hidden_channels)
        feats = self.dropout(feats)
        cls_logits = self.fc2(feats)
        return cls_logits

    def loss(self, cls_logits: torch.Tensor, gt_labels: torch.Tensor):
        losses = dict()
        cls_logits = cls_logits.view(-1, self.class_num)
        batch_size = cls_logits.size(0)
        cls_preds = cls_logits.argmax(dim=1)
        losses['loss_cls'] = F.cross_entropy(cls_logits, gt_labels.view(-1))
        losses['accuracy'] = (cls_preds.eq(gt_labels.view(-1))).float().sum()
        losses['accuracy'] = losses['accuracy'] / batch_size
        return losses
