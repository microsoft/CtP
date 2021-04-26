# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from ...train_step_mixin import TrainStepMixin
from ....builder import build_backbone, MODELS


@MODELS.register_module()
class MoCo(nn.Module, TrainStepMixin):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722

    The code is mainly ported from the official repo:
    https://github.com/facebookresearch/moco

    """

    def __init__(self,
                 backbone: dict,
                 in_channels: int,
                 out_channels: int,
                 queue_size: int = 65536,
                 momentum: float = 0.999,
                 temperature: float = 0.07,
                 mlp: bool = False):
        super(MoCo, self).__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        # create the encoders
        # num_classes is the output fc dimension
        self.backbone = build_backbone(backbone)  # q encoder
        self.key_encoder = build_backbone(backbone)  # key encoder

        if mlp:
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, out_channels)
            )
            self.key_fc = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, out_channels)
            )
        else:
            self.fc = nn.Linear(in_channels, out_channels)
            self.key_fc = nn.Linear(in_channels, out_channels)

        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.fc.parameters(),
                                    self.key_fc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(out_channels, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(),
                                    self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc.parameters(),
                                    self.key_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, imgs_q, imgs_k):

        q = self.backbone(imgs_q)  # [N, C, t, h, w]
        # [N, out_channels]
        q = self.fc(q.view(q.shape[0:2] + (-1,)).mean(dim=-1))
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(imgs_k)

            k = self.key_encoder(im_k)  # keys: NxC
            k = self.key_fc(k.view(k.shape[0:2] + (-1,)).mean(dim=-1))
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        nce_loss = nn.functional.cross_entropy(logits, labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return dict(nce_loss=nce_loss)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
