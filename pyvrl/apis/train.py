# Copyright 2018-2019 Open-MMLab. All rights reserved.

from __future__ import division

import torch

from mmcv.runner import (DistSamplerSeedHook, build_optimizer,
                         EpochBasedRunner, OptimizerHook)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from ..core import EvalHook, DistEvalHook
from ..datasets.dataloader import build_dataloader
from ..builder import build_dataset


def train_network(model,
                  dataset,
                  cfg,
                  distributed=False,
                  validate=False,
                  logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    multiprocessing_context = None
    if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
        # https://github.com/pytorch/pytorch/issues/5059
        logger.info("Known numpy random seed issue in "
                    "https://github.com/pytorch/pytorch/issues/5059")
        logger.info("Switch to use spawn method for dataloader.")
        multiprocessing_context = 'spawn'
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=cfg.gpus,
            dist=distributed,
            multiprocessing_context=multiprocessing_context)
    ]

    # start training
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # convert to syncbnc
        if cfg.get('syncbn', False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(model,
                              optimizer=optimizer,
                              work_dir=cfg.work_dir,
                              logger=logger)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        # it seems that fp16 still has some unknown bugs.
        # I cannot get a reasonable results in this mode.
        raise NotImplementedError("Cannot support FP16 yet.")
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            drop_last=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    logger.info("Finish training... Exit... ")
