# Copyright 2018-2019 Open-MMLab. All rights reserved.

import torch
import os
import mmcv

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDistributedDataParallel
from .env import get_root_logger
from .test import multi_gpu_test
from ..datasets.dataloader import build_dataloader


def test_network(model,
                 dataset,
                 cfg,
                 distributed=False,
                 logger=None,
                 progress=False):
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    logger.info(f"Ckpt path: {cfg.checkpoint}")
    if cfg.checkpoint.endswith('.pth'):
        prefix = os.path.basename(cfg.checkpoint)[:-4]
    else:
        prefix = 'unspecified'

    out_name = f'eval_{dataset.__class__.__name__}_{dataset.name}'
    output_dir = os.path.join(cfg.work_dir, out_name)
    mmcv.mkdir_or_exist(output_dir)

    cache_path = os.path.join(output_dir, f'{prefix}_results.pkl')
    if os.path.isfile(cache_path):
        logger.info(f"Load results from {cache_path}")
        results = mmcv.load(cache_path)
    else:
        load_checkpoint(model, cfg.checkpoint, logger=logger)
        # save config in model
        model.cfg = cfg
        # build dataloader
        multiprocessing_context = None
        if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
            multiprocessing_context = 'spawn'
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            multiprocessing_context=multiprocessing_context
        )

        # start training
        if distributed:
            if cfg.get('syncbn', False):
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False
            )
        else:
            raise NotImplementedError

        results = multi_gpu_test(model, data_loader, progress=progress)
        mmcv.dump(results, cache_path)

    # evaluate results
    eval_results = dataset.evaluate(results, logger)
    mmcv.dump(eval_results, os.path.join(output_dir,
                                         f'{prefix}_eval_results.json'))
