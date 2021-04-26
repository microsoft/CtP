# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import division

import _init_paths
import os
import re
import argparse

from mmcv import Config
from mmcv.runner import init_dist, wrap_fp16_model

from pyvrl.apis import test_network, get_root_logger, set_random_seed
from pyvrl.builder import build_model, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train an network')
    parser.add_argument('--cfg', default='', type=str, help='train config file path')
    parser.add_argument('--gpus', default=1, type=int, help='Gpu number')
    parser.add_argument('--work_dir', help='the dir to save logs and models, ')
    parser.add_argument('--checkpoint', help='checkpoint path')
    parser.add_argument('--progress', action='store_true', help='show progress bar')
    parser.add_argument('--data_dir', default='data/', type=str, help='the dir that save training data')
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.cfg)

    if 'backbone' in cfg.model:
        if 'pretrained' in cfg.model.backbone:
            cfg.model.backbone.pretrained = None

    # update configs according to CLI args
    cfg.gpus = args.gpus
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if args.checkpoint is not None:
        checkpoint = args.checkpoint
    else:
        chk_name_list = [fn for fn in os.listdir(cfg.work_dir) if fn.endswith('.pth')]
        chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
        chk_epoch_list.sort()
        checkpoint = os.path.join(cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')
    cfg.checkpoint = checkpoint

    # setup data root directory
    if args.data_dir is not None:
        if 'train' in cfg.data:
            cfg.data.train.data_dir = args.data_dir
        if 'val' in cfg.data:
            cfg.data.val.data_dir = args.data_dir
        if 'test' in cfg.data:
            cfg.data.test.data_dir = args.data_dir

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    test_dataset = build_dataset(cfg.data.val)

    test_network(model,
                 test_dataset,
                 cfg=cfg,
                 distributed=distributed,
                 logger=logger,
                 progress=args.progress)
