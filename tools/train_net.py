# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import _init_paths
import os
import re
import mmcv
import torch
import argparse

from mmcv import Config
from mmcv.runner import init_dist
from mmcv.utils import collect_env
from pyvrl.apis import train_network, get_root_logger, set_random_seed
from pyvrl.builder import build_model, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train an network.')
    parser.add_argument('--cfg',
                        default='',
                        type=str, help='train config file path *.py')
    parser.add_argument('--work_dir',
                        help='the dir to save logs and models.'
                             'if not specified, program will use the path '
                             'defined in the configuration file.')
    parser.add_argument('--data_dir',
                        default='data/',
                        type=str,
                        help='the dir that save training data.'
                             '(data/ by default)')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--validate',
                        action='store_true',
                        help='whether to evaluate the checkpoint during '
                             'training')
    parser.add_argument('--gpus',
                        type=int,
                        default=1,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')
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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    cfg.gpus = args.gpus
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # if the CLI args has already specified a resume_from path,
    # we will recover training process from it.
    # otherwise, if there exists a trained model in work directory, we will resume training from it
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    else:
        if os.path.isdir(cfg.work_dir):
            chk_name_list = [fn for fn in os.listdir(cfg.work_dir) if fn.endswith('.pth')]
            # if there may exists multiple checkpoint, we will select a latest one (with a highest epoch number)
            if len(chk_name_list) > 0:
                chk_epoch_list = [int(re.findall(r'\d+', fn)[0]) for fn in chk_name_list if fn.startswith('epoch')]
                chk_epoch_list.sort()
                cfg.resume_from = os.path.join(cfg.work_dir, f'epoch_{chk_epoch_list[-1]}.pth')

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

    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.cfg)))

    # init logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_model(cfg.model)
    train_dataset = build_dataset(cfg.data.train)

    train_network(model,
                  train_dataset,
                  cfg,
                  distributed=distributed,
                  validate=args.validate,
                  logger=logger)
