# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from mmcv.runner import obj_from_dict

from . import backends
from . import data_sources
from . import frame_samplers


def build_backend(cfg, default_args=None):
    return obj_from_dict(cfg, backends, default_args)


def build_data_source(cfg, default_args=None):
    return obj_from_dict(cfg, data_sources, default_args)


def build_frame_sampler(cfg, default_args=None):
    return obj_from_dict(cfg, frame_samplers, default_args)
