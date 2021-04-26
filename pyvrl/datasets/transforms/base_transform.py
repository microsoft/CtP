# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from typing import Any


class BaseTransform(object):
    """ This is the base class for data augmentation. """

    def get_transform_param(self, *args, **kwargs):
        """ Generate some necessary transformer parameters. """
        return None

    def apply_image(self,
                    data: Any,
                    transform_param: Any = None,
                    return_transform_param: bool = False):
        if transform_param is None:
            transform_param = self.get_transform_param(data)
        data = self._apply_image(data, transform_param)
        if return_transform_param:
            return data, transform_param
        else:
            return data

    def _apply_image(self,
                     data: Any,
                     transform_param: Any):
        raise NotImplementedError

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: Any):
        return boxes

    def apply_flow(self,
                   flows: Any,
                   transform_param: Any):
        return flows

