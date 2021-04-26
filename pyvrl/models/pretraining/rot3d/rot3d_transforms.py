import numpy as np
import random
import cv2
from typing import List

from ....datasets import BaseTransform
from ....builder import TRANSFORMS


@TRANSFORMS.register_module()
class GroupRectRotate(BaseTransform):

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        flag = random.randint(0, 3)
        return dict(flag=flag, img_shape=data[0].shape)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        return self.rotate(data, transform_param['flag'])

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        if transform_param['flag'] == 0:
            return boxes
        h, w = transform_param['img_shape'][0:2]
        # calculate the center of the image
        center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(center, 90 * transform_param['flag'], 1.0)
        boxes = [self.warp_boxes(box, mat) for box in boxes]
        return boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        raise NotImplementedError

    @staticmethod
    def warp_boxes(boxes, mat):
        """ Warp point by proj matrix.
        Args:
            boxes (np.ndarray): in shape of [N, 4]
            mat (np.ndarray): in shape of [2, 3], projection matrix
        """
        pts = boxes.reshape(-1, 2)
        pts_ext = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
        pts_ext = mat.dot(pts_ext.T).T
        pts_ext = pts_ext.reshape(-1, 4)
        boxes = np.concatenate([
            np.min(pts_ext[:, 0::2], axis=1, keepdims=True),
            np.min(pts_ext[:, 1::2], axis=1, keepdims=True),
            np.max(pts_ext[:, 0::2], axis=1, keepdims=True),
            np.max(pts_ext[:, 1::2], axis=1, keepdims=True),
        ], axis=1)

        return boxes

    @staticmethod
    def rotate(image, rot_flag):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        """
        if rot_flag == 0:
            return image

        # get image height, width
        if isinstance(image, np.ndarray):
            (h, w) = image.shape[0:2]
        elif isinstance(image, list):
            (h, w) = image[0].shape[0:2]
        else:
            raise NotImplementedError

        # calculate the center of the image
        center = (w / 2, h / 2)

        if rot_flag == 1:
            # Perform the counter clockwise rotation holding at the center
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
        elif rot_flag == 2:
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
        elif rot_flag == 3:
            M = cv2.getRotationMatrix2D(center, 270, 1.0)
        else:
            raise NotImplementedError

        if isinstance(image, list):
            image = [cv2.warpAffine(i, M, (h, w), borderMode=cv2.BORDER_REPLICATE) for i in image]
        else:
            image = cv2.warpAffine(image, M, (h, w), borderMode=cv2.BORDER_REPLICATE)

        return image

