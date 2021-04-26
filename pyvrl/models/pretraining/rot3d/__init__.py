"""
Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction
Longlong Jing, Xiaodong Yang, Jingen Liu, Yingli Tian
"""
from .rot3d_dataset import RotateDataset
from .rot3d_model import Rot3D
from .rot3d_transforms import GroupRectRotate

__all__ = ['Rot3D', 'RotateDataset', 'GroupRectRotate']
