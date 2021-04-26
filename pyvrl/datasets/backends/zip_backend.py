import os
import cv2
import numpy as np
import zipfile
from typing import List


class ZipItem(object):
    """ Zip storage item for loading images from a video.
    Each video clip has one corresponding zip file, which stores
    the video frames (like 00001.jpg, 00002.jpg, ...) or flow information.
    """

    def __init__(self, video_info: dict, zip_fmt: str, frame_fmt: str):
        self.zip_path = zip_fmt.format(video_info['name'])
        self.frame_fmt = frame_fmt
        self.frame_zip_fid = None

    def __len__(self):
        if self.frame_zip_fid is None:
            self._check_available(self.zip_path)
            self.frame_zip_fid = zipfile.ZipFile(self.zip_path, 'r')
        namelist = self.frame_zip_fid.namelist()
        namelist = [name for name in namelist if name.endswith('.jpg')]
        return len(namelist)

    def close(self):
        if self.frame_zip_fid is not None:
            self.frame_zip_fid.close()

    def get_frame(self, indices: List[int]) -> List[np.ndarray]:
        """ Load image frames from the given zip file.
        Args:
            indices: frame index list (0-based index)
        Returns:
            img_list: the loaded image list, each element is a np.ndarray in shape of [H, W 3]
        """
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        if self.frame_zip_fid is None:
            self._check_available(self.zip_path)
            self.frame_zip_fid = zipfile.ZipFile(self.zip_path, 'r')

        for idx in indices:
            file_name = self.frame_fmt.format(int(idx) + 1)
            img = self.load_image_from_zip(self.frame_zip_fid, file_name, cv2.IMREAD_COLOR)
            img_list.append(img)
        return img_list

    @staticmethod
    def load_image_from_zip(zip_fid, file_name, flag=cv2.IMREAD_COLOR):
        file_content = zip_fid.read(file_name)
        img = cv2.imdecode(np.fromstring(file_content, dtype=np.uint8), flag)
        return img

    @staticmethod
    def _check_available(zip_path):
        if zip_path is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError("Cannot find zip file {}".format(zip_path))


class ZipBackend(object):

    def __init__(self,
                 zip_fmt: str,
                 frame_fmt: str = 'img_{:05d}.jpg',
                 data_dir: str = None):
        if data_dir is not None:
            zip_fmt = os.path.join(data_dir, zip_fmt)
        self.zip_fmt = zip_fmt
        self.frame_fmt = frame_fmt

    def open(self, video_info) -> ZipItem:
        return ZipItem(video_info, self.zip_fmt, self.frame_fmt)
