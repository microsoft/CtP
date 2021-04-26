import os
import json


class JsonClsDataSource(object):

    def __init__(self, ann_file: str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'
        with open(self.ann_file, 'r') as f:
            self.video_info_list = json.load(f)

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]
