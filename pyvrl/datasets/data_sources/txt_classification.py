import os


class TxtClsDataSource(object):

    def __init__(self, ann_file: str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.txt'), f'Support .txt file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'
        with open(self.ann_file, 'r') as f:
            lines = f.read().splitlines()
        self.video_info_list = []
        for line in lines:
            if line.strip() == '':
                continue
            split = line.split(' ')
            assert len(split) == 2
            name, label = split[0], int(split[1])
            self.video_info_list.append(dict(name=name, label=int(label)))

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]
