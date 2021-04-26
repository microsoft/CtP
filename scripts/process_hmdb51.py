# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Preprocessing for HMDB-51 dataset. """
import os
import argparse
import cv2
import zipfile
import numpy as np
from mmcv import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(description='Build HMDB-51 Dataset')
    parser.add_argument('--raw_dir', default='data/hmdb51/HMDB51_raw/', type=str, help='HMDB-51 data dir')
    parser.add_argument('--ann_dir', default='data/hmdb51/test_train_splits/', type=str, help='HMDB-51 split info dir')
    parser.add_argument('--out_dir', default='data/hmdb51/', help='output dir')
    args = parser.parse_args()

    return args


def video_to_zip(video_path, zip_path, frame_fmt='img_{:05d}.jpg'):
    vid = cv2.VideoCapture(video_path)
    zid = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED)
    frame_id = 1  # start from index 1
    while True:
        ret, frame = vid.read()
        if frame is None:
            break
        encode_data = cv2.imencode('.jpg', frame)[1]
        encode_str = np.array(encode_data).tostring()
        zid.writestr(frame_fmt.format(frame_id), encode_str)
        frame_id += 1

    zid.close()
    vid.release()


def main():
    args = parse_args()
    data_dir = args.raw_dir
    output_dir = args.out_dir
    zip_dir = os.path.join(output_dir, 'zips')
    ann_dir = os.path.join(output_dir, 'annotations')
    mkdir_or_exist(zip_dir)
    mkdir_or_exist(ann_dir)

    cls_name_list = [fn for fn in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fn))]
    assert len(cls_name_list) == 51
    print("All {} classes".format(len(cls_name_list)))

    video_path_list = []
    video_name_list = []
    for cls_id, cls_name in enumerate(cls_name_list):
        cls_path = os.path.join(data_dir, cls_name)
        file_list = [fn for fn in os.listdir(cls_path) if fn.endswith('.avi')]
        for file_name in file_list:
            video_path_list.append(os.path.join(cls_path, file_name))
            video_name_list.append(file_name[:-4])
    print("All {} videos".format(len(video_path_list)))

    print("Generate annotations...")
    for sp_id in range(3):
        train_fid = open(os.path.join(ann_dir, 'train_split_{}.txt'.format(sp_id+1)), 'w')
        test_fid = open(os.path.join(ann_dir, 'test_split_{}.txt'.format(sp_id+1)), 'w')
        print("Annotation split {}".format(sp_id+1))
        prog_bar = ProgressBar(len(cls_name_list))
        for cls_id, cls_name in enumerate(cls_name_list):
            sp_file_path = os.path.join(args.ann_dir, '{}_test_split{}.txt'.format(cls_name, sp_id + 1))
            with open(sp_file_path, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                if line.strip() == '':
                    continue
                video_name, tid = line.split(' ')[0:2]
                assert video_name.endswith('.avi')
                video_name = video_name[:-4]
                tid = int(tid)
                assert tid in (0, 1, 2)
                if tid == 1:
                    train_fid.write('{} {}\n'.format(video_name, cls_id+1))
                elif tid == 2:
                    test_fid.write('{} {}\n'.format(video_name, cls_id+1))
            prog_bar.update()

        train_fid.close()
        test_fid.close()

    print("Generate zip files...")
    prog_bar = ProgressBar(len(video_path_list))
    for i in range(len(video_path_list)):
        video_name = video_name_list[i]
        video_path = video_path_list[i]
        zip_path = os.path.join(zip_dir, '{}.zip'.format(video_name))
        video_to_zip(video_path, zip_path)
        prog_bar.update()


if __name__ == '__main__':
    main()
