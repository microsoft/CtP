# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Data preprocessing for UCF-101 dataset. """
import os
import zipfile
import mmcv
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess UCF-101 dataset')
    parser.add_argument('--raw_dir', default='data/ucf101/UCF101_raw/',
                        type=str, help='raw data directory')
    parser.add_argument('--out_dir', default='data/ucf101/',
                        type=str, help='output data directory.')
    parser.add_argument('--ann_dir', default='data/ucf101/ucfTrainTestlist/',
                        type=str, help='train/test split annotations directory.')
    return parser.parse_args()


def load_file(file_path):
    assert os.path.isfile(file_path), f'Cannot find file: {file_path}'
    rets = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.strip() == '':
                continue
            rets.append(line.split(' '))
    return rets


def video_to_zip(video_path, zip_path):
    assert os.path.isfile(video_path)
    vid = cv2.VideoCapture(video_path)
    mmcv.mkdir_or_exist(os.path.dirname(zip_path))
    zid = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED)
    frame_index = 1
    while True:
        ret, img = vid.read()
        if img is None:
            break
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        zid.writestr('img_{:05d}.jpg'.format(frame_index), img_str)
        frame_index += 1
    zid.close()
    vid.release()


if __name__ == '__main__':
    args = parse_args()
    # Step 1, load class ID annotations
    ids_file = os.path.join(args.ann_dir, 'classInd.txt')
    ids_map = {sp[1]: int(sp[0]) for sp in load_file(ids_file)}

    # Step 2, load training & test annotations
    all_video_name_list = []
    for prefix in ['train', 'test']:
        ann_file = os.path.join(args.ann_dir, f'{prefix}list01.txt')
        video_name_list = [sp[0][:-4] for sp in load_file(ann_file)]
        video_label_list = [ids_map[name.split('/')[0]] for name in video_name_list]
        out_file = os.path.join(args.out_dir, 'annotations', f'{prefix}_split_1.txt')
        mmcv.mkdir_or_exist(os.path.dirname(out_file))
        with open(out_file, 'w') as f:
            for video_name, video_label in zip(video_name_list, video_label_list):
                f.write(f'{video_name} {video_label}\n')
        all_video_name_list.extend(video_name_list)

    # Step 3, convert .avi raw video to zipfile
    prog_bar = mmcv.ProgressBar(len(all_video_name_list))
    for video_name in all_video_name_list:
        video_path = os.path.join(args.raw_dir, f'{video_name.split("/")[1]}.avi')
        zip_path = os.path.join(args.out_dir, 'zips', f'{video_name}.zip')
        video_to_zip(video_path, zip_path)
        prog_bar.update()
