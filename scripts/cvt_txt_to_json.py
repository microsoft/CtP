# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Convert txt annotations to json annotations. """
import json


def main():
    ann_path = 'train_split_1.txt'
    out_path = 'train.json'
    with open(ann_path, 'r') as f:
        lines = f.read().splitlines()
    anns = []
    for line in lines:
        if line.strip() == '':
            continue
        name, label = line.split(' ')
        anns.append(dict(name=name, label=int(label)+1))
    with open(out_path, 'w') as f:
        json.dump(anns, f, indent=2)


if __name__ == '__main__':
    main()
