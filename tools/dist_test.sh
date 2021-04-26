#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 $(dirname "$0")/test_net.py --cfg $1 --launcher pytorch ${@:3}
