#!/usr/bin/env bash

## run the training
python test.py \
--dataroot datasets/headspace100 \
--name headspace100 \
--ncf 64 128 256 256 \
--pool_res 9000 5000 3000 1800 \
--pool_res 9000 7000 5000 3800 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--niter_decay 0 \
--gpu_ids -1 \
--niter 30 \
--ninput_edges 13000 \
--num_aug 1 \
--dataset_mode regression \
--print_freq 1 \
--num_threads 0 ## for debugging