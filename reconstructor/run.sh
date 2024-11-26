#!/usr/bin/env bash
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))
export OMP_NUM_THREADS=4

python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --nnodes=$NUM_NODES \
        --node_rank $NODE_RANK \
        --rdzv_backend=c10d \
        train_main.py --use_ddp=True --train_nl_color=False
