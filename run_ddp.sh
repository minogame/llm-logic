#!/bin/bash

# DDP训练启动脚本
# 使用torchrun启动分布式训练

# 设置环境变量 - 这里指定物理GPU设备
export CUDA_VISIBLE_DEVICES=3,2,1,0 # 指定要使用的GPU

# 单机多卡训练
torchrun --nproc_per_node=4 \
         --master_port=29500 \
         train_rs.py

# 如果需要多机训练，使用以下命令：
# torchrun --nnodes=2 \
#          --nproc_per_node=4 \
#          --node_rank=0 \
#          --master_addr="192.168.1.100" \
#          --master_port=29500 \
#          pretrain_better_transformer.py