#! /bin/bash

# print the command to be executed
set -x

# This script runs the offline GRPO sampling for the boolean logic dataset.

for dataset_sampling_id in {0..49}
do
    CUDA_VISIBLE_DEVICES=2 python3 offline_grpo_sampling.py --dataset_sampling_id=$dataset_sampling_id &
    CUDA_VISIBLE_DEVICES=3 python3 offline_grpo_sampling.py --dataset_sampling_id=$(($dataset_sampling_id + 50))
done