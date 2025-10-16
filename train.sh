#!/bin/bash

export EXP=predict2_video2world_training_2b_video2world_data_lora
export CUDA_VISIBLE_DEVICES=0,1,2,3
export IMAGINAIRE_OUTPUT_ROOT="/nas/jiangyuxin/code/cosmos-predict2/checkpoints"
export HF_HOME="/nas/jiangyuxin/huggingface/cache"
export WANDB_API_KEY="4e696f41874d8dda7c35b3f71231413f1289b5be"


echo "Starting training for experiment: $EXP"
echo "Number of GPUs: $CUDA_VISIBLE_DEVICES"
echo "Master port: 12341"


torchrun \
    --nproc_per_node=4 \
    --master_port=12341 \
    -m train \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment=${EXP}