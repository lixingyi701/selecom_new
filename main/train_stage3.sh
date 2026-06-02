#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export MASTER_PORT=29507
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_PATH="../data/stage2/stage2_train_data.jsonl"
ENCODER_NAME="../baselineModel/Qwen3-Embedding-0.6B"
GENERATOR_NAME="../baselineModel/Qwen2.5-7B-Instruct"
STAGE1_CHECKPOINT_DIR="../checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector"
STAGE2_CHECKPOINT_DIR="../checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-generator"
OUTPUT_DIR="../results/stage3"
LOG_DIR="../results/stage3/logs"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_stage3.py \
    --data_path "$DATA_PATH" \
    --encoder_name "$ENCODER_NAME" \
    --generator_name "$GENERATOR_NAME" \
    --stage1_checkpoint_dir "$STAGE1_CHECKPOINT_DIR" \
    --stage2_checkpoint_dir "$STAGE2_CHECKPOINT_DIR" \
    --model_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --log_name "train_stage3.log" \
    --max_samples 10000 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --projector_lr_multiplier 1.5
