#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

RESULT_PATH="../results/eval_results/nq_top1_results.jsonl" 
JUDGE_MODEL_NAME="../baselineModel/Mistral-7B-Instruct-v0.2"

python evaluate_step2_metric.py \
    --result_path "$RESULT_PATH" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    --num_gpus 4
