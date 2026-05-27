#!/bin/bash

PATH_TO_DATASET="../data/nq/eval/nq_eval.jsonl"
ENCODER_NAME="../baselineModel/Qwen3-Embedding-0.6B"                  
GENERATOR_NAME="../baselineModel/Mistral-7B-Instruct-v0.2"

GENERATOR_CHECKPOINT="../checkpoint/pretrainedModel/Qwen3embedding0.6B-Mistral7B-generator"
ENCODER_CHECKPOINT="../checkpoint/pretrainedModel/Qwen3embedding0.6B-Mistral7B-selector"
OUTPUT_DIR="../results/eval_results" 

mkdir -p "$OUTPUT_DIR"

python evaluate_step1_gen_results.py \
    --dataset "nq" \
    --data_path "$PATH_TO_DATASET" \
    --encoder_name "$ENCODER_NAME" \
    --encoder_checkpoint_dir "$ENCODER_CHECKPOINT" \
    --generator_name "$GENERATOR_NAME" \
    --generator_checkpoint_dir "$GENERATOR_CHECKPOINT" \
    --evaluation_results_path "$OUTPUT_DIR" \
    --batch_size 128 \
    --encoder_max_length 2560 \
    --generator_max_length 1024 \
    --num_emb_tokens 8 \
    --num_doc_tokens 2 \
    --rerank_top_k 1 \
    --device_id 0