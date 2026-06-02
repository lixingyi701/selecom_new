#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0          # vLLM V1 engine imports flashinfer which has an ABI mismatch

# ── Paths ──────────────────────────────────────────────────────────────────
PATH_TO_DATASET="../data/nq/eval/nq_eval.jsonl"

ENCODER_NAME="../baselineModel/Qwen3-Embedding-0.6B"
GENERATOR_NAME="../baselineModel/Qwen2.5-7B-Instruct"

# Stage 1 selector checkpoint — provides frozen encoder base weights (encoder.pt)
STAGE1_CHECKPOINT_DIR="../checkpoint/pretrainedModel/Qwen3embedding0.6B-Qwen2.57B-selector"

# Stage 3 checkpoint — provides encoder_lora/, generator_lora/, projector.pt, frozen/
STAGE3_CHECKPOINT_DIR="../results/stage3/checkpoint-1875"

# Judge model for LLM-as-a-Judge metric
JUDGE_MODEL_NAME="../baselineModel/Qwen2.5-7B-Instruct"

OUTPUT_DIR="../results/stage3/eval_results"
mkdir -p "$OUTPUT_DIR"

# ── Step 1: Generate answers ───────────────────────────────────────────────
echo "[Step 1] Generating answers with Stage 3 model..."

python evaluate_step3_gen_results.py \
    --dataset "nq" \
    --data_path "$PATH_TO_DATASET" \
    --encoder_name "$ENCODER_NAME" \
    --generator_name "$GENERATOR_NAME" \
    --stage1_checkpoint_dir "$STAGE1_CHECKPOINT_DIR" \
    --stage3_checkpoint_dir "$STAGE3_CHECKPOINT_DIR" \
    --evaluation_results_path "$OUTPUT_DIR" \
    --batch_size 8 \
    --encoder_max_length 2560 \
    --generator_max_length 1024 \
    --num_emb_tokens 8 \
    --num_doc_tokens 2 \
    --rerank_top_k 1 \
    --device_id 0

# ── Step 2: Compute metrics ────────────────────────────────────────────────
echo "[Step 2] Computing metrics..."

python evaluate_step3_metric.py \
    --result_path "$OUTPUT_DIR/nq_top1_results.jsonl" \
    --judge_model_name "$JUDGE_MODEL_NAME" \
    --num_gpus 2

# CUDA_VISIBLE_DEVICES=2,3 python evaluate_step3_metric.py \
#     --result_path "../results/stage3/eval_results/nq_top1_results.jsonl" \
#     --judge_model_name "../baselineModel/Qwen2.5-7B-Instruct" \
#     --num_gpus 2