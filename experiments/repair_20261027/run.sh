#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd -- "$EXPERIMENT_DIR/../.." && pwd)"
export SELECOM_ROOT="${SELECOM_ROOT:-$REPOSITORY_DIR}"
export SELECOM_RUN_DIR="${SELECOM_RUN_DIR:-$SELECOM_ROOT/research/reproduced-20261027}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ACTION="${1:-check}"
case "$ACTION" in
  check)
    exec "$PYTHON_BIN" "$EXPERIMENT_DIR/check_assets.py"
    ;;
  audit) SCRIPT=remote_audit.py; EXTRA=() ;;
  evaluate-baseline)
    SCRIPT=canonical_eval.py
    EXTRA=(--samples 4462 --batch-size 4 --output-name canonical-4462-greedy-cpuload)
    ;;
  train-front-g0) SCRIPT=controlled_train.py; EXTRA=(--condition G0 --output controlled-G0-fp32) ;;
  train-front-g2) SCRIPT=controlled_train.py; EXTRA=(--condition G2 --output controlled-G2-fp32) ;;
  evaluate-fronts) SCRIPT=eval_controlled.py; EXTRA=() ;;
  train-generator) SCRIPT=generator_only_train.py; EXTRA=() ;;
  evaluate-generator) SCRIPT=eval_generator_only.py; EXTRA=() ;;
  intervention) SCRIPT=intervention_eval.py; EXTRA=() ;;
  *) printf '%s\n' 'Unknown action; see experiments/repair_20261027/README.md' >&2; exit 2 ;;
esac
"$PYTHON_BIN" "$EXPERIMENT_DIR/check_assets.py"
mkdir -p -- "$SELECOM_RUN_DIR"
timeout --signal=TERM 10800 "$PYTHON_BIN" -u "$EXPERIMENT_DIR/scripts/$SCRIPT" "${EXTRA[@]}" \
  2>&1 | tee -a "$SELECOM_RUN_DIR/$ACTION.log"
