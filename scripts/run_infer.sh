#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/run_infer.sh <checkpoint_path>  <output_pred_csv> <test_root> [config_path]"
  exit 1
fi

export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CKPT_PATH=$1
CKPT_LORA_PATH=$2
OUT_CSV=$3
TEST_ROOT=$4
CONFIG_PATH=${5:-configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml}
CKPT_PATH="${CKPT_PATH//$'\r'/}"
CKPT_LORA_PATH="${CKPT_LORA_PATH//$'\r'/}"
OUT_CSV="${OUT_CSV//$'\r'/}"
TEST_ROOT="${TEST_ROOT//$'\r'/}"
CONFIG_PATH="${CONFIG_PATH//$'\r'/}"

python3 -m src.infer_test --config "${CONFIG_PATH}" --checkpoint "${CKPT_PATH}" --checkpoint_lora "${CKPT_LORA_PATH}" --test_root "${TEST_ROOT}" --output_csv "${OUT_CSV}"
