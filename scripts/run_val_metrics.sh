#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch

CONFIG_PATH=${1:-configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml}
CHECKPOINT_PATH=${2:-/kaggle/input/models/pictureinthenoise/finvolution-teach-voice-ai-when-to-speak-cp-14/pytorch/default/1/best.pt}
OUTPUT_PATH=${3:-/kaggle/working/output.csv}
MAX_EVAL_BATCHES=${4:-100}

CONFIG_PATH="${CONFIG_PATH//$'\r'/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH//$'\r'/}"
OUTPUT_PATH="${OUTPUT_PATH//$'\r'/}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES//$'\r'/}"

python src/val_metrics --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT_PATH}" --output "${OUTPUT_PATH}" --max_eval_batches "${MAX_EVAL_BATCHES}"
