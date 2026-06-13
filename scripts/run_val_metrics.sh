#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export TORCH_HOME=/root/.cache/torch
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

CONFIG_PATH=${1:-configs/whisper_qwen0_6b_constrained_event_formal_5labels_competition.yaml}
NPROC=${2:-4}
CONFIG_PATH="${CONFIG_PATH//$'\r'/}"
NPROC="${NPROC//$'\r'/}"

torchrun --nproc_per_node="${NPROC}" -m src.val_metrics --config "${CONFIG_PATH}"