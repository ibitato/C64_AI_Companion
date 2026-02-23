#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

if [ "$#" -gt 0 ]; then
  python scripts/fine_tune_mistral_8b.py "$@"
  exit 0
fi

python scripts/fine_tune_mistral_8b.py \
  --phase both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --precision bf16 \
  --no-assistant-only-loss \
  --no-packing \
  --use-lora
