#!/usr/bin/env bash
set -euo pipefail

cd /workspace
# Keep HF cache local to the project mount for reproducible container runs.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

if [ "$#" -gt 0 ]; then
  python scripts/fine_tune_mistral_8b.py "$@"
  exit 0
fi

MODEL_PROFILE="${MODEL_PROFILE:-8b}"
case "${MODEL_PROFILE}" in
  8b)
    MODEL_PATH="models/Ministral-3-8B-Thinking"
    OUTPUT_DIR="models/fine-tuned"
    ;;
  14b)
    MODEL_PATH="models/Ministral-3-14B-Thinking"
    OUTPUT_DIR="models/fine-tuned-14b"
    ;;
  *)
    echo "ERROR: unsupported MODEL_PROFILE='${MODEL_PROFILE}'. Use: 8b or 14b" >&2
    exit 1
    ;;
esac

python scripts/fine_tune_mistral_8b.py \
  --model-profile "${MODEL_PROFILE}" \
  --phase both \
  --model-path "${MODEL_PATH}" \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir "${OUTPUT_DIR}" \
  --precision bf16 \
  --assistant-only-loss \
  --strict-assistant-only-loss \
  --chat-template-path scripts/templates/mistral3_chat_template_assistant_mask.jinja \
  --no-packing \
  --use-lora
