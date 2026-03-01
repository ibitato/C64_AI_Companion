#!/usr/bin/env bash
set -euo pipefail

cd /workspace
# Keep HF cache local to the project mount for reproducible container runs.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

if [ "$#" -gt 0 ]; then
  python scripts/export_gguf.py "$@"
  exit 0
fi

MODEL_PROFILE="${MODEL_PROFILE:-8b}"
case "${MODEL_PROFILE}" in
  8b|14b)
    ;;
  *)
    echo "ERROR: unsupported MODEL_PROFILE='${MODEL_PROFILE}'. Use: 8b or 14b" >&2
    exit 1
    ;;
esac

python scripts/export_gguf.py \
  --model-profile "${MODEL_PROFILE}" \
  --quantization Q4_K_M
