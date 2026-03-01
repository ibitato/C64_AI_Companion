#!/usr/bin/env bash
set -euo pipefail

cd /workspace
# Keep HF cache local to the project mount for reproducible container runs.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

if [ "$#" -gt 0 ]; then
  python scripts/data_pipeline.py "$@"
else
  MODEL_PROFILE="${MODEL_PROFILE:-8b}"
  case "${MODEL_PROFILE}" in
    8b)
      MODEL_PATH="models/Ministral-3-8B-Thinking"
      ;;
    14b)
      MODEL_PATH="models/Ministral-3-14B-Thinking"
      ;;
    *)
      echo "ERROR: unsupported MODEL_PROFILE='${MODEL_PROFILE}'. Use: 8b or 14b" >&2
      exit 1
      ;;
  esac

  python scripts/data_pipeline.py \
    --stage all \
    --allow-ocr \
    --model-profile "${MODEL_PROFILE}" \
    --model-path "${MODEL_PATH}"
fi

if [ ! -f data/processed/validation_report.json ]; then
  echo "WARN: data/processed/validation_report.json not found. Skipping docs/data_qc_report.md generation." >&2
  exit 0
fi

python scripts/data_qc_report.py \
  --input data/processed/validation_report.json \
  --output docs/data_qc_report.md
