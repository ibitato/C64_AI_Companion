#!/usr/bin/env bash
set -euo pipefail

cd /workspace
# Keep HF cache local to the project mount for reproducible container runs.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

MODEL_PROFILE="${MODEL_PROFILE:-8b}"
HAS_STAGE=0
HAS_MODEL_PROFILE=0
HAS_MODEL_PATH=0
HAS_ALLOW_OCR=0
HAS_HELP=0

args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  arg="${args[$i]}"
  case "$arg" in
    -h|--help)
      HAS_HELP=1
      ;;
    --stage|--stage=*)
      HAS_STAGE=1
      ;;
    --model-profile)
      HAS_MODEL_PROFILE=1
      if [ $((i + 1)) -lt "${#args[@]}" ]; then
        MODEL_PROFILE="${args[$((i + 1))]}"
      fi
      i=$((i + 1))
      ;;
    --model-profile=*)
      HAS_MODEL_PROFILE=1
      MODEL_PROFILE="${arg#*=}"
      ;;
    --model-path|--model-path=*)
      HAS_MODEL_PATH=1
      if [ "$arg" = "--model-path" ]; then
        i=$((i + 1))
      fi
      ;;
    --allow-ocr)
      HAS_ALLOW_OCR=1
      ;;
  esac
  i=$((i + 1))
done

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

if [ "${HAS_HELP}" -eq 1 ]; then
  python scripts/data_pipeline.py "$@"
else
  cmd=(python scripts/data_pipeline.py)
  if [ "${HAS_STAGE}" -eq 0 ]; then
    cmd+=(--stage all)
  fi
  if [ "${HAS_ALLOW_OCR}" -eq 0 ]; then
    cmd+=(--allow-ocr)
  fi
  if [ "${HAS_MODEL_PROFILE}" -eq 0 ]; then
    cmd+=(--model-profile "${MODEL_PROFILE}")
  fi
  if [ "${HAS_MODEL_PATH}" -eq 0 ]; then
    cmd+=(--model-path "${MODEL_PATH}")
  fi
  cmd+=("$@")
  "${cmd[@]}"
fi

if [ ! -f data/processed/validation_report.json ]; then
  echo "WARN: data/processed/validation_report.json not found. Skipping docs/data_qc_report.md generation." >&2
  exit 0
fi

python scripts/data_qc_report.py \
  --input data/processed/validation_report.json \
  --output docs/data_qc_report.md
