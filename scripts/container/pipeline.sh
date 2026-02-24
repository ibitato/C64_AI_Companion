#!/usr/bin/env bash
set -euo pipefail

cd /workspace
# Keep HF cache local to the project mount for reproducible container runs.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

python scripts/data_pipeline.py --stage all --allow-ocr "$@"
python scripts/data_qc_report.py \
  --input data/processed/validation_report.json \
  --output docs/data_qc_report.md
