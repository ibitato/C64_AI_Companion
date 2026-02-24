# Data Pipeline Guide

## Purpose

Prepare a clean and auditable C64 dataset for DAPT and SFT from source manuals in `c64_docs/`.

## Pipeline Stages

- `manifest`
- `extract`
- `normalize`
- `dedup`
- `build_dapt`
- `build_sft`
- `validate`
- `all`

## Standard Execution

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh
```

## Expected Outputs

- `data/interim/manifest/manifest.parquet`
- `data/interim/extracted/pages.parquet`
- `data/interim/normalized/pages_normalized.parquet`
- `data/interim/dedup/pages_dedup.parquet`
- `data/processed/dapt/{train,validation,test}.parquet`
- `data/processed/sft/{train,validation,test}.jsonl`
- `data/processed/validation_report.json`
- `docs/data_qc_report.md`

## Data Quality Validation

The quality report is generated from `validation_report.json`.

```bash
python scripts/data_qc_report.py \
  --input data/processed/validation_report.json \
  --output docs/data_qc_report.md
```

## Notes

- Base model tokenizer path is policy-restricted to `models/Ministral-3-8B-Thinking`.
- OCR is enabled in container pipeline execution.
