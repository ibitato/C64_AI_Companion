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

Manual run with explicit contract flags:

```bash
python scripts/data_pipeline.py \
  --stage all \
  --allow-ocr \
  --max-examples-per-page 3 \
  --strict-thinking-contract
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
- SFT generation filters low-signal boilerplate pages (for example: table-of-contents and copyright pages) and very noisy pages before creating chat examples.
- SFT now includes multi-turn examples to improve format retention during chat.
- Validation tracks:
  - THINK-tag coverage
  - THINK diversity (`unique_think_texts`)
  - multi-turn coverage
- In strict mode, validation fails the pipeline if the reasoning contract is broken.
