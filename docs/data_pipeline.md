# C64 Data Pipeline

This project uses `scripts/data_pipeline.py` to prepare Commodore 64 documents for training.

## Stages

- `manifest`: create document inventory with hashes and metadata.
- `extract`: extract text from PDF/HTML, with optional OCR fallback for low-quality PDF pages.
- `normalize`: conservative cleanup (keeps punctuation/case and technical symbols).
- `dedup`: remove exact and near-duplicate pages.
- `build_dapt`: build domain-adaptation dataset in token-sized chunks.
- `build_sft`: build chat-format supervised dataset from extracted references.
- `validate`: compute coverage/stats and write `data/processed/validation_report.json`.
- `all`: run every stage in order.

## Recommended run

```bash
python scripts/data_pipeline.py --stage all --allow-ocr
python scripts/data_qc_report.py
```

## Key outputs

- `data/interim/manifest/manifest.parquet`
- `data/interim/extracted/pages.parquet`
- `data/interim/normalized/pages_normalized.parquet`
- `data/interim/dedup/pages_dedup.parquet`
- `data/processed/dapt/train.parquet`
- `data/processed/dapt/validation.parquet`
- `data/processed/dapt/test.parquet`
- `data/processed/sft/train.jsonl`
- `data/processed/sft/validation.jsonl`
- `data/processed/sft/test.jsonl`
- `data/processed/validation_report.json`

## Notes

- Model path is project-local only: `models/Ministral-3-8B-Thinking`.
- OCR requires local tools such as `ocrmypdf`, `tesseract`, `ghostscript`, and `unpaper`.
