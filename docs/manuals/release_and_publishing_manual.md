# Release and Publishing Manual

## Purpose

Publish trained artifacts to Hugging Face in a reproducible and auditable way.

## Preconditions

- `.env` contains `HF_TOKEN` with write permissions.
- LoRA and GGUF artifacts are generated and validated.

## Procedure

1. Load local `.env` token into shell session.
2. Run `scripts/release/publish_hf.py`.
3. Verify files and model card content in both target repos.

## Validation Checkpoints

- LoRA repo includes adapter artifacts and metadata.
- GGUF repo includes all quantizations and Modelfiles.
- Model cards are complete and professionally formatted.

## Failure Modes

- 403 on repo create/upload (token role issue).
- Large file transfer interruptions.

## Recovery

- Regenerate/upgrade token permissions.
- Re-run publisher (idempotent upload behavior expected).
