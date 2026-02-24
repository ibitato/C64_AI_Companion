# Release Guide: Hugging Face Publication

## Purpose

Publish LoRA and GGUF artifacts to Hugging Face with model cards and reproducible metadata.

## Preconditions

- `HF_TOKEN` available in local `.env` (write permission).
- Trained adapter artifacts exist in `models/fine-tuned`.
- GGUF artifacts exist in `models/gguf`.

## Publish Command

```bash
set -a && . ./.env && set +a
python3 scripts/release/publish_hf.py
```

## Output Repositories

Default targets:

- `ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora`
- `ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf`

## Verification

- Open the repository pages and verify all expected files.
- Confirm model cards include objective, context length, training summary, and usage examples.
