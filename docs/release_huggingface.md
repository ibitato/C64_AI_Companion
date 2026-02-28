# Release Guide: Hugging Face Publication

## Purpose

Publish LoRA and GGUF artifacts to Hugging Face with model cards and reproducible metadata.

Authoritative targets:

- GitHub repository: https://github.com/ibitato/C64_AI_Companion
- LoRA model: https://huggingface.co/ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora
- GGUF model: https://huggingface.co/ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf

## Preconditions

- `HF_TOKEN` available in local `.env` (write permission).
- Trained adapter artifacts exist in `models/fine-tuned`.
- GGUF artifacts exist in `models/gguf`.

## Publish Command

```bash
set -a && . ./.env && set +a
python3 scripts/release/publish_hf.py
```

Notes:

- `scripts/release/publish_hf.py` derives training metadata from local artifacts (latest DAPT/SFT checkpoints, processed dataset split sizes, and training arguments).
- The script updates model cards and uploads reproducibility files (`training_summary.json`, `trainer_state_*.json`) in the LoRA repo.
- Both Hugging Face model cards include an explicit backlink to this GitHub repository.

## Output Repositories

Default targets:

- `ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora`
- `ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf`

## Verification

- Open the repository pages and verify all expected files.
- Confirm model cards include objective, context length, training summary, and usage examples.
- Confirm bidirectional references:
  - GitHub `README.md` links to LoRA/GGUF HF repos.
  - HF model cards link back to GitHub.

Optional API-level verification:

```bash
python3 - <<'PY'
import json, urllib.request
repos = [
    "ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora",
    "ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf",
]
for repo in repos:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        info = json.load(r)
    print(repo, "lastModified=", info.get("lastModified"))
PY
```
