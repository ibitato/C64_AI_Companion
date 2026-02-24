#!/usr/bin/env python3
"""Publish C64 fine-tuning artifacts to Hugging Face Hub.

This script publishes two model repos:
1) LoRA adapter repo (PEFT artifacts)
2) GGUF repo (llama.cpp / Ollama artifacts)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_LORA_REPO = "ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora"
DEFAULT_GGUF_REPO = "ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf"
BASE_MODEL_ID = "mistralai/Ministral-3-8B-Reasoning-2512"
GITHUB_REPO_URL = "https://github.com/ibitato/C64_AI_Companion"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_DIR = PROJECT_ROOT / "models" / "fine-tuned"
GGUF_DIR = PROJECT_ROOT / "models" / "gguf"
BASE_CONFIG = PROJECT_ROOT / "models" / "Ministral-3-8B-Thinking" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish LoRA + GGUF artifacts to Hugging Face.")
    parser.add_argument("--lora-repo-id", default=DEFAULT_LORA_REPO)
    parser.add_argument("--gguf-repo-id", default=DEFAULT_GGUF_REPO)
    parser.add_argument("--private", action="store_true", help="Create private repos.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable that stores the HF token (default: HF_TOKEN).",
    )
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_token(token_env: str) -> str:
    """Read HF token from environment and fail if missing."""
    token = os.getenv(token_env) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF token found. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) with write access."
        )
    return token


def hardlink_or_copy(src: Path, dst: Path) -> None:
    """Stage files via hardlink when possible, copy otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_context_length() -> int:
    """Load max context length from the local base-model config."""
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["text_config"]["max_position_embeddings"])


def load_training_summary() -> dict[str, object]:
    """Collect compact metadata from local training outputs for model cards."""
    sft_state_path = LORA_DIR / "checkpoint-132" / "trainer_state.json"
    dapt_state_path = PROJECT_ROOT / "models" / "fine-tuned-dapt" / "checkpoint-39" / "trainer_state.json"

    sft_state = {}
    dapt_state = {}
    if sft_state_path.exists():
        sft_state = json.loads(sft_state_path.read_text(encoding="utf-8"))
    if dapt_state_path.exists():
        dapt_state = json.loads(dapt_state_path.read_text(encoding="utf-8"))

    return {
        "data_splits": {
            "dapt_train": 408,
            "dapt_validation": 27,
            "dapt_test": 45,
            "sft_train": 1387,
            "sft_validation": 166,
            "sft_test": 109,
        },
        "training_recipe": {
            "pipeline": "DAPT + SFT (LoRA)",
            "precision": "bf16",
            "max_length": 2048,
            "batch_size_per_device": 2,
            "gradient_accumulation": 16,
            "learning_rate": 2e-5,
            "epochs": 3.0,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_scope": "language_qkvo",
            "assistant_only_loss": False,
            "packing": False,
        },
        "run_state": {
            "dapt_global_step": dapt_state.get("global_step"),
            "sft_global_step": sft_state.get("global_step"),
            "sft_last_logged_loss": (
                sft_state.get("log_history", [{}])[-1].get("loss")
                if sft_state.get("log_history")
                else None
            ),
            "sft_last_logged_token_accuracy": (
                sft_state.get("log_history", [{}])[-1].get("mean_token_accuracy")
                if sft_state.get("log_history")
                else None
            ),
        },
    }


def render_lora_card(context_length: int, summary: dict[str, object], repo_id: str) -> str:
    """Render markdown model card content for the LoRA repository."""
    data = summary["data_splits"]
    recipe = summary["training_recipe"]
    run = summary["run_state"]

    return f"""---
license: apache-2.0
base_model:
- {BASE_MODEL_ID}
library_name: peft
pipeline_tag: text-generation
tags:
- peft
- lora
- reasoning
- commodore-64
- c64
- rocm
language:
- en
---

# {repo_id}

## Overview

This repository contains the **LoRA adapter** produced by fine-tuning a reasoning-capable Ministral 3 8B model on technical Commodore 64 material.

Objective:
- keep the reasoning behavior of the base model,
- inject C64-specific technical knowledge,
- support practical troubleshooting and low-level explanations (BASIC, KERNAL, memory map, VIC-II, SID, 6502/6510).

Project source code and pipeline:
- {GITHUB_REPO_URL}

## Base Model

- Base model: `{BASE_MODEL_ID}`
- Architecture: `Mistral3ForConditionalGeneration` (language component fine-tuned with LoRA)
- Max context length: **{context_length:,} tokens** (from `text_config.max_position_embeddings`)

## Training Data (project-local corpus)

- DAPT: train={data["dapt_train"]}, validation={data["dapt_validation"]}, test={data["dapt_test"]}
- SFT: train={data["sft_train"]}, validation={data["sft_validation"]}, test={data["sft_test"]}
- Sources: curated Commodore 64 manuals and technical documents from this project.

## Training Recipe

- Pipeline: {recipe["pipeline"]}
- Precision: {recipe["precision"]}
- Max sequence length: {recipe["max_length"]}
- Batch size per device: {recipe["batch_size_per_device"]}
- Gradient accumulation: {recipe["gradient_accumulation"]}
- Learning rate: {recipe["learning_rate"]}
- Epochs: {recipe["epochs"]}
- LoRA: r={recipe["lora_r"]}, alpha={recipe["lora_alpha"]}, dropout={recipe["lora_dropout"]}, scope={recipe["lora_scope"]}
- SFT options: assistant_only_loss={recipe["assistant_only_loss"]}, packing={recipe["packing"]}

Run summary:
- DAPT global steps: {run["dapt_global_step"]}
- SFT global steps: {run["sft_global_step"]}
- Last logged SFT loss: {run["sft_last_logged_loss"]}
- Last logged SFT token accuracy: {run["sft_last_logged_token_accuracy"]}

## Usage (Transformers + PEFT)

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from transformers.models.mistral3 import Mistral3ForConditionalGeneration

base_id = "{BASE_MODEL_ID}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base_model = Mistral3ForConditionalGeneration.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, adapter_id)

prompt = "Explain the C64 SID chip in one concise paragraph."
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Limitations

- This adapter is specialized for C64 technical assistance, not for broad benchmark optimization.
- No additional safety fine-tuning was applied beyond the base model behavior.
- Evaluation in this repo focuses on training diagnostics; downstream benchmarking should be done per target use case.
"""


def render_gguf_card(context_length: int, repo_id: str) -> str:
    """Render markdown model card content for the GGUF repository."""
    files = [
        "c64-ministral-3-8b-thinking-c64-F16.gguf",
        "c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf",
        "c64-ministral-3-8b-thinking-c64-Q6_K.gguf",
        "c64-ministral-3-8b-thinking-c64-Q8_0.gguf",
    ]
    sizes = {}
    for name in files:
        p = GGUF_DIR / name
        sizes[name] = f"{p.stat().st_size / (1024 ** 3):.2f} GiB" if p.exists() else "missing"

    return f"""---
license: apache-2.0
base_model:
- {DEFAULT_LORA_REPO}
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- ollama
- reasoning
- commodore-64
- c64
- rocm
language:
- en
---

# {repo_id}

## Overview

GGUF exports of the C64-focused reasoning fine-tune, ready for **llama.cpp** and **Ollama**.

Project source code and training pipeline:
- {GITHUB_REPO_URL}

## Technical Details

- Derived from: `{BASE_MODEL_ID}` + project LoRA adaptation
- Context length in GGUF metadata: **{context_length:,} tokens**
- Architecture in GGUF: `mistral3`

## Included Files

| File | Size |
| --- | --- |
| `c64-ministral-3-8b-thinking-c64-F16.gguf` | {sizes["c64-ministral-3-8b-thinking-c64-F16.gguf"]} |
| `c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf` | {sizes["c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf"]} |
| `c64-ministral-3-8b-thinking-c64-Q6_K.gguf` | {sizes["c64-ministral-3-8b-thinking-c64-Q6_K.gguf"]} |
| `c64-ministral-3-8b-thinking-c64-Q8_0.gguf` | {sizes["c64-ministral-3-8b-thinking-c64-Q8_0.gguf"]} |

`Modelfile` templates are included for direct Ollama import.

## Quick Start

### Ollama

```bash
ollama create c64-ministral-c64 -f Modelfile.Q4_K_M
ollama create c64-ministral-c64-q6 -f Modelfile.Q6_K
ollama create c64-ministral-c64-q8 -f Modelfile.Q8_0
```

### llama.cpp

```bash
llama-cli -m c64-ministral-3-8b-thinking-c64-Q6_K.gguf -ngl 99 -c 4096 -n 256 -p "Explain VIC-II timing."
```

## Reference Throughput (this workstation)

Measured with `llama-bench` on ROCm (single run, `pp256` and `tg64`):

| Quant | pp256 (tok/s) | tg64 (tok/s) |
| --- | ---: | ---: |
| Q4_K_M | 1080.50 | 33.52 |
| Q6_K | 820.06 | 26.31 |
| Q8_0 | 404.59 | 21.20 |
| F16 | 546.18 | 10.68 |

Numbers are hardware/runtime dependent and should be treated as reference only.
"""


def prepare_lora_stage(stage_dir: Path, repo_id: str) -> None:
    """Build a temporary publish folder for LoRA artifacts."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "training_args.bin",
    ]:
        src = LORA_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected LoRA artifact: {src}")
        hardlink_or_copy(src, stage_dir / name)

    # Include trainer states for reproducibility.
    sft_state = LORA_DIR / "checkpoint-132" / "trainer_state.json"
    dapt_state = PROJECT_ROOT / "models" / "fine-tuned-dapt" / "checkpoint-39" / "trainer_state.json"
    if sft_state.exists():
        hardlink_or_copy(sft_state, stage_dir / "trainer_state_sft.json")
    if dapt_state.exists():
        hardlink_or_copy(dapt_state, stage_dir / "trainer_state_dapt.json")

    summary = load_training_summary()
    (stage_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (stage_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    context_length = load_context_length()
    (stage_dir / "README.md").write_text(
        render_lora_card(context_length=context_length, summary=summary, repo_id=repo_id),
        encoding="utf-8",
    )


def prepare_gguf_stage(stage_dir: Path, repo_id: str) -> None:
    """Build a temporary publish folder for GGUF artifacts."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "c64-ministral-3-8b-thinking-c64-F16.gguf",
        "c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf",
        "c64-ministral-3-8b-thinking-c64-Q6_K.gguf",
        "c64-ministral-3-8b-thinking-c64-Q8_0.gguf",
        "Modelfile",
        "Modelfile.F16",
        "Modelfile.Q4_K_M",
        "Modelfile.Q6_K",
        "Modelfile.Q8_0",
    ]:
        src = GGUF_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected GGUF artifact: {src}")
        hardlink_or_copy(src, stage_dir / name)

    (stage_dir / ".gitattributes").write_text(
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    context_length = load_context_length()
    (stage_dir / "README.md").write_text(
        render_gguf_card(context_length=context_length, repo_id=repo_id),
        encoding="utf-8",
    )


def upload_repo(api: HfApi, repo_id: str, stage_dir: Path, private: bool, dry_run: bool) -> None:
    """Upload a prepared folder into a Hugging Face model repository."""
    print(f"\n==> Preparing upload for {repo_id}")
    if dry_run:
        print(f"[dry-run] Would create repo: {repo_id} (private={private})")
        print(f"[dry-run] Would upload folder: {stage_dir}")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        commit_message=f"Publish fine-tuning artifacts ({datetime.now(UTC).isoformat()})",
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")


def upload_gguf_repo(api: HfApi, repo_id: str, private: bool, dry_run: bool) -> None:
    """Upload GGUF artifacts incrementally to reduce temporary disk pressure."""
    print(f"\n==> Preparing upload for {repo_id}")
    if dry_run:
        print(f"[dry-run] Would create repo: {repo_id} (private={private})")
        print(f"[dry-run] Would upload README + Modelfiles + GGUF files from {GGUF_DIR}")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    context_length = load_context_length()
    readme = render_gguf_card(context_length=context_length, repo_id=repo_id)
    with tempfile.TemporaryDirectory(prefix="hf_publish_gguf_meta_") as tmp:
        tmp_root = Path(tmp)
        readme_path = tmp_root / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        gitattributes_path = tmp_root / ".gitattributes"
        gitattributes_path.write_text("*.gguf filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card",
        )
        api.upload_file(
            path_or_fileobj=str(gitattributes_path),
            path_in_repo=".gitattributes",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Ensure GGUF tracked with LFS",
        )

    for name in [
        "Modelfile",
        "Modelfile.F16",
        "Modelfile.Q4_K_M",
        "Modelfile.Q6_K",
        "Modelfile.Q8_0",
        "c64-ministral-3-8b-thinking-c64-F16.gguf",
        "c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf",
        "c64-ministral-3-8b-thinking-c64-Q6_K.gguf",
        "c64-ministral-3-8b-thinking-c64-Q8_0.gguf",
    ]:
        src = GGUF_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected GGUF artifact: {src}")
        print(f"Uploading {name} ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {name}",
        )

    print(f"Uploaded: https://huggingface.co/{repo_id}")


def main() -> None:
    """Publish LoRA and GGUF repositories to Hugging Face Hub."""
    args = parse_args()
    token = require_token(args.token_env)
    api = HfApi(token=token)

    # Fast fail on read-only tokens.
    whoami = api.whoami()
    role = (
        whoami.get("auth", {})
        .get("accessToken", {})
        .get("role", "")
        .strip()
        .lower()
    )
    if not args.dry_run and role and role == "read":
        raise PermissionError(
            "HF token is read-only. Use a write token to create/upload model repos."
        )

    with tempfile.TemporaryDirectory(prefix="hf_publish_") as tmp:
        tmp_root = Path(tmp)
        if not args.skip_lora:
            lora_stage = tmp_root / "lora"
            prepare_lora_stage(lora_stage, repo_id=args.lora_repo_id)
            upload_repo(api, args.lora_repo_id, lora_stage, private=args.private, dry_run=args.dry_run)
        if not args.skip_gguf:
            upload_gguf_repo(api, args.gguf_repo_id, private=args.private, dry_run=args.dry_run)

    print("\nDone.")
    if not args.skip_lora:
        print(f"LoRA repo: https://huggingface.co/{args.lora_repo_id}")
    if not args.skip_gguf:
        print(f"GGUF repo: https://huggingface.co/{args.gguf_repo_id}")


if __name__ == "__main__":
    main()
