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
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/ibitato/C64_AI_Companion"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import torch
from huggingface_hub import HfApi
from model_profiles import DEFAULT_PROFILE_KEY, ModelProfile, available_profiles, get_model_profile

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def profile_lora_dir(profile: ModelProfile) -> Path:
    return PROJECT_ROOT / profile.sft_output_dir


def profile_dapt_dir(profile: ModelProfile) -> Path:
    return PROJECT_ROOT / profile.dapt_output_dir


def profile_gguf_dir(profile: ModelProfile) -> Path:
    return PROJECT_ROOT / profile.gguf_dir


def profile_base_config(profile: ModelProfile) -> Path:
    return PROJECT_ROOT / profile.base_model_path / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish LoRA + GGUF artifacts to Hugging Face.")
    parser.add_argument("--model-profile", choices=available_profiles(), default=DEFAULT_PROFILE_KEY)
    parser.add_argument("--lora-repo-id", default=None)
    parser.add_argument("--gguf-repo-id", default=None)
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


def load_context_length(profile: ModelProfile) -> int:
    """Load max context length from the local base-model config."""
    with profile_base_config(profile).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["text_config"]["max_position_embeddings"])


def latest_checkpoint_dir(model_dir: Path) -> Path | None:
    """Return latest numeric checkpoint dir (checkpoint-N) if available."""
    candidates: list[tuple[int, float, Path]] = []
    for p in model_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        match = CHECKPOINT_RE.fullmatch(p.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), p.stat().st_mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def load_json_file(path: Path) -> dict[str, object]:
    """Load a JSON dictionary from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl_rows(path: Path) -> int:
    """Count rows in a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_parquet_rows(path: Path) -> int:
    """Count rows in a parquet file via metadata."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet row counts for training summary publication."
        ) from exc
    return int(pq.ParquetFile(path).metadata.num_rows)


def load_data_split_summary() -> dict[str, int]:
    """Collect DAPT and SFT split sizes from project-local processed datasets."""
    dapt_train = PROCESSED_DIR / "dapt" / "train.parquet"
    dapt_val = PROCESSED_DIR / "dapt" / "validation.parquet"
    dapt_test = PROCESSED_DIR / "dapt" / "test.parquet"
    sft_train = PROCESSED_DIR / "sft" / "train.jsonl"
    sft_val = PROCESSED_DIR / "sft" / "validation.jsonl"
    sft_test = PROCESSED_DIR / "sft" / "test.jsonl"
    required = [dapt_train, dapt_val, dapt_test, sft_train, sft_val, sft_test]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed dataset files: {missing}")
    return {
        "dapt_train": count_parquet_rows(dapt_train),
        "dapt_validation": count_parquet_rows(dapt_val),
        "dapt_test": count_parquet_rows(dapt_test),
        "sft_train": count_jsonl_rows(sft_train),
        "sft_validation": count_jsonl_rows(sft_val),
        "sft_test": count_jsonl_rows(sft_test),
    }


def normalize_enum(value: object) -> object:
    """Normalize enum-like values into plain JSON-safe scalars."""
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


def guess_lora_scope(target_modules: object) -> str:
    """Infer a human-readable LoRA scope from adapter target modules."""
    if not isinstance(target_modules, list) or not target_modules:
        return "unknown"
    suffixes = (".q_proj", ".k_proj", ".v_proj", ".o_proj")
    if all(isinstance(m, str) and m.endswith(suffixes) for m in target_modules):
        return "language_qkvo"
    return "language_all_linear"


def load_training_recipe(profile: ModelProfile) -> dict[str, object]:
    """Collect key training recipe values from local artifacts."""
    lora_dir = profile_lora_dir(profile)
    args_path = lora_dir / "training_args.bin"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing expected training arguments artifact: {args_path}")
    args = torch.load(str(args_path), map_location="cpu", weights_only=False)

    adapter_config_path = lora_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing expected adapter config: {adapter_config_path}")
    adapter_cfg = load_json_file(adapter_config_path)

    return {
        "pipeline": "DAPT + SFT (LoRA)",
        "precision": "bf16" if bool(getattr(args, "bf16", False)) else ("fp16" if bool(getattr(args, "fp16", False)) else "fp32"),
        "max_length": int(getattr(args, "max_length", 2048)),
        "batch_size_per_device": int(getattr(args, "per_device_train_batch_size", 0)),
        "gradient_accumulation": int(getattr(args, "gradient_accumulation_steps", 0)),
        "learning_rate": float(getattr(args, "learning_rate", 0.0)),
        "epochs": float(getattr(args, "num_train_epochs", 0.0)),
        "warmup_steps": int(getattr(args, "warmup_steps", 0)),
        "logging_steps": int(getattr(args, "logging_steps", 0)),
        "save_steps": int(getattr(args, "save_steps", 0)),
        "eval_steps": int(getattr(args, "eval_steps", 0)),
        "optim": normalize_enum(getattr(args, "optim", None)),
        "seed": int(getattr(args, "seed", 0)),
        "eval_strategy": normalize_enum(getattr(args, "eval_strategy", None)),
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
        "assistant_only_loss": bool(getattr(args, "assistant_only_loss", False)),
        "packing": bool(getattr(args, "packing", False)),
        "lora_r": int(adapter_cfg.get("r", 0)),
        "lora_alpha": int(adapter_cfg.get("lora_alpha", 0)),
        "lora_dropout": float(adapter_cfg.get("lora_dropout", 0.0)),
        "lora_scope": guess_lora_scope(adapter_cfg.get("target_modules")),
    }


def load_git_revision() -> str | None:
    """Return current git revision (short SHA) when available."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except Exception:
        return None


def load_training_summary(profile: ModelProfile) -> dict[str, object]:
    """Collect compact metadata from local training outputs for model cards."""
    lora_dir = profile_lora_dir(profile)
    dapt_dir = profile_dapt_dir(profile)
    sft_ckpt = latest_checkpoint_dir(lora_dir)
    dapt_ckpt = latest_checkpoint_dir(dapt_dir)
    if sft_ckpt is None:
        raise FileNotFoundError(f"No SFT checkpoint found in {lora_dir}")
    if dapt_ckpt is None:
        raise FileNotFoundError(f"No DAPT checkpoint found in {dapt_dir}")

    sft_state_path = sft_ckpt / "trainer_state.json"
    dapt_state_path = dapt_ckpt / "trainer_state.json"
    if not sft_state_path.exists():
        raise FileNotFoundError(f"Missing expected SFT trainer state: {sft_state_path}")
    if not dapt_state_path.exists():
        raise FileNotFoundError(f"Missing expected DAPT trainer state: {dapt_state_path}")

    sft_state = load_json_file(sft_state_path)
    dapt_state = load_json_file(dapt_state_path)
    sft_log_history = sft_state.get("log_history", [])
    sft_last_log = sft_log_history[-1] if isinstance(sft_log_history, list) and sft_log_history else {}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": load_git_revision(),
        "data_splits": load_data_split_summary(),
        "training_recipe": load_training_recipe(profile),
        "artifacts": {
            "dapt_checkpoint": dapt_ckpt.name,
            "sft_checkpoint": sft_ckpt.name,
        },
        "run_state": {
            "dapt_global_step": dapt_state.get("global_step"),
            "dapt_max_steps": dapt_state.get("max_steps"),
            "sft_global_step": sft_state.get("global_step"),
            "sft_max_steps": sft_state.get("max_steps"),
            "sft_last_logged_step": sft_last_log.get("step"),
            "sft_last_logged_loss": sft_last_log.get("loss"),
            "sft_last_logged_token_accuracy": sft_last_log.get("mean_token_accuracy"),
        },
    }


def render_lora_card(
    *,
    context_length: int,
    summary: dict[str, object],
    repo_id: str,
    profile: ModelProfile,
    lora_repo_id: str,
    gguf_repo_id: str,
) -> str:
    """Render markdown model card content for the LoRA repository."""
    data = summary["data_splits"]
    recipe = summary["training_recipe"]
    run = summary["run_state"]
    artifacts = summary.get("artifacts", {})
    generated_at = summary.get("generated_at_utc", "unknown")
    git_rev = summary.get("git_revision", "unknown")

    collection_line = (
        f"- Collection: {profile.collection_url}\n"
        if profile.collection_url
        else ""
    )

    return f"""---
license: apache-2.0
base_model:
- {profile.base_model_id}
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

This repository contains the **LoRA adapter** produced by fine-tuning a reasoning-capable Ministral 3 {profile.model_size_label} model on technical Commodore 64 material.

Objective:
- keep the reasoning behavior of the base model,
- inject C64-specific technical knowledge,
- support practical troubleshooting and low-level explanations (BASIC, KERNAL, memory map, VIC-II, SID, 6502/6510).

Project source code and pipeline:
- {GITHUB_REPO_URL}

Related repositories:
- LoRA: https://huggingface.co/{lora_repo_id}
- GGUF: https://huggingface.co/{gguf_repo_id}
{collection_line}

## Base Model

- Base model: `{profile.base_model_id}`
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
- Warmup steps: {recipe["warmup_steps"]}
- Logging/save/eval steps: {recipe["logging_steps"]}/{recipe["save_steps"]}/{recipe["eval_steps"]}
- Optimizer: {recipe["optim"]}, eval strategy: {recipe["eval_strategy"]}
- Gradient checkpointing: {recipe["gradient_checkpointing"]}
- LoRA: r={recipe["lora_r"]}, alpha={recipe["lora_alpha"]}, dropout={recipe["lora_dropout"]}, scope={recipe["lora_scope"]}
- SFT options: assistant_only_loss={recipe["assistant_only_loss"]}, packing={recipe["packing"]}

Run summary:
- DAPT checkpoint: {artifacts.get("dapt_checkpoint")}
- SFT checkpoint: {artifacts.get("sft_checkpoint")}
- DAPT global steps: {run["dapt_global_step"]} / {run["dapt_max_steps"]}
- SFT global steps: {run["sft_global_step"]} / {run["sft_max_steps"]}
- Last logged SFT step: {run["sft_last_logged_step"]}
- Last logged SFT loss: {run["sft_last_logged_loss"]}
- Last logged SFT token accuracy: {run["sft_last_logged_token_accuracy"]}
- Card generated at (UTC): {generated_at}
- Source git revision: {git_rev}

## Usage (Transformers + PEFT)

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from transformers.models.mistral3 import Mistral3ForConditionalGeneration

base_id = "{profile.base_model_id}"
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


def render_gguf_card(
    *,
    context_length: int,
    summary: dict[str, object],
    repo_id: str,
    profile: ModelProfile,
    lora_repo_id: str,
    gguf_repo_id: str,
) -> str:
    """Render markdown model card content for the GGUF repository."""
    data = summary["data_splits"]
    run = summary["run_state"]
    artifacts = summary.get("artifacts", {})
    generated_at = summary.get("generated_at_utc", "unknown")
    git_rev = summary.get("git_revision", "unknown")
    gguf_dir = profile_gguf_dir(profile)
    files = [
        f"{profile.gguf_prefix}-F16.gguf",
        f"{profile.gguf_prefix}-Q4_K_M.gguf",
        f"{profile.gguf_prefix}-Q6_K.gguf",
        f"{profile.gguf_prefix}-Q8_0.gguf",
    ]
    sizes: dict[str, str] = {}
    for name in files:
        p = gguf_dir / name
        sizes[name] = f"{p.stat().st_size / (1024 ** 3):.2f} GiB" if p.exists() else "missing"

    collection_line = (
        f"- Collection: {profile.collection_url}\n"
        if profile.collection_url
        else ""
    )
    throughput_section = ""
    if profile.key == "8b":
        throughput_section = """
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
    else:
        throughput_section = """
## Reference Throughput

No workstation reference benchmarks are bundled yet for this profile.
"""

    return f"""---
license: apache-2.0
base_model:
- {lora_repo_id}
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

GGUF exports of the C64-focused Ministral 3 {profile.model_size_label} reasoning fine-tune, ready for **llama.cpp** and **Ollama**.

Project source code and training pipeline:
- {GITHUB_REPO_URL}

Related repositories:
- LoRA: https://huggingface.co/{lora_repo_id}
- GGUF: https://huggingface.co/{gguf_repo_id}
{collection_line}

## Technical Details

- Derived from: `{profile.base_model_id}` + project LoRA adaptation
- Context length in GGUF metadata: **{context_length:,} tokens**
- Architecture in GGUF: `mistral3`

## Training Provenance

- DAPT checkpoint used: {artifacts.get("dapt_checkpoint")}
- SFT checkpoint used: {artifacts.get("sft_checkpoint")}
- DAPT steps: {run["dapt_global_step"]} / {run["dapt_max_steps"]}
- SFT steps: {run["sft_global_step"]} / {run["sft_max_steps"]}
- Data splits: DAPT {data["dapt_train"]}/{data["dapt_validation"]}/{data["dapt_test"]}, SFT {data["sft_train"]}/{data["sft_validation"]}/{data["sft_test"]}
- Card generated at (UTC): {generated_at}
- Source git revision: {git_rev}

## Included Files

| File | Size |
| --- | --- |
| `{profile.gguf_prefix}-F16.gguf` | {sizes[f"{profile.gguf_prefix}-F16.gguf"]} |
| `{profile.gguf_prefix}-Q4_K_M.gguf` | {sizes[f"{profile.gguf_prefix}-Q4_K_M.gguf"]} |
| `{profile.gguf_prefix}-Q6_K.gguf` | {sizes[f"{profile.gguf_prefix}-Q6_K.gguf"]} |
| `{profile.gguf_prefix}-Q8_0.gguf` | {sizes[f"{profile.gguf_prefix}-Q8_0.gguf"]} |

`Modelfile` templates are included for direct Ollama import.

## Quick Start

### Ollama

```bash
ollama create {profile.ollama_base_name} -f Modelfile.Q4_K_M
ollama create {profile.ollama_base_name}-q6 -f Modelfile.Q6_K
ollama create {profile.ollama_base_name}-q8 -f Modelfile.Q8_0
```

### llama.cpp

```bash
llama-cli -m {profile.gguf_prefix}-Q6_K.gguf -ngl 99 -c 4096 -n 256 -p "Explain VIC-II timing."
```

### llama-server (OpenAI-compatible API / GUI reasoning panel)

```bash
python3 scripts/prompt_contract.py --model-profile {profile.key} --print-full > .cache/runtime/c64_system_prompt_{profile.key}.txt
llama-server \\
  -hf {gguf_repo_id}:F16 \\
  --host 0.0.0.0 --port 8080 \\
  --jinja \\
  --reasoning-format deepseek \\
  --reasoning-budget -1 \\
  --system-prompt-file .cache/runtime/c64_system_prompt_{profile.key}.txt \\
  --ctx-size 32768 \\
  -ngl 99 \\
  --temp 0.15 \\
  --threads "$(nproc)" \\
  --fit on
```

Use `--reasoning-format none` for raw `[THINK]...[/THINK]` tags in `content` instead of separated reasoning fields.
{throughput_section}
"""


def prepare_lora_stage(
    stage_dir: Path,
    repo_id: str,
    profile: ModelProfile,
    lora_repo_id: str,
    gguf_repo_id: str,
) -> None:
    """Build a temporary publish folder for LoRA artifacts."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = profile_lora_dir(profile)
    dapt_dir = profile_dapt_dir(profile)
    for name in [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "training_args.bin",
    ]:
        src = lora_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected LoRA artifact: {src}")
        hardlink_or_copy(src, stage_dir / name)

    # Include trainer states for reproducibility.
    sft_ckpt = latest_checkpoint_dir(lora_dir)
    dapt_ckpt = latest_checkpoint_dir(dapt_dir)
    sft_state = sft_ckpt / "trainer_state.json" if sft_ckpt else None
    dapt_state = dapt_ckpt / "trainer_state.json" if dapt_ckpt else None
    if sft_state and sft_state.exists():
        hardlink_or_copy(sft_state, stage_dir / "trainer_state_sft.json")
    if dapt_state and dapt_state.exists():
        hardlink_or_copy(dapt_state, stage_dir / "trainer_state_dapt.json")

    summary = load_training_summary(profile)
    (stage_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (stage_dir / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    context_length = load_context_length(profile)
    (stage_dir / "README.md").write_text(
        render_lora_card(
            context_length=context_length,
            summary=summary,
            repo_id=repo_id,
            profile=profile,
            lora_repo_id=lora_repo_id,
            gguf_repo_id=gguf_repo_id,
        ),
        encoding="utf-8",
    )


def prepare_gguf_stage(
    stage_dir: Path,
    repo_id: str,
    profile: ModelProfile,
    lora_repo_id: str,
    gguf_repo_id: str,
) -> None:
    """Build a temporary publish folder for GGUF artifacts."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    gguf_dir = profile_gguf_dir(profile)
    for name in [
        f"{profile.gguf_prefix}-F16.gguf",
        f"{profile.gguf_prefix}-Q4_K_M.gguf",
        f"{profile.gguf_prefix}-Q6_K.gguf",
        f"{profile.gguf_prefix}-Q8_0.gguf",
        "Modelfile",
        "Modelfile.F16",
        "Modelfile.Q4_K_M",
        "Modelfile.Q6_K",
        "Modelfile.Q8_0",
    ]:
        src = gguf_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected GGUF artifact: {src}")
        hardlink_or_copy(src, stage_dir / name)

    (stage_dir / ".gitattributes").write_text(
        "*.gguf filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    context_length = load_context_length(profile)
    summary = load_training_summary(profile)
    (stage_dir / "README.md").write_text(
        render_gguf_card(
            context_length=context_length,
            summary=summary,
            repo_id=repo_id,
            profile=profile,
            lora_repo_id=lora_repo_id,
            gguf_repo_id=gguf_repo_id,
        ),
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
        commit_message=f"Publish fine-tuning artifacts ({datetime.now(timezone.utc).isoformat()})",
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")


def upload_gguf_repo(
    api: HfApi,
    repo_id: str,
    private: bool,
    dry_run: bool,
    profile: ModelProfile,
    lora_repo_id: str,
    gguf_repo_id: str,
) -> None:
    """Upload GGUF artifacts incrementally to reduce temporary disk pressure."""
    gguf_dir = profile_gguf_dir(profile)
    print(f"\n==> Preparing upload for {repo_id}")
    if dry_run:
        print(f"[dry-run] Would create repo: {repo_id} (private={private})")
        print(f"[dry-run] Would upload README + Modelfiles + GGUF files from {gguf_dir}")
        return

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    context_length = load_context_length(profile)
    summary = load_training_summary(profile)
    readme = render_gguf_card(
        context_length=context_length,
        summary=summary,
        repo_id=repo_id,
        profile=profile,
        lora_repo_id=lora_repo_id,
        gguf_repo_id=gguf_repo_id,
    )
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
        f"{profile.gguf_prefix}-F16.gguf",
        f"{profile.gguf_prefix}-Q4_K_M.gguf",
        f"{profile.gguf_prefix}-Q6_K.gguf",
        f"{profile.gguf_prefix}-Q8_0.gguf",
    ]:
        src = gguf_dir / name
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
    profile = get_model_profile(args.model_profile)
    lora_repo_id = args.lora_repo_id or profile.lora_repo_id
    gguf_repo_id = args.gguf_repo_id or profile.gguf_repo_id

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
            prepare_lora_stage(
                lora_stage,
                repo_id=lora_repo_id,
                profile=profile,
                lora_repo_id=lora_repo_id,
                gguf_repo_id=gguf_repo_id,
            )
            upload_repo(api, lora_repo_id, lora_stage, private=args.private, dry_run=args.dry_run)
        if not args.skip_gguf:
            upload_gguf_repo(
                api,
                gguf_repo_id,
                private=args.private,
                dry_run=args.dry_run,
                profile=profile,
                lora_repo_id=lora_repo_id,
                gguf_repo_id=gguf_repo_id,
            )

    print("\nDone.")
    if not args.skip_lora:
        print(f"LoRA repo: https://huggingface.co/{lora_repo_id}")
    if not args.skip_gguf:
        print(f"GGUF repo: https://huggingface.co/{gguf_repo_id}")


if __name__ == "__main__":
    main()
