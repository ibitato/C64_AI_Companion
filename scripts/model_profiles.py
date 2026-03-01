#!/usr/bin/env python3
"""Central model profile registry for 8B/14B pipeline variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    """Declarative settings for a supported base-model profile."""

    key: str
    base_model_path: Path
    base_model_id: str
    model_size_label: str
    sft_output_dir: Path
    dapt_output_dir: Path
    merged_output_dir: Path
    gguf_dir: Path
    gguf_prefix: str
    lora_repo_id: str
    gguf_repo_id: str
    collection_url: str | None
    ollama_base_name: str


PROFILES: dict[str, ModelProfile] = {
    "8b": ModelProfile(
        key="8b",
        base_model_path=Path("models/Ministral-3-8B-Thinking"),
        base_model_id="mistralai/Ministral-3-8B-Reasoning-2512",
        model_size_label="8B",
        sft_output_dir=Path("models/fine-tuned"),
        dapt_output_dir=Path("models/fine-tuned-dapt"),
        merged_output_dir=Path("models/fine-tuned-merged-hf"),
        gguf_dir=Path("models/gguf"),
        gguf_prefix="c64-ministral-3-8b-thinking-c64",
        lora_repo_id="ibitato/c64-ministral-3-8b-thinking-c64-reasoning-lora",
        gguf_repo_id="ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf",
        collection_url="https://huggingface.co/collections/ibitato/c64-ministral-3-8b-thinking-c64-reasoning-699d67350911049ec1a82e18",
        ollama_base_name="c64-ministral-c64",
    ),
    "14b": ModelProfile(
        key="14b",
        base_model_path=Path("models/Ministral-3-14B-Thinking"),
        base_model_id="mistralai/Ministral-3-14B-Reasoning-2512",
        model_size_label="14B",
        sft_output_dir=Path("models/fine-tuned-14b"),
        dapt_output_dir=Path("models/fine-tuned-dapt-14b"),
        merged_output_dir=Path("models/fine-tuned-merged-hf-14b"),
        gguf_dir=Path("models/gguf-14b"),
        gguf_prefix="c64-ministral-3-14b-thinking-c64",
        lora_repo_id="ibitato/c64-ministral-3-14b-thinking-c64-reasoning-lora",
        gguf_repo_id="ibitato/c64-ministral-3-14b-thinking-c64-reasoning-gguf",
        collection_url=None,
        ollama_base_name="c64-ministral-c64-14b",
    ),
}


DEFAULT_PROFILE_KEY = "8b"


def available_profiles() -> tuple[str, ...]:
    """Return profile keys sorted for stable CLI choices."""
    return tuple(sorted(PROFILES.keys()))


def get_model_profile(profile_key: str) -> ModelProfile:
    """Resolve model profile by key, raising ValueError on unknown keys."""
    key = (profile_key or "").strip().lower()
    if key in PROFILES:
        return PROFILES[key]
    raise ValueError(f"Unsupported model profile '{profile_key}'. Allowed: {', '.join(available_profiles())}")

