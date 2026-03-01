"""Tests for multi-profile model registry and defaults."""

import pytest

from scripts.model_profiles import DEFAULT_PROFILE_KEY, available_profiles, get_model_profile


def test_registry_exposes_8b_and_14b():
    assert available_profiles() == ("14b", "8b")
    assert DEFAULT_PROFILE_KEY == "8b"


def test_profile_8b_defaults_are_backward_compatible():
    profile = get_model_profile("8b")
    assert str(profile.base_model_path) == "models/Ministral-3-8B-Thinking"
    assert str(profile.sft_output_dir) == "models/fine-tuned"
    assert str(profile.gguf_dir) == "models/gguf"
    assert profile.gguf_prefix == "c64-ministral-3-8b-thinking-c64"


def test_profile_14b_uses_separate_artifact_paths_and_repos():
    profile = get_model_profile("14b")
    assert str(profile.base_model_path) == "models/Ministral-3-14B-Thinking"
    assert str(profile.sft_output_dir) == "models/fine-tuned-14b"
    assert str(profile.gguf_dir) == "models/gguf-14b"
    assert profile.gguf_prefix == "c64-ministral-3-14b-thinking-c64"
    assert profile.lora_repo_id.endswith("-14b-thinking-c64-reasoning-lora")
    assert profile.gguf_repo_id.endswith("-14b-thinking-c64-reasoning-gguf")


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        get_model_profile("invalid")
