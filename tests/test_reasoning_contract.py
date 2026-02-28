"""Tests for prompt and chat-template reasoning contract."""

from pathlib import Path

from scripts.prompt_contract import build_c64_system_prompt, choose_reasoning_trace


def test_build_c64_system_prompt_appends_specialization(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "SYSTEM_PROMPT.txt").write_text("BASE PROMPT", encoding="utf-8")
    out = build_c64_system_prompt(model_dir)
    assert out.startswith("BASE PROMPT")
    assert "C64 SPECIALIZATION" in out
    assert "[THINK]" in out


def test_choose_reasoning_trace_is_deterministic():
    a = choose_reasoning_trace("id-1", task="task-x")
    b = choose_reasoning_trace("id-1", task="task-x")
    c = choose_reasoning_trace("id-2", task="task-x")
    assert a == b
    assert isinstance(c, str)
    assert len(c) > 10


def test_chat_template_contains_generation_mask_blocks():
    path = Path("scripts/templates/mistral3_chat_template_assistant_mask.jinja")
    text = path.read_text(encoding="utf-8")
    assert "{% generation %}" in text
    assert "{% endgeneration %}" in text
