#!/usr/bin/env python3
"""
Shared prompt and reasoning contract for training and runtime.

This module is the single source of truth for:
- base model reasoning system prompt loading,
- C64 specialization append block,
- deterministic reasoning-trace diversity templates.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_BASE_MODEL_PATH = Path("models/Ministral-3-8B-Thinking")
BASE_SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.txt"

C64_APPEND_PROMPT = """# C64 SPECIALIZATION

You are a specialized Commodore 64 technical assistant.

Scope:
- Only answer Commodore 64 and directly related topics: C64 hardware specs, memory map, VIC-II, SID, CIA, KERNAL, BASIC V2, 6502/6510 machine language, programming, debugging, and emulation.

Behavior:
- Be concise, precise, and polite.
- Give enough detail to be useful; avoid one-word answers.
- If a request is outside scope, say it briefly and ask for a C64-focused question.
- If information is uncertain, state uncertainty and avoid guessing.
- Respond in the same language as the user.

Reasoning output contract:
- Always emit reasoning as a visible [THINK]...[/THINK] block.
- Keep [THINK] concise and technical (typically 1-4 sentences).
- After [/THINK], provide the final answer clearly and directly.
"""

REASONING_TRACE_TEMPLATES: tuple[str, ...] = (
    "I isolate the relevant C64 details and map them to the requested outcome: {task}.",
    "I check C64-specific constraints first, then structure the answer for: {task}.",
    "I identify concrete technical facts from the excerpt before concluding for: {task}.",
    "I prioritize practical C64 implementation details and filter noise for: {task}.",
    "I align the response with C64 hardware/software semantics to solve: {task}.",
    "I infer only from explicit C64 evidence and keep assumptions minimal for: {task}.",
    "I transform the source details into actionable C64 guidance for: {task}.",
    "I extract register/memory-relevant points where present to answer: {task}.",
    "I reconcile terminology and context, then deliver a precise answer for: {task}.",
    "I first validate scope and certainty, then provide the final C64 answer for: {task}.",
)

FALLBACK_BASE_SYSTEM_PROMPT = (
    "# HOW YOU SHOULD THINK AND ANSWER\n\n"
    "First draft your thinking process (inner monologue) until you arrive at a response. "
    "Format your response using Markdown, and use LaTeX for any mathematical equations. "
    "Write both your thoughts and the response in the same language as the input.\n\n"
    "Your thinking process must follow the template below:"
    "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
    "Be as casual and as long as you want until you are confident to generate the response to the user."
    "[/THINK]Here, provide a self-contained response."
)


def load_base_system_prompt(base_model_path: Path = DEFAULT_BASE_MODEL_PATH) -> str:
    """Load the official base-model system prompt text from disk."""
    path = (base_model_path / BASE_SYSTEM_PROMPT_FILE).resolve()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return FALLBACK_BASE_SYSTEM_PROMPT


def build_c64_system_prompt(base_model_path: Path = DEFAULT_BASE_MODEL_PATH) -> str:
    """Return official base prompt plus C64 specialization append block."""
    base = load_base_system_prompt(base_model_path).strip()
    return f"{base}\n\n{C64_APPEND_PROMPT.strip()}"


def choose_reasoning_trace(seed_key: str, task: str) -> str:
    """Choose a deterministic reasoning trace template from a stable key."""
    digest = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(REASONING_TRACE_TEMPLATES)
    return REASONING_TRACE_TEMPLATES[idx].format(task=task.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt contract helper.")
    parser.add_argument("--base-model-path", default=str(DEFAULT_BASE_MODEL_PATH))
    parser.add_argument("--print-base", action="store_true")
    parser.add_argument("--print-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_model_path = Path(args.base_model_path)
    if args.print_base:
        print(load_base_system_prompt(base_model_path))
        return
    if args.print_full:
        print(build_c64_system_prompt(base_model_path))
        return
    raise SystemExit("Use one of: --print-base or --print-full")


if __name__ == "__main__":
    main()

