# Reasoning Contract (llama.cpp / Ollama)

## Purpose

Define a single, testable contract for visible reasoning output so training, export, and runtime remain aligned.

## Output Contract

Assistant outputs must follow this shape:

1. A visible reasoning block: `[THINK]...[/THINK]`
2. A final answer after `[/THINK]` (non-empty)

Constraints:

- `[THINK]` and `[/THINK]` must be balanced.
- THINK content must be concise technical reasoning.
- Final answer must be separated from THINK content.
- Contract applies to single-turn and multi-turn interactions.

## Prompt Contract

- Base model official system prompt is loaded from:
  - `models/Ministral-3-8B-Thinking/SYSTEM_PROMPT.txt`
- Project specialization is appended (not replacing base prompt).
- Shared source of truth:
  - `scripts/prompt_contract.py`

## Training Contract

- SFT must run with `assistant_only_loss=True`.
- Chat template must include generation mask blocks:
  - `{% generation %}`
  - `{% endgeneration %}`
- Project template:
  - `scripts/templates/mistral3_chat_template_assistant_mask.jinja`
- Training fails if strict mode is enabled and generation blocks are missing.

## Data Contract

SFT dataset requirements:

- 99.5%+ assistant messages with valid THINK tags.
- Minimum THINK trace diversity (`unique_think_texts >= 8`).
- Multi-turn examples included (`multi_turn_ratio >= 0.15`).

These checks are enforced in:

- `scripts/data_pipeline.py` validation stage
- `data/processed/validation_report.json`
- `docs/data_qc_report.md` (generated report)

## Runtime Contract

For local runtime (`llama.cpp` / `Ollama`):

- Prefer `--reasoning-format none` to keep raw `[THINK]...[/THINK]` text visible.
- Use deterministic settings for contract checks (fixed seed, low temperature).

Validation script:

- `scripts/inference/validate_reasoning_behavior.sh`
- Emits:
  - `metrics.csv`
  - `summary.md`
  - raw per-run logs

Thresholds (default):

- `single_think_tag_rate >= 0.95`
- `single_balanced_tag_rate >= 0.95`
- `single_final_after_think_rate >= 0.95`
- `multi_turn_retention_rate >= 0.90`
