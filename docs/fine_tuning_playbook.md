# Fine-Tuning Playbook

## Objective

Fine-tune Ministral 3 Reasoning (8B or 14B profile) on technical C64 data while preserving a stable visible reasoning format:

- `[THINK]...[/THINK]`
- Final answer after `[/THINK]`

## Recommended Recipe

- Phase order: DAPT -> SFT
- Precision: `bf16`
- LoRA: enabled
- Max length: `2048`
- Batch size: `2`
- Gradient accumulation: `16`
- Learning rate: `2e-5`
- Epochs: `3`
- `assistant_only_loss`: enabled
- `strict_assistant_only_loss`: enabled
- Chat template override: `scripts/templates/mistral3_chat_template_assistant_mask.jinja`

## Prompt and Template Contract

- The base model official prompt is loaded from the selected profile base path:
  - `models/Ministral-3-8B-Thinking/SYSTEM_PROMPT.txt`
  - `models/Ministral-3-14B-Thinking/SYSTEM_PROMPT.txt`
- Project C64 specialization is appended (not replacing base prompt).
- Shared contract source:
  - `scripts/prompt_contract.py`
  - `docs/specs/reasoning_contract.md`

SFT uses a custom chat template with generation mask blocks so `assistant_only_loss` is enforced correctly:

- `{% generation %}`
- `{% endgeneration %}`

## Commands

### DAPT only

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase dapt \
  --model-profile 8b \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --output-dir models/fine-tuned-dapt \
  --precision bf16 \
  --use-lora
```

### SFT only

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase sft \
  --model-profile 8b \
  --model-path models/Ministral-3-8B-Thinking \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --precision bf16 \
  --assistant-only-loss \
  --strict-assistant-only-loss \
  --chat-template-path scripts/templates/mistral3_chat_template_assistant_mask.jinja \
  --no-packing \
  --use-lora
```

14B equivalents: use `--model-profile 14b`, `--model-path models/Ministral-3-14B-Thinking`, and profile output paths (`models/fine-tuned-dapt-14b`, `models/fine-tuned-14b`).

### Full flow

```bash
docker compose run --rm trainer bash scripts/container/train.sh
```

14B default flow:

```bash
docker compose run --rm trainer bash scripts/container/train.sh --model-profile 14b
```

## Acceptance Criteria

- DAPT and SFT complete without runtime errors.
- Artifacts are written under `models/`.
- Training logs and checkpoints are present.
- Post-training tests pass.
- `assistant_only_loss` remains enabled in saved training args.
- Reasoning contract validation passes in runtime checks.

## Risk Signals

- Validation split empty.
- Loss divergence or NaNs.
- GPU backend instability.
- Low THINK coverage or low THINK diversity in `data/processed/validation_report.json`.
- Missing generation mask blocks in chat template.
