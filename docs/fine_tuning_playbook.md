# Fine-Tuning Playbook

## Objective

Fine-tune Ministral 3 8B Reasoning on technical C64 data while preserving a stable visible reasoning format:

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

- The base model official prompt is loaded from `models/Ministral-3-8B-Thinking/SYSTEM_PROMPT.txt`.
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
  --model-path models/Ministral-3-8B-Thinking \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned-sft \
  --precision bf16 \
  --assistant-only-loss \
  --strict-assistant-only-loss \
  --chat-template-path scripts/templates/mistral3_chat_template_assistant_mask.jinja \
  --no-packing \
  --use-lora
```

### Full flow

```bash
docker compose run --rm trainer bash scripts/container/train.sh
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
