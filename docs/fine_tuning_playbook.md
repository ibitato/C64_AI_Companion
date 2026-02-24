# Fine-Tuning Playbook

## Objective

Fine-tune a reasoning-capable Ministral 3 8B base model on technical C64 data using a two-phase workflow.

## Recommended Recipe

- Phase order: DAPT -> SFT
- Precision: `bf16`
- LoRA: enabled
- Max length: `2048`
- Batch size: `2`
- Gradient accumulation: `16`
- Learning rate: `2e-5`
- Epochs: `3`

## System Prompt Used in SFT Data

The SFT generator now injects a constrained C64-specialist system prompt that:

- restricts scope to Commodore 64 and directly related topics,
- enforces concise and polite answers,
- requests explicit uncertainty when needed,
- asks for C64-focused reformulation when a request is out of scope,
- keeps responses in the user's language.

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
  --no-assistant-only-loss \
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

## Risk Signals

- Validation split empty.
- Loss divergence or NaNs.
- GPU backend instability.
