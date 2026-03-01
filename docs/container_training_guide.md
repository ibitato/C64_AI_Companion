# Container Training Guide

## Purpose

Run the complete C64 fine-tuning workflow in a reproducible ROCm container.

## Preconditions

1. Host requirements from `system_requirements.md` are satisfied.
2. Base model exists for the selected profile:
   - `models/Ministral-3-8B-Thinking` (`8b`, default)
   - `models/Ministral-3-14B-Thinking` (`14b`)
3. UID/GID variables are exported.

## Procedure

### 1) Build container image

```bash
docker compose build trainer
```

### 2) Validate GPU runtime inside container

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```

### 3) Build datasets

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh
```

For 14B profile:

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh --model-profile 14b
```

### 4) Run training (DAPT + SFT)

```bash
docker compose run --rm trainer bash scripts/container/train.sh
```

For 14B profile:

```bash
docker compose run --rm trainer bash scripts/container/train.sh --model-profile 14b
```

### 5) Validate reasoning behavior on exported runtime artifacts

```bash
docker compose run --rm trainer bash scripts/inference/validate_reasoning_behavior.sh --in-container --model-profile 8b
```

For 14B profile:

```bash
docker compose run --rm trainer bash scripts/inference/validate_reasoning_behavior.sh --in-container --model-profile 14b
```

### 6) Run tests

```bash
docker compose run --rm trainer pytest -q
```

## Parameterized Training Example

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase both \
  --model-profile 14b \
  --model-path models/Ministral-3-14B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned-14b \
  --precision bf16 \
  --assistant-only-loss \
  --strict-assistant-only-loss \
  --chat-template-path scripts/templates/mistral3_chat_template_assistant_mask.jinja \
  --no-packing \
  --use-lora
```

## Validation Outputs

- DAPT outputs in `models/*-dapt` or configured output path.
- Final SFT output in target `models/*` path.
- Training checkpoints under the selected output directory.
- Data validation report includes SFT THINK coverage under `checks.sft_thinking`.
- Reasoning validation report is written under `results/reasoning_validation/<profile>/`.

## Failure Modes

- `torch.cuda.is_available() == False`
  - Verify `/dev/kfd`, `/dev/dri`, and user groups.
- Base model policy failure
  - Ensure `--model-path` matches the selected `--model-profile` canonical path.
- OOM or unstable training
  - Reduce `--batch-size`, increase `--grad-accum`, reduce `--max-length`.
