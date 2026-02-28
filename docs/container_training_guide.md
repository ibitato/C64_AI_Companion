# Container Training Guide

## Purpose

Run the complete C64 fine-tuning workflow in a reproducible ROCm container.

## Preconditions

1. Host requirements from `system_requirements.md` are satisfied.
2. Base model exists at `models/Ministral-3-8B-Thinking`.
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

### 4) Run training (DAPT + SFT)

```bash
docker compose run --rm trainer bash scripts/container/train.sh
```

### 5) Validate reasoning behavior on exported runtime artifacts

```bash
docker compose run --rm trainer bash scripts/inference/validate_reasoning_behavior.sh --in-container
```

### 6) Run tests

```bash
docker compose run --rm trainer pytest -q
```

## Parameterized Training Example

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
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
- Reasoning validation report is written under `results/reasoning_validation/`.

## Failure Modes

- `torch.cuda.is_available() == False`
  - Verify `/dev/kfd`, `/dev/dri`, and user groups.
- Base model policy failure
  - Ensure `--model-path` equals `models/Ministral-3-8B-Thinking`.
- OOM or unstable training
  - Reduce `--batch-size`, increase `--grad-accum`, reduce `--max-length`.
