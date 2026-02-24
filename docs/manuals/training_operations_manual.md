# Training Operations Manual

## Purpose

Operational checklist for reliable DAPT+SFT training runs.

## Preconditions

- Base model exists at `models/Ministral-3-8B-Thinking`.
- Processed datasets exist under `data/processed/`.
- Container image built successfully.

## Procedure

1. Build image.
2. Run GPU smoke test.
3. Run data pipeline.
4. Launch training.
5. Validate checkpoints and logs.

## Validation Checkpoints

- GPU smoke test passes.
- DAPT and SFT steps advance without crashes.
- Output model and checkpoints exist under `models/`.

## Failure Modes

- OOM
- Data schema mismatch
- Runtime backend mismatch

## Recovery

- Reduce memory pressure (`--batch-size`, `--max-length`).
- Re-run pipeline and verify split files.
- Revalidate container runtime stack.
