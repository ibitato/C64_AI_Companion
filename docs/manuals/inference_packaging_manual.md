# Inference Packaging Manual

## Purpose

Operational procedure to produce and validate deployable GGUF artifacts.

## Preconditions

- Trained adapter available in `models/fine-tuned`.
- `llama.cpp` tooling available in `.cache/llama.cpp` (script can bootstrap it).

## Procedure

1. Export merged HF + GGUF.
2. Generate additional quantizations.
3. Prepare Modelfiles.
4. Register Ollama models.
5. Run llama.cpp sanity prompts.
6. Run reasoning contract validation.

## Validation Checkpoints

- `F16`, `Q4_K_M`, `Q6_K`, `Q8_0` GGUF files present.
- Modelfiles present for each quantization.
- Ollama model creation succeeds.
- Reasoning validation passes with configured thresholds.

## Failure Modes and Recovery

- Missing `sentencepiece`: rebuild image with pinned requirements.
- Missing `llama-quantize`: build llama.cpp via script path.
- Quantization failure: verify F16 source file integrity.
- Thinking visible only intermittently: run `scripts/inference/validate_reasoning_behavior.sh` and inspect `summary.md`.
