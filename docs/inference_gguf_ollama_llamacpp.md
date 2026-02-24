# Inference Guide: GGUF, Ollama, and llama.cpp

## Purpose

Package fine-tuned outputs into GGUF and run inference with Ollama or llama.cpp.

## Export GGUF

```bash
docker compose run --rm trainer bash scripts/container/export_gguf.sh \
  --base-model-path models/Ministral-3-8B-Thinking \
  --adapter-path models/fine-tuned \
  --gguf-dir models/gguf \
  --quantization Q4_K_M
```

## Generate Additional Quantizations

```bash
bash scripts/inference/quantize_additional_gguf.sh
```

## Prepare Runtime Assets

```bash
bash scripts/inference/prepare_runtime_assets.sh
```

`prepare_runtime_assets.sh` writes `Modelfile*` files with a C64-specialist `SYSTEM` prompt so Ollama runs keep the same scope/behavior constraints used during training data construction.

## Register Models in Ollama

```bash
bash scripts/inference/create_ollama_models.sh
```

## Run with llama.cpp

```bash
bash scripts/inference/run_llama_cpp.sh Q8_0 "Explain VIC-II badlines in concise terms."
```

## Expected GGUF Files

- `models/gguf/c64-ministral-3-8b-thinking-c64-F16.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q6_K.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q8_0.gguf`
