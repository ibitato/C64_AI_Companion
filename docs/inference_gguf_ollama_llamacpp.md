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
When `Q8_0` is available, `Modelfile` (the default alias) points to `Q8_0` for better instruction quality; `Q4_K_M` remains available as a lower-memory option.

## Register Models in Ollama

```bash
bash scripts/inference/create_ollama_models.sh
```

## Run with llama.cpp

```bash
bash scripts/inference/run_llama_cpp.sh Q8_0 "Explain VIC-II badlines in concise terms."
```

## Benchmark GGUF Variants (Reproducible)

```bash
bash scripts/inference/benchmark_gguf_matrix.sh
```

Notes:

- Runs container-first by default (`docker compose run --rm trainer ...`).
- Benchmarks `F16`, `Q4_K_M`, `Q6_K`, `Q8_0`.
- Writes a timestamped CSV to `results/benchmarks/` with:
  - Offloaded layers (`offload_layers/offload_total`)
  - llama.cpp performance (`eval_ms`, `tok_per_s`, `total_ms`)
  - sampled ROCm telemetry (`gpu_use_avg_pct`, `gpu_use_max_pct`, `vram_max_pct`, `power_max_w`)

Example with custom options:

```bash
bash scripts/inference/benchmark_gguf_matrix.sh \
  --models "Q4_K_M Q6_K Q8_0" \
  --n-predict 128 \
  --ctx-size 4096 \
  --prompt "Explain SID envelope generators in concise technical terms."
```

## Expected GGUF Files

- `models/gguf/c64-ministral-3-8b-thinking-c64-F16.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q6_K.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q8_0.gguf`
