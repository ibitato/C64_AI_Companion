# Inference Guide: GGUF, Ollama, and llama.cpp

## Purpose

Package fine-tuned outputs into GGUF and run inference with Ollama or llama.cpp.

## Export GGUF

```bash
docker compose run --rm trainer bash scripts/container/export_gguf.sh \
  --model-profile 8b \
  --quantization Q4_K_M
```

## Generate Additional Quantizations

```bash
bash scripts/inference/quantize_additional_gguf.sh --model-profile 8b
```

## Prepare Runtime Assets

```bash
bash scripts/inference/prepare_runtime_assets.sh --model-profile 8b
```

`prepare_runtime_assets.sh` writes `Modelfile*` files with a C64-specialist `SYSTEM` prompt so Ollama runs keep the same scope/behavior constraints used during training data construction.
When `Q8_0` is available, `Modelfile` (the default alias) points to `Q8_0` for better instruction quality; `Q4_K_M` remains available as a lower-memory option.

## Register Models in Ollama

```bash
bash scripts/inference/create_ollama_models.sh --model-profile 8b
```

## Run with llama.cpp

```bash
bash scripts/inference/run_llama_cpp.sh Q8_0 "Explain VIC-II badlines in concise terms." --model-profile 8b
```

Notes:

- The wrapper defaults to chat-template mode (`-cnv --jinja`) so Ministral prompt/template logic is applied.
- The wrapper enables special-token printing (`-sp`) so `[THINK]...[/THINK]` delimiters stay visible.
- The wrapper auto-injects the project contract system prompt (`scripts/prompt_contract.py --print-full`) unless overridden with `-sys/-sysf` or `LLAMA_USE_CONTRACT_PROMPT=0`.
- The wrapper defaults to `--reasoning-format none` to keep raw `[THINK]...[/THINK]` output visible.
- Use `--multi-turn` to disable the single-turn shortcut (`-st`) and keep interactive context.

Example:

```bash
bash scripts/inference/run_llama_cpp.sh Q8_0 "Explain SID ADSR in brief." --model-profile 8b --multi-turn --simple-io
```

## Run llama-server (OpenAI-compatible API / GUI reasoning panel)

```bash
python3 scripts/prompt_contract.py --model-profile 8b --print-full > .cache/runtime/c64_system_prompt_8b.txt
llama-server \
  -hf ibitato/c64-ministral-3-8b-thinking-c64-reasoning-gguf:F16 \
  --host 0.0.0.0 --port 8080 \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --system-prompt-file .cache/runtime/c64_system_prompt_8b.txt \
  --ctx-size 32768 \
  -ngl 99 \
  --temp 0.15 \
  --threads "$(nproc)" \
  --fit on
```

Use `--reasoning-format none` when you want raw `[THINK]...[/THINK]` tags inside `content` instead of separated reasoning in GUI/OpenAI-compatible responses.

14B variant:

```bash
python3 scripts/prompt_contract.py --model-profile 14b --print-full > .cache/runtime/c64_system_prompt_14b.txt
llama-server \
  -hf ibitato/c64-ministral-3-14b-thinking-c64-reasoning-gguf:F16 \
  --host 0.0.0.0 --port 8080 \
  --jinja \
  --reasoning-format deepseek \
  --reasoning-budget -1 \
  --system-prompt-file .cache/runtime/c64_system_prompt_14b.txt \
  --ctx-size 32768 \
  -ngl 99 \
  --temp 0.15 \
  --threads "$(nproc)" \
  --fit on
```

## Validate Reasoning Contract (Reproducible)

```bash
bash scripts/inference/validate_reasoning_behavior.sh --model-profile 8b
```

Validation now forces `-cnv --jinja -sp`, injects the contract system prompt, and applies a per-run timeout to avoid stuck multi-turn sessions in container runs.

Outputs:

- `results/reasoning_validation/<profile>/<timestamp>/metrics.csv`
- `results/reasoning_validation/<profile>/<timestamp>/summary.md`
- `results/reasoning_validation/<profile>/<timestamp>/raw/*.log`

## Benchmark GGUF Variants (Reproducible)

```bash
bash scripts/inference/benchmark_gguf_matrix.sh --model-profile 8b
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

8B profile:

- `models/gguf/c64-ministral-3-8b-thinking-c64-F16.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q6_K.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q8_0.gguf`

14B profile:

- `models/gguf-14b/c64-ministral-3-14b-thinking-c64-F16.gguf`
- `models/gguf-14b/c64-ministral-3-14b-thinking-c64-Q4_K_M.gguf`
- `models/gguf-14b/c64-ministral-3-14b-thinking-c64-Q6_K.gguf`
- `models/gguf-14b/c64-ministral-3-14b-thinking-c64-Q8_0.gguf`
