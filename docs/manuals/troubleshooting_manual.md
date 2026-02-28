# Troubleshooting Manual

## Purpose

Centralized troubleshooting procedures for runtime, training, and packaging issues.

## 1) GPU Not Available in Container

Symptoms:

- `torch.cuda.is_available() == False`

Checks:

```bash
id
ls -l /dev/kfd /dev/dri/renderD128
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```

## 2) ROCm/HIP Kernel Mismatch Symptoms

Symptoms:

- `no kernel image is available for execution on the device`
- `hipErrorInvalidDeviceFunction`

Actions:

1. Confirm canonical container runtime is used.
2. Rebuild container image.
3. Validate torch/HIP runtime from inside container.

## 3) Data Pipeline Errors

Symptoms:

- Missing DAPT/SFT split files.

Actions:

1. Re-run pipeline.
2. Check source docs in `c64_docs/`.
3. Validate `validation_report.json`.

## 4) GGUF Export and Quantization Failures

Symptoms:

- Missing conversion dependency.
- Quantization command fails.

Actions:

1. Ensure `sentencepiece` is installed in runtime.
2. Rebuild/validate `llama.cpp` toolchain.
3. Re-run export from clean merged model output.

## 5) Thinking Is Intermittent or Missing in GUI

Symptoms:

- `[THINK]...[/THINK]` appears in some turns but not others.
- Reasoning panel appears for 1-2 interactions and then disappears.

Checks:

```bash
bash scripts/inference/validate_reasoning_behavior.sh
```

Actions:

1. Confirm runtime uses contract-preserving mode (`--reasoning-format none`).
2. Rebuild SFT data and verify `validation_report.json` THINK metrics.
3. Confirm training used generation-mask template and `assistant_only_loss=True`.
4. Re-export GGUF and regenerate Modelfiles from `prepare_runtime_assets.sh`.
