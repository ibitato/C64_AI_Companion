# AI Agent Operating Guidelines for C64 AI Companion

## Purpose

These guidelines define how AI agents should operate in this repository to keep outputs reproducible, auditable, and technically correct.

## 1. Core Model Policy (Hard Requirement)

- The only valid base model path is `models/Ministral-3-8B-Thinking`.
- Do not use user-global model cache paths as training base paths.
- Fine-tuned outputs must remain under `models/`.

## 2. Runtime Strategy

- This project is container-first for training and packaging.
- Standard runtime image: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`.
- Host exposes `/dev/kfd` and `/dev/dri`; container runtime is the reproducible baseline.

## 3. Cache Policy

- Use project-local Hugging Face cache: `.cache/huggingface/`.
- Avoid hidden dependencies on user-global cache state.

## 4. Official Workflow

1. `docker compose build trainer`
2. `docker compose run --rm trainer bash scripts/container/gpu_smoke.sh`
3. `docker compose run --rm trainer bash scripts/container/pipeline.sh`
4. `docker compose run --rm trainer bash scripts/container/train.sh`
5. `docker compose run --rm trainer pytest -q`

## 5. Documentation Policy

- All maintained documentation must be English.
- Update docs in the same change set as behavior changes.
- Keep docs aligned with actual scripts and commands in this repo.

## 6. Security and Hygiene

- Never commit secrets.
- Keep `.env` local and ignored.
- Do not commit heavy artifacts, model weights, or local caches.
- Keep `.gitignore` aligned with real repository behavior.

## 7. Reproducibility

- Dependencies are defined in:
  - `requirements.base.txt`
  - `requirements.rocm72.txt`
  - `requirements.txt`
- Prefer deterministic, scripted steps over manual, ad hoc operations.

## 8. References

- ROCm docs: https://rocm.docs.amd.com/
- Transformers docs: https://huggingface.co/docs/transformers/index
- llama.cpp: https://github.com/ggml-org/llama.cpp

## 9. Attribution Context

AI-assisted work is acknowledged in `CREDITS.md`.
