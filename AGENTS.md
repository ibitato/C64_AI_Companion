# Best Practices for Using AI Agents in C64_AI_Companion

## Objetivo
Este proyecto se centra en fine-tuning de `Ministral-3-8B-Thinking` para conocimiento de Commodore 64 con un flujo reproducible, trazable y mantenible.

## 1. Política obligatoria de modelos (hard requirement)

- El único directorio válido para **modelos originales/base** es `./models/`.
- La única ruta base permitida en este proyecto es:
  - `./models/Ministral-3-8B-Thinking`
- Está prohibido operar con rutas de modelo base fuera del repo (por ejemplo, `~/.cache/huggingface/...`) como ruta de trabajo de entrenamiento.
- Las salidas de fine-tuning también deben quedar bajo `./models/`.

## 2. Estrategia de entorno

- El entrenamiento y pipeline se ejecutan en contenedor.
- Runtime estándar: `Docker`.
- Imagen estándar: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`.
- El host debe exponer `/dev/kfd` y `/dev/dri` al contenedor.

## 3. Política de caché

- Cache de Hugging Face por proyecto:
  - `.cache/huggingface/` (ignorada por git).
- No usar cache global de usuario como dependencia operativa del proyecto.

## 4. Flujo oficial

1. `docker compose build trainer`
2. `docker compose run --rm trainer bash scripts/container/gpu_smoke.sh`
3. `docker compose run --rm trainer bash scripts/container/pipeline.sh`
4. `docker compose run --rm trainer bash scripts/container/train.sh`
5. `docker compose run --rm trainer pytest -q`

## 5. Preparación de datos C64

- Fuente: documentos en `c64_docs/`.
- Pipeline: `scripts/data_pipeline.py`.
- Salidas obligatorias:
  - `data/processed/dapt/*.parquet`
  - `data/processed/sft/*.jsonl`
  - `data/processed/validation_report.json`

## 6. Entrenamiento

- Script oficial: `scripts/fine_tune_mistral_8b.py`.
- Fases soportadas: `dapt`, `sft`, `both`.
- Recomendación inicial: LoRA + `bf16`.

## 7. Versionado y reproducibilidad

- Dependencias separadas en:
  - `requirements.base.txt`
  - `requirements.rocm72.txt`
  - `requirements.txt` (agregador)
- Commits atómicos y descriptivos.
- Documentación actualizada en el mismo cambio cuando se altera el flujo.

## 8. Seguridad y limpieza

- No commitear pesos de modelos, datos intermedios pesados ni caches.
- Limpiar artefactos temporales de instalaciones fallidas.
- Mantener `.gitignore` alineado con el flujo real del repositorio.

## 9. Referencias

- ROCm docs: https://rocm.docs.amd.com/
- PyTorch ROCm containers: https://hub.docker.com/r/rocm/pytorch/tags
- Transformers: https://huggingface.co/docs/transformers/index
