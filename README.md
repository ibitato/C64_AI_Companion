# C64_AI_Companion

Fine-tuning de `Ministral-3-8B-Thinking` para conocimiento técnico de Commodore 64.

## Estrategia vigente

El proyecto opera en modo **container-first** para entrenamiento:
- host: Fedora 43 (solo runtime de contenedor + acceso a GPU AMD),
- contenedor: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`,
- modelo base obligatorio: `models/Ministral-3-8B-Thinking`.

## Estructura principal

```text
C64_AI_Companion/
├── c64_docs/                     # Manuales/documentación C64 de entrada
├── data/
├── docs/
├── models/                       # Modelo base + salidas fine-tuned
├── scripts/
│   ├── data_pipeline.py
│   ├── export_gguf.py
│   ├── fine_tune_mistral_8b.py
│   └── container/
│       ├── export_gguf.sh
│       ├── gpu_smoke.sh
│       ├── pipeline.sh
│       └── train.sh
├── Dockerfile.train
├── docker-compose.yml
└── requirements*.txt
```

## Requisitos del host

- Docker Engine + Docker Compose plugin.
- GPU AMD con `/dev/kfd` y `/dev/dri` accesibles.
- Usuario en grupos `video` y `render`.
- Exportar UID/GID antes de usar `docker compose` en este repo:
  ```bash
  export LOCAL_UID=$(id -u)
  export LOCAL_GID=$(id -g)
  ```

## Flujo rápido

### 1) Build de imagen

```bash
docker compose build trainer
```

### 2) Smoke test de GPU en contenedor

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```

### 3) Preparar dataset C64

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh
```

Genera:
- `data/processed/dapt/{train,validation,test}.parquet`
- `data/processed/sft/{train,validation,test}.jsonl`
- `data/processed/validation_report.json`
- `docs/data_qc_report.md`

### 4) Entrenamiento

Entrenamiento completo (DAPT + SFT):

```bash
docker compose run --rm trainer bash scripts/container/train.sh
```

Ejemplo DAPT corto:

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase dapt \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --output-dir models/fine-tuned-dapt \
  --precision bf16 \
  --use-lora \
  --max-steps 20
```

Nota SFT:
- con el chat template actual de `Ministral-3-8B-Thinking`, el proyecto usa por defecto
  `assistant_only_loss=False` y `packing=False` para evitar incompatibilidades de máscara.

### 5) Exportar a GGUF (Ollama / llama.cpp)

```bash
docker compose run --rm trainer bash scripts/container/export_gguf.sh \
  --base-model-path models/Ministral-3-8B-Thinking \
  --adapter-path models/fine-tuned \
  --gguf-dir models/gguf \
  --quantization Q4_K_M
```

Salidas:
- `models/gguf/c64-ministral-3-8b-thinking-c64-F16.gguf`
- `models/gguf/c64-ministral-3-8b-thinking-c64-Q4_K_M.gguf`
- `models/gguf/Modelfile`

Generar cuantizaciones adicionales (`Q6_K` y `Q8_0`) desde `F16`:

```bash
bash scripts/inference/quantize_additional_gguf.sh
```

Preparar todos los `Modelfile` locales (`Q4`, `Q6`, `Q8`, `F16`):

```bash
bash scripts/inference/prepare_runtime_assets.sh
```

Uso en Ollama (registra Q4/Q6/Q8):

```bash
bash scripts/inference/create_ollama_models.sh
```

Uso en llama.cpp (elige `Q4_K_M`, `Q6_K`, `Q8_0` o `F16`):

```bash
bash scripts/inference/run_llama_cpp.sh Q6_K "Explica el SID del Commodore 64 en 2 frases."
```

## Tests

```bash
docker compose run --rm trainer pytest -q
```

Para forzar fallo si no hay GPU ROCm disponible:

```bash
docker compose run --rm -e C64_REQUIRE_GPU=1 trainer pytest -q tests/test_gpu.py
```

## Política de modelo base

- Única ruta válida para el modelo original: `models/Ministral-3-8B-Thinking`.
- No se usa cache global de usuario como directorio de trabajo de modelo.
- El script de entrenamiento valida esta política en runtime.

## Documentación relacionada

- `AGENTS.md`
- `docs/container_training.md`
- `docs/software_requerido.md`
- `docs/fine_tuning_best_practices.md`
- `docs/data_pipeline.md`
