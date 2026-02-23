# Fine-Tuning Best Practices (Ministral-3-8B-Thinking + ROCm 7.2)

Guía alineada con la implementación actual del repositorio.

## 1. Flujo recomendado

- Paso 1: construir contenedor.
- Paso 2: preparar dataset C64.
- Paso 3: entrenar en dos fases (`DAPT` + `SFT`) con LoRA.

```bash
docker compose build trainer
docker compose run --rm trainer bash scripts/container/pipeline.sh
docker compose run --rm trainer bash scripts/container/train.sh
```

## 2. Política de modelo

- Modelo original único permitido: `models/Ministral-3-8B-Thinking`.
- No usar rutas fuera del proyecto para el modelo base.
- El entrenamiento valida esta restricción y falla si no se cumple.

## 3. Parámetros de arranque recomendados

- `max_length`: 2048
- `batch_size`: 2
- `grad_accum`: 16
- `learning_rate`: 2e-5
- `epochs`: 3
- `precision`: `bf16`
- `use_lora`: activado
- `assistant_only_loss`: desactivado por defecto para este template
- `packing`: desactivado por defecto (activar solo si validas atención compatible)

## 4. Comandos de referencia

### DAPT

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase dapt \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --output-dir models/fine-tuned-dapt \
  --precision bf16 \
  --use-lora
```

### SFT

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase sft \
  --model-path models/Ministral-3-8B-Thinking \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned-sft \
  --precision bf16 \
  --no-assistant-only-loss \
  --no-packing \
  --use-lora
```

### DAPT + SFT

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --precision bf16 \
  --no-assistant-only-loss \
  --no-packing \
  --use-lora
```

## 5. Señales de alerta

- `validation_report.json` con baja cobertura OCR.
- `validation` vacío en DAPT/SFT.
- divergencia de loss o nans: bajar LR, revisar chunks y calidad OCR.

## 6. Verificación posterior

```bash
docker compose run --rm trainer pytest -q
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```
