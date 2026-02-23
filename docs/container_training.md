# Container Training Guide (ROCm 7.2)

## 1. Build

```bash
docker compose build trainer
```

## 2. Verificar GPU dentro del contenedor

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```

## 3. Preparar dataset

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh
```

## 4. Entrenar

### Flujo completo

```bash
docker compose run --rm trainer bash scripts/container/train.sh
```

### Comando parametrizado

```bash
docker compose run --rm trainer bash scripts/container/train.sh \
  --phase both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --precision bf16 \
  --use-lora
```

## 5. Ejecutar tests

```bash
docker compose run --rm trainer pytest -q
```

Forzar error si no hay GPU ROCm:

```bash
docker compose run --rm -e C64_REQUIRE_GPU=1 trainer pytest -q tests/test_gpu.py
```

## 6. Troubleshooting

### `torch.cuda.is_available() == False`

- Verificar dispositivos en host:
  - `ls -l /dev/kfd /dev/dri/renderD128`
- Verificar grupos del usuario:
  - `id` (debe incluir `video` y `render`)
- Verificar que se usa `docker-compose.yml` del repo (incluye devices/group_add).

### Error por ruta de modelo

El proyecto solo permite:
- `models/Ministral-3-8B-Thinking`

Si el script falla por política de ruta, corrige `--model-path`.

### Lentitud o OOM

- Bajar `--batch-size`.
- Subir `--grad-accum` para mantener batch efectivo.
- Reducir `--max-length` temporalmente.
