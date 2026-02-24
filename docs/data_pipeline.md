# C64 Data Pipeline (Container-first)

El pipeline oficial se ejecuta dentro del contenedor ROCm 7.2.

## Etapas

- `manifest`: inventario de documentos con hash y metadatos.
- `extract`: extracción de texto PDF/HTML, con OCR opcional.
- `normalize`: limpieza conservadora (sin perder sintaxis técnica).
- `dedup`: deduplicación exacta y aproximada.
- `build_dapt`: dataset DAPT en chunks por tokens.
- `build_sft`: dataset conversacional SFT (`messages`).
- `validate`: reporte de cobertura y consistencia.
- `all`: ejecuta todo en orden.

## Ejecución recomendada

Antes de correr `docker compose`, exporta UID/GID del usuario actual:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
```

Luego:

```bash
docker compose run --rm trainer bash scripts/container/pipeline.sh
```

## Salidas esperadas

- `data/interim/manifest/manifest.parquet`
- `data/interim/extracted/pages.parquet`
- `data/interim/normalized/pages_normalized.parquet`
- `data/interim/dedup/pages_dedup.parquet`
- `data/processed/dapt/{train,validation,test}.parquet`
- `data/processed/sft/{train,validation,test}.jsonl`
- `data/processed/validation_report.json`
- `docs/data_qc_report.md`

## Notas operativas

- Ruta de modelo base obligatoria: `models/Ministral-3-8B-Thinking`.
- Caché HF del proyecto: `.cache/huggingface/`.
- Para habilitar OCR local, `pipeline.sh` ya ejecuta `--allow-ocr`.
