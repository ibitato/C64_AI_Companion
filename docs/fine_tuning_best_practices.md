# Fine-Tuning Best Practices (C64_AI_Companion)

Esta guía está alineada con la implementación actual del repositorio:
- modelo objetivo: `Ministral-3-8B-Thinking`,
- pipeline de datos: `scripts/data_pipeline.py`,
- entrenamiento unificado: `scripts/fine_tune_mistral_8b.py`.

## 1. Estrategia recomendada

Usar `DAPT + SFT`:
- **DAPT** (Domain Adaptive Pre-Training): mejora conocimiento técnico C64 desde manuales.
- **SFT** (Supervised Fine-Tuning): mejora formato conversacional y utilidad de respuesta.

## 2. Preparación de datos (obligatoria)

Ejecuta el pipeline completo:

```bash
source venv/bin/activate
python scripts/data_pipeline.py --stage all --allow-ocr
python scripts/data_qc_report.py
```

Artefactos clave:
- `data/processed/dapt/{train,validation,test}.parquet`
- `data/processed/sft/{train,validation,test}.jsonl`
- `data/processed/validation_report.json`
- `docs/data_qc_report.md`

## 3. Reglas de calidad de datos

- Mantener puntuación y mayúsculas (no degradar sintaxis técnica).
- Hacer chunking por tokens para DAPT (no truncar documentos completos a 512).
- Separar `train/validation/test` por documento para evitar leakage.
- Usar OCR local si hay PDFs escaneados.

## 4. Entrenamiento

### 4.1 DAPT
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode dapt \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --output-dir models/fine-tuned-dapt \
  --use-lora
```

### 4.2 SFT
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode sft \
  --model-path models/Ministral-3-8B-Thinking \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned-sft \
  --use-lora
```

### 4.3 DAPT + SFT
```bash
python scripts/fine_tune_mistral_8b.py \
  --mode both \
  --model-path models/Ministral-3-8B-Thinking \
  --dapt-dir data/processed/dapt \
  --sft-dir data/processed/sft \
  --output-dir models/fine-tuned \
  --use-lora
```

## 5. Parámetros iniciales recomendados

- `max_length`: 2048
- `batch_size`: 2
- `grad_accum`: 16
- `learning_rate`: 2e-5
- `epochs`: 3

Ajusta según memoria real y estabilidad de loss.

## 6. Señales de alerta

- `coverage_ratio` bajo en `validation_report.json`:
  indica OCR insuficiente o extracción pobre.
- `validation` vacío:
  revisar split y tamaño de corpus.
- loss de validación no mejora:
  bajar LR o reducir mezcla de datos ruidosos.

## 7. Compatibilidad con reasoning model

- No generar CoT sintético artificial.
- Mantener formato chat consistente con tokenizer/template del modelo.
- Priorizar respuestas finales factuales y trazables a fuente.

## 8. Referencias

- Transformers language modeling: https://huggingface.co/docs/transformers/tasks/language_modeling
- Transformers chat templating: https://huggingface.co/docs/transformers/chat_templating
- TRL SFT: https://huggingface.co/docs/trl/sft_trainer
