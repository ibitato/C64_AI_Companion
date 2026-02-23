# Software Requerido (ROCm 6.4.4 + Pipeline C64)

Este documento define el software mínimo para ejecutar el flujo completo del proyecto:
- preparación de datos C64 (incluyendo OCR local),
- fine-tuning con `Ministral-3-8B-Thinking`,
- validación y reporte de calidad.

## 1. Sistema operativo y GPU

- Fedora Linux 43.
- Kernel 6.18.x.
- ROCm 6.4.4.
- GPU AMD compatible con ROCm.

## 2. Paquetes del sistema (Fedora)

### 2.1 Base ML/GPU
```bash
sudo dnf install rocm
```

### 2.2 OCR para manuales escaneados
```bash
sudo dnf install ocrmypdf tesseract tesseract-langpack-eng ghostscript unpaper poppler-utils
```

## 3. Entorno Python (venv)

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notas:
- `requirements.txt` ya está configurado para resolver desde PyPI y usar ruedas ROCm para PyTorch.
- El comando anterior debe bastar sin pasos manuales extra de índices.

## 4. Modelo base (ruta local obligatoria)

Ruta válida del modelo:
- `models/Ministral-3-8B-Thinking`

Ejemplo de descarga:
```bash
hf download mistralai/Ministral-3-8B-Reasoning-2512 --local-dir ./models/Ministral-3-8B-Thinking
```

## 5. Verificación rápida

```bash
source venv/bin/activate
python - <<'PY'
import importlib
for m in ["torch","transformers","datasets","peft","accelerate","lm_eval","pypdf","ocrmypdf"]:
    importlib.import_module(m)
print("OK")
PY
```

## 6. Referencias

- ROCm: https://rocm.docs.amd.com/
- Transformers: https://huggingface.co/docs/transformers/index
- OCRmyPDF: https://ocrmypdf.readthedocs.io/
