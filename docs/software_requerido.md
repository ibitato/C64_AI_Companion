# Software Requerido (Container-first ROCm 7.2)

Este proyecto ya no instala ROCm userland en Fedora host. El entrenamiento y pipeline se ejecutan dentro de contenedor ROCm 7.2.

## 1. Host

- Fedora 43 (u otra distro con Docker funcional).
- Docker Engine + Docker Compose plugin.
- Kernel/driver con acceso a GPU AMD:
  - `/dev/kfd`
  - `/dev/dri`
- Usuario en grupos `video` y `render`.

Comprobación rápida:

```bash
id
ls -l /dev/kfd /dev/dri/renderD128
```

## 2. Contenedor estándar del proyecto

- Imagen base: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`
- Python: 3.10
- PyTorch ROCm: 7.2 (wheel/pila ROCm 7.x)

Construcción:

```bash
docker compose build trainer
```

## 3. Dependencias Python reproducibles

- `requirements.base.txt`: stack de aplicación/pipeline/tests
- `requirements.rocm72.txt`: stack ROCm 7.2 (`torch`, `torchvision`, `triton`)
- `requirements.txt`: agregador de ambos

Instalación (dentro de contenedor):

```bash
pip install -r requirements.txt
```

## 4. OCR y pipeline de datos

El contenedor incluye:
- `ocrmypdf`
- `tesseract-ocr`
- `ghostscript`
- `unpaper`
- `poppler-utils`

## 5. Modelo base (obligatorio)

Ruta válida:
- `models/Ministral-3-8B-Thinking`

Caché HF del proyecto:
- `/workspace/.cache/huggingface` (montado desde `.cache/huggingface` local)

## 6. Verificación mínima

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
docker compose run --rm trainer pytest -q
```
