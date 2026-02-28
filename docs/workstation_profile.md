# Workstation Profile and Runtime Compatibility

## Purpose

Document the fine-tuning workstation used for this project and clearly define the host/container runtime model required for Strix Halo support.

## Host Profile (Captured)

- Date captured: 2026-02-24
- OS: Fedora Linux 43 (Server Edition)
- Kernel: `6.18.8-200.fc43.x86_64`
- CPU: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S` (16 cores / 32 threads)
- System RAM: 30 GiB
- GPU: `AMD Radeon 8060S`
- GPU VRAM (PyTorch visible): 96.00 GiB

## Runtime Compatibility Matrix

| Layer | Purpose | Observed Stack | Notes |
| --- | --- | --- | --- |
| Host kernel/driver layer | Device access and kernel-side GPU support | Fedora 43 + Linux 6.18.8 | Provides `/dev/kfd` and `/dev/dri` to containers |
| Host Python runtime (local) | Local utility context only | torch `2.9.1+rocm6.4`, HIP `6.4.x` | Not the canonical training runtime |
| Container runtime (canonical) | Training, data pipeline, packaging | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` | Source of truth for reproducible training |

## Strix Halo Compatibility Policy

This project intentionally uses a container ROCm/HIP runtime profile for Strix Halo compatibility.

- Host and container ROCm/HIP versions may differ.
- This version split is expected and valid.
- Reproducibility and support statements are defined by the **container runtime**, not by host userland versions.

## Required Host Configuration

- Docker Engine + Docker Compose plugin.
- Device exposure:
  - `/dev/kfd`
  - `/dev/dri`
- User in groups:
  - `video`
  - `render`
- UID/GID export before `docker compose` commands:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
```

## Validation Commands

### Host-level sanity

```bash
id
ls -l /dev/kfd /dev/dri/renderD128
```

### Container-level runtime verification

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
```

Optional runtime detail check:

```bash
docker compose run --rm trainer python - <<'PY'
import torch
print(torch.__version__)
print(getattr(torch.version, 'hip', None))
print(torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, round(p.total_memory / 1024**3, 2))
PY
```

## Known Failure Signatures

- `torch.cuda.is_available() == False`
  - Usually device mapping/permissions issue.
- `no kernel image is available for execution on the device`
  - Usually host/container runtime mismatch for selected wheel/kernel profile.
- `hipErrorInvalidDeviceFunction`
  - Usually incompatible kernel binary for active runtime stack.

## Recovery Guidance

1. Validate host device nodes and groups.
2. Rebuild and re-run inside canonical container image.
3. Verify container torch/HIP runtime and GPU visibility.
4. Do not diagnose training support from host local Python runtime alone.
