# System Requirements

## Host Requirements

- Linux host with working Docker Engine and Docker Compose plugin.
- AMD GPU host device exposure:
  - `/dev/kfd`
  - `/dev/dri`
- User in `video` and `render` groups.

## Container Runtime Baseline

- Training image: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`
- Python: 3.10
- Dependency lock strategy:
  - `requirements.base.txt`
  - `requirements.rocm72.txt`
  - `requirements.txt`

## Required Host Export

Before `docker compose` commands:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
```

## Build

```bash
docker compose build trainer
```

## Minimum Validation

```bash
docker compose run --rm trainer bash scripts/container/gpu_smoke.sh
docker compose run --rm trainer pytest -q
```
