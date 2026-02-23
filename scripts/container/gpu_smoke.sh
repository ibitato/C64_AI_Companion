#!/usr/bin/env bash
set -euo pipefail

cd /workspace
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

python - <<'PY'
import json
import torch

payload = {
    "torch": torch.__version__,
    "hip": getattr(torch.version, "hip", None),
    "cuda_available": torch.cuda.is_available(),
}
if not payload["cuda_available"]:
    raise SystemExit("ROCm/CUDA backend not available inside container")

props = torch.cuda.get_device_properties(0)
payload["device_name"] = props.name
payload["vram_gb"] = round(props.total_memory / 1024**3, 2)
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = (a @ b).sum()
torch.cuda.synchronize()
payload["kernel_ok"] = float(c.item())
print(json.dumps(payload, indent=2))
PY
