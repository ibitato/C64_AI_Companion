import os
import shutil
import subprocess
import sys
import textwrap

import pytest


REQUIRE_GPU = os.environ.get("C64_REQUIRE_GPU", "0") == "1"


def _skip_or_fail(message: str) -> None:
    if REQUIRE_GPU:
        pytest.fail(message)
    pytest.skip(message)


def _rocminfo_ok() -> bool:
    if shutil.which("rocminfo") is None:
        return False
    proc = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False)
    return proc.returncode == 0 and "HSA Agents" in proc.stdout


def test_rocm_runtime_available():
    if not _rocminfo_ok():
        _skip_or_fail("rocminfo not available or ROCm runtime is not operational")


def test_torch_reports_rocm7_stack():
    try:
        import torch
    except Exception as exc:
        _skip_or_fail(f"torch import failed: {exc}")
        return

    if not torch.cuda.is_available():
        _skip_or_fail("torch.cuda.is_available() is false")

    hip_version = getattr(torch.version, "hip", None)
    if hip_version is None:
        _skip_or_fail("torch build is missing HIP runtime metadata")
    if not hip_version.startswith("7."):
        _skip_or_fail(f"expected ROCm 7.x torch, got HIP={hip_version}")


def test_torch_basic_gpu_kernel():
    code = textwrap.dedent(
        """
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(2)

        a = torch.ones((256, 256), device="cuda")
        b = torch.full((256, 256), 2.0, device="cuda")
        c = (a @ b).sum()
        torch.cuda.synchronize()
        if float(c.item()) <= 0:
            raise SystemExit(3)
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    if proc.returncode == 2:
        _skip_or_fail("GPU backend unavailable for torch kernel test")
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    known_mismatch = (
        "invalid device function",
        "hiperrorinvaliddevicefunction",
        "no kernel image is available for execution on the device",
        "hiperrornobinaryforgpu",
    )
    if any(marker in combined for marker in known_mismatch):
        _skip_or_fail("ROCm/PyTorch kernel mismatch for this host wheel/GPU")
    assert proc.returncode == 0, f"unexpected kernel failure:\nstdout={proc.stdout}\nstderr={proc.stderr}"
