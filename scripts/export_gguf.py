#!/usr/bin/env python3
"""Export fine-tuned Ministral-3-8B-Thinking LoRA output to GGUF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer
from transformers.models.mistral3 import Mistral3ForConditionalGeneration

try:
    from prompt_contract import build_c64_system_prompt
except ImportError:  # pragma: no cover - import path differs under test runner
    from scripts.prompt_contract import build_c64_system_prompt


ALLOWED_BASE_MODEL = Path("models/Ministral-3-8B-Thinking")
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    """Run a shell command and fail fast on non-zero exit status."""
    cmd_str = " ".join(cmd)
    print(f"$ {cmd_str}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge LoRA adapter into Ministral-3-8B-Thinking, convert to GGUF, "
            "and optionally quantize for Ollama/llama.cpp."
        )
    )
    parser.add_argument("--base-model-path", default=str(ALLOWED_BASE_MODEL))
    parser.add_argument("--adapter-path", default="models/fine-tuned")
    parser.add_argument("--merged-model-dir", default="models/fine-tuned-merged-hf")
    parser.add_argument("--gguf-dir", default="models/gguf")
    parser.add_argument("--name-prefix", default="c64-ministral-3-8b-thinking-c64")
    parser.add_argument("--quantization", default="Q4_K_M")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--llama-cpp-dir", default=".cache/llama.cpp")
    parser.add_argument("--llama-cpp-ref", default="master")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--clean-merged", action="store_true")
    parser.add_argument("--export-lora-gguf", action="store_true")
    parser.add_argument("--no-update-llama-cpp", action="store_true")
    parser.add_argument("--no-modelfile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_allowed_base_model(path_str: str) -> Path:
    """Enforce repository policy for the canonical base model location."""
    workspace = Path.cwd().resolve()
    allowed = (workspace / ALLOWED_BASE_MODEL).resolve()
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != allowed:
        raise ValueError(
            "Base model path is restricted by project policy. "
            f"Use exactly: {ALLOWED_BASE_MODEL}"
        )
    return candidate


def resolve_dtype(dtype_arg: str, device: str) -> tuple[torch.dtype, str]:
    """Resolve export dtype from CLI input and active device."""
    if dtype_arg == "float16":
        return torch.float16, "float16"
    if dtype_arg == "bfloat16":
        return torch.bfloat16, "bfloat16"
    if dtype_arg == "float32":
        return torch.float32, "float32"

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    return torch.float32, "float32"


def resolve_device(device_arg: str) -> str:
    """Resolve runtime device; 'auto' prefers CUDA/ROCm when available."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def merge_lora_adapter(
    *,
    base_model_path: Path,
    adapter_path: Path,
    merged_model_dir: Path,
    device: str,
    dtype: torch.dtype,
    dtype_name: str,
    max_shard_size: str,
    dry_run: bool,
) -> None:
    """Merge LoRA adapter into base model and persist merged HF checkpoint."""
    if dry_run:
        print(
            f"[dry-run] Would merge adapter '{adapter_path}' into '{base_model_path}' "
            f"using device={device}, dtype={dtype_name}, output='{merged_model_dir}'."
        )
        return

    config = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=True)
    if hasattr(config, "tie_word_embeddings"):
        config.tie_word_embeddings = False

    print(f"Loading base model from {base_model_path} ...")
    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        str(base_model_path),
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        base_model.to("cuda")

    print(f"Attaching LoRA adapter from {adapter_path} ...")
    merged = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    merged = merged.merge_and_unload()

    print(f"Saving merged HF model to {merged_model_dir} ...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        str(merged_model_dir),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_model_dir))

    del merged
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()


def ensure_llama_cpp(
    *,
    llama_cpp_dir: Path,
    llama_cpp_ref: str,
    update_repo: bool,
    dry_run: bool,
) -> None:
    """Clone or update llama.cpp repository used for GGUF conversion."""
    git_dir = llama_cpp_dir / ".git"
    if not git_dir.exists():
        llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                llama_cpp_ref,
                LLAMA_CPP_REPO,
                str(llama_cpp_dir),
            ],
            dry_run=dry_run,
        )
        return

    if update_repo:
        run(
            ["git", "-C", str(llama_cpp_dir), "fetch", "--depth", "1", "origin", llama_cpp_ref],
            dry_run=dry_run,
        )
        run(["git", "-C", str(llama_cpp_dir), "checkout", "FETCH_HEAD"], dry_run=dry_run)


def ensure_llama_quantize_binary(*, llama_cpp_dir: Path, dry_run: bool) -> Path:
    """Build llama.cpp quantizer if not already available."""
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if quantize_bin.exists():
        return quantize_bin

    run(
        [
            "cmake",
            "-S",
            str(llama_cpp_dir),
            "-B",
            str(llama_cpp_dir / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        dry_run=dry_run,
    )
    run(
        [
            "cmake",
            "--build",
            str(llama_cpp_dir / "build"),
            "--config",
            "Release",
            "-j",
        ],
        dry_run=dry_run,
    )
    return quantize_bin


def write_modelfile(*, gguf_dir: Path, model_file: Path, system_prompt: str, dry_run: bool) -> None:
    """Write Ollama Modelfile pointing at the selected GGUF artifact."""
    modelfile_path = gguf_dir / "Modelfile"
    contents = f'FROM ./{model_file.name}\nSYSTEM """{system_prompt}"""\n'
    if dry_run:
        print(f"[dry-run] Would write {modelfile_path} with:")
        print(contents.rstrip())
        return

    modelfile_path.write_text(contents, encoding="utf-8")
    print(f"Wrote {modelfile_path}")


def ensure_sentencepiece_available(*, dry_run: bool) -> None:
    """Fail early if required tokenizer dependency is missing."""
    if dry_run:
        return
    try:
        import sentencepiece  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'sentencepiece'. Rebuild trainer image after updating requirements "
            "or install it in the container."
        ) from exc


def main() -> None:
    """End-to-end GGUF export orchestration."""
    args = parse_args()

    base_model_path = resolve_allowed_base_model(args.base_model_path)
    system_prompt = build_c64_system_prompt(base_model_path)
    adapter_path = Path(args.adapter_path).resolve()
    merged_model_dir = Path(args.merged_model_dir).resolve()
    gguf_dir = Path(args.gguf_dir).resolve()
    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()

    if not args.dry_run and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    device = resolve_device(args.device)
    if device == "cuda" and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA/ROCm backend is not available for merge step.")

    dtype, dtype_name = resolve_dtype(args.dtype, device)
    print(f"Using device={device}, dtype={dtype_name}")

    if not args.skip_merge:
        merge_lora_adapter(
            base_model_path=base_model_path,
            adapter_path=adapter_path,
            merged_model_dir=merged_model_dir,
            device=device,
            dtype=dtype,
            dtype_name=dtype_name,
            max_shard_size=args.max_shard_size,
            dry_run=args.dry_run,
        )
    elif not args.dry_run and not merged_model_dir.exists():
        raise FileNotFoundError(
            f"--skip-merge was used but merged model dir does not exist: {merged_model_dir}"
        )

    ensure_llama_cpp(
        llama_cpp_dir=llama_cpp_dir,
        llama_cpp_ref=args.llama_cpp_ref,
        update_repo=not args.no_update_llama_cpp,
        dry_run=args.dry_run,
    )

    ensure_sentencepiece_available(dry_run=args.dry_run)

    gguf_dir.mkdir(parents=True, exist_ok=True)
    f16_gguf = gguf_dir / f"{args.name_prefix}-F16.gguf"
    run(
        [
            "python",
            str(llama_cpp_dir / "convert_hf_to_gguf.py"),
            str(merged_model_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        dry_run=args.dry_run,
    )

    final_gguf = f16_gguf
    quantization = args.quantization.strip()
    if quantization.lower() != "none":
        quant_method = quantization.upper()
        quant_bin = ensure_llama_quantize_binary(llama_cpp_dir=llama_cpp_dir, dry_run=args.dry_run)
        quant_gguf = gguf_dir / f"{args.name_prefix}-{quant_method}.gguf"
        run(
            [
                str(quant_bin),
                str(f16_gguf),
                str(quant_gguf),
                quant_method,
            ],
            dry_run=args.dry_run,
        )
        final_gguf = quant_gguf

    if args.export_lora_gguf:
        lora_gguf = gguf_dir / f"{args.name_prefix}-LoRA-F16.gguf"
        run(
            [
                "python",
                str(llama_cpp_dir / "convert_lora_to_gguf.py"),
                str(adapter_path),
                "--base",
                str(base_model_path),
                "--outfile",
                str(lora_gguf),
                "--outtype",
                "f16",
            ],
            dry_run=args.dry_run,
        )

    if not args.no_modelfile:
        write_modelfile(
            gguf_dir=gguf_dir,
            model_file=final_gguf,
            system_prompt=system_prompt,
            dry_run=args.dry_run,
        )

    if args.clean_merged:
        if args.dry_run:
            print(f"[dry-run] Would remove merged dir: {merged_model_dir}")
        else:
            shutil.rmtree(merged_model_dir, ignore_errors=True)
            print(f"Removed merged dir: {merged_model_dir}")

    print("")
    print("Export finished.")
    print(f"Main GGUF: {final_gguf}")
    print(f"Ollama create command: ollama create c64-ministral-c64 -f {gguf_dir / 'Modelfile'}")
    print(f"llama.cpp example: llama-cli -m {final_gguf} -p \"LOAD\" -n 64")


if __name__ == "__main__":
    main()
