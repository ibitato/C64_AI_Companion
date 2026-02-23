#!/usr/bin/env python3
"""
Backward-compatible wrapper for the new data pipeline.

Equivalent to:
  python scripts/data_pipeline.py --stage all --allow-ocr
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C64 preprocessing pipeline.")
    parser.add_argument("--source-dir", default="c64_docs")
    parser.add_argument("--model-path", default="models/Ministral-3-8B-Thinking")
    parser.add_argument("--allow-ocr", action="store_true", default=True)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-chunk-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = Path(__file__).with_name("data_pipeline.py")
    allow_ocr = args.allow_ocr and not args.no_ocr

    cmd = [
        sys.executable,
        str(pipeline),
        "--stage",
        "all",
        "--source-dir",
        args.source_dir,
        "--model-path",
        args.model_path,
        "--block-size",
        str(args.block_size),
        "--stride",
        str(args.stride),
        "--min-chunk-tokens",
        str(args.min_chunk_tokens),
        "--seed",
        str(args.seed),
    ]
    if allow_ocr:
        cmd.append("--allow-ocr")

    proc = subprocess.run(cmd, check=False)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
