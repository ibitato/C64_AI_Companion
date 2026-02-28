#!/usr/bin/env python3
"""
Generate Markdown quality report from data pipeline validation JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("data/processed/validation_report.json")
DEFAULT_OUTPUT = Path("docs/data_qc_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data QC markdown report.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Validation report JSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown output path.")
    return parser.parse_args()


def load_report(path: Path) -> dict:
    """Load validation JSON report from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Validation report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_markdown(report: dict) -> str:
    """Render a compact markdown summary from validation metrics."""
    checks = report.get("checks", {})
    dapt = checks.get("dapt_chunks", {})
    sft = checks.get("sft_examples", {})
    thinking = checks.get("sft_thinking", {})
    has_thinking = bool(thinking)
    warnings = report.get("warnings", [])

    lines: list[str] = []
    lines.append("# Data QC Report")
    lines.append("")
    lines.append(f"- `ok`: `{report.get('ok', False)}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total pages | {checks.get('total_pages', 0)} |")
    lines.append(f"| Pages with text | {checks.get('pages_with_text', 0)} |")
    lines.append(f"| Coverage ratio | {checks.get('coverage_ratio', 0)} |")
    lines.append(f"| Pages after dedup | {checks.get('dedup_pages', 0)} |")
    lines.append("")
    lines.append("## DAPT Dataset")
    lines.append("")
    lines.append("| Split | Chunks |")
    lines.append("|---|---:|")
    lines.append(f"| train | {dapt.get('train', 0)} |")
    lines.append(f"| validation | {dapt.get('validation', 0)} |")
    lines.append(f"| test | {dapt.get('test', 0)} |")
    lines.append(f"| total | {dapt.get('total', 0)} |")
    lines.append("")
    lines.append("## SFT Dataset")
    lines.append("")
    lines.append("| Split | Examples |")
    lines.append("|---|---:|")
    lines.append(f"| train | {sft.get('train', 0)} |")
    lines.append(f"| validation | {sft.get('validation', 0)} |")
    lines.append(f"| test | {sft.get('test', 0)} |")
    lines.append(f"| total | {sft.get('total', 0)} |")
    lines.append("")
    lines.append("## SFT Thinking Coverage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    if has_thinking:
        lines.append(f"| Assistant messages | {thinking.get('assistant_messages_total', 0)} |")
        lines.append(f"| Assistant messages with THINK | {thinking.get('assistant_with_think_total', 0)} |")
        lines.append(f"| THINK ratio | {thinking.get('assistant_with_think_ratio', 0)} |")
        lines.append(f"| Unique THINK traces | {thinking.get('unique_think_texts', 0)} |")
        lines.append(f"| Conversations with >1 assistant turn | {thinking.get('conversations_with_multiple_assistant_turns', 0)} |")
        lines.append(f"| Multi-turn ratio | {thinking.get('multi_turn_ratio', 0)} |")
    else:
        lines.append("| Assistant messages | n/a |")
        lines.append("| Assistant messages with THINK | n/a |")
        lines.append("| THINK ratio | n/a |")
        lines.append("| Unique THINK traces | n/a |")
        lines.append("| Conversations with >1 assistant turn | n/a |")
        lines.append("| Multi-turn ratio | n/a |")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for markdown report generation."""
    args = parse_args()
    report = load_report(args.input)
    md = render_markdown(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"QC report written to {args.output}")


if __name__ == "__main__":
    main()
