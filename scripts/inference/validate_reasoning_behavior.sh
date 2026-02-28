#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${ROOT_DIR}/results/reasoning_validation"
IN_CONTAINER=0

MODELS="Q8_0 F16"
SEEDS="11 22 33"
N_PREDICT=192
CTX_SIZE=4096
OUTPUT_DIR=""

PROMPT_SINGLE="Explain VIC-II badlines in 3 concise technical bullets."
PROMPT_MULTI_1="Explain what the SID ADSR envelope does in concise technical terms."
PROMPT_MULTI_2="Now give 2 practical debugging checks for wrong SID envelope behavior."

THRESHOLD_THINK_TAG="${THRESHOLD_THINK_TAG:-0.95}"
THRESHOLD_BALANCED="${THRESHOLD_BALANCED:-0.95}"
THRESHOLD_FINAL="${THRESHOLD_FINAL:-0.95}"
THRESHOLD_MULTI="${THRESHOLD_MULTI:-0.90}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/inference/validate_reasoning_behavior.sh [options]

Options:
  --in-container           Internal flag used by docker wrapper.
  --models "LIST"          Space-separated quant list. Default: "Q8_0 F16"
  --seeds "LIST"           Space-separated seeds. Default: "11 22 33"
  --n-predict N            Max generated tokens. Default: 192
  --ctx-size N             Context size. Default: 4096
  --output-dir DIR         Output directory. Default: results/reasoning_validation/<timestamp>
  -h, --help               Show this help.

Notes:
  - Produces raw logs, per-run metrics CSV, and summary markdown.
  - Exits non-zero if contract thresholds are not met.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-container)
      IN_CONTAINER=1
      shift
      ;;
    --models)
      MODELS="${2:-}"
      shift 2
      ;;
    --seeds)
      SEEDS="${2:-}"
      shift 2
      ;;
    --n-predict)
      N_PREDICT="${2:-}"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${IN_CONTAINER}" -eq 0 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required for reproducible validation." >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose is required." >&2
    exit 1
  fi
  compose_cmd=(
    docker compose run --rm trainer
    bash scripts/inference/validate_reasoning_behavior.sh --in-container
    --models "$MODELS"
    --seeds "$SEEDS"
    --n-predict "$N_PREDICT"
    --ctx-size "$CTX_SIZE"
  )
  if [[ -n "${OUTPUT_DIR}" ]]; then
    compose_cmd+=(--output-dir "$OUTPUT_DIR")
  fi
  exec "${compose_cmd[@]}"
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/.cache/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_DIR="${RESULTS_DIR}/${TS}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT_DIR}/${OUTPUT_DIR}"
fi
RAW_DIR="${OUTPUT_DIR}/raw"
mkdir -p "${RAW_DIR}"

RUNS_FILE="${OUTPUT_DIR}/runs.csv"
METRICS_CSV="${OUTPUT_DIR}/metrics.csv"
SUMMARY_MD="${OUTPUT_DIR}/summary.md"

echo "run_type,quant,seed,log_path" > "${RUNS_FILE}"

echo "Running reasoning behavior validation..."
for quant in ${MODELS}; do
  for seed in ${SEEDS}; do
    single_log="${RAW_DIR}/single_${quant}_seed${seed}.log"
    multi_log="${RAW_DIR}/multi_${quant}_seed${seed}.log"

    bash "${ROOT_DIR}/scripts/inference/run_llama_cpp.sh" "${quant}" "${PROMPT_SINGLE}" \
      --reasoning-format none \
      --reasoning-budget -1 \
      --temp 0 \
      --seed "${seed}" \
      -n "${N_PREDICT}" \
      -c "${CTX_SIZE}" \
      --no-show-timings \
      --no-perf \
      > "${single_log}" 2>&1
    echo "single,${quant},${seed},${single_log}" >> "${RUNS_FILE}"

    printf "%s\n/exit\n" "${PROMPT_MULTI_2}" | \
      bash "${ROOT_DIR}/scripts/inference/run_llama_cpp.sh" "${quant}" "${PROMPT_MULTI_1}" \
        --multi-turn \
        --simple-io \
        --reasoning-format none \
        --reasoning-budget -1 \
        --temp 0 \
        --seed "${seed}" \
        -n "${N_PREDICT}" \
        -c "${CTX_SIZE}" \
        --no-show-timings \
        --no-perf \
        > "${multi_log}" 2>&1
    echo "multi,${quant},${seed},${multi_log}" >> "${RUNS_FILE}"
  done
done

python3 - <<'PY' "${RUNS_FILE}" "${METRICS_CSV}" "${SUMMARY_MD}" "${THRESHOLD_THINK_TAG}" "${THRESHOLD_BALANCED}" "${THRESHOLD_FINAL}" "${THRESHOLD_MULTI}"
import csv
import re
import sys
from pathlib import Path

runs_file = Path(sys.argv[1])
metrics_csv = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
threshold_think = float(sys.argv[4])
threshold_balanced = float(sys.argv[5])
threshold_final = float(sys.argv[6])
threshold_multi = float(sys.argv[7])

think_open = re.compile(r"\[THINK\]")
think_close = re.compile(r"\[/THINK\]")

rows = []
with runs_file.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for rec in reader:
        log_text = Path(rec["log_path"]).read_text(encoding="utf-8", errors="ignore")
        open_count = len(think_open.findall(log_text))
        close_count = len(think_close.findall(log_text))
        balanced = int(open_count > 0 and open_count == close_count)
        has_think = int(open_count > 0 and close_count > 0)
        final_after = ""
        if "[/THINK]" in log_text:
            final_after = log_text.rsplit("[/THINK]", 1)[-1].strip()
        has_final = int(bool(final_after))
        multi_retained = int(rec["run_type"] == "multi" and open_count >= 2 and balanced == 1)
        rows.append(
            {
                "run_type": rec["run_type"],
                "quant": rec["quant"],
                "seed": rec["seed"],
                "log_path": rec["log_path"],
                "think_open_count": open_count,
                "think_close_count": close_count,
                "has_think_tag": has_think,
                "balanced_think_tags": balanced,
                "has_final_after_think": has_final,
                "multi_turn_retained": multi_retained,
            }
        )

with metrics_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_type",
            "quant",
            "seed",
            "think_open_count",
            "think_close_count",
            "has_think_tag",
            "balanced_think_tags",
            "has_final_after_think",
            "multi_turn_retained",
            "log_path",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

single = [r for r in rows if r["run_type"] == "single"]
multi = [r for r in rows if r["run_type"] == "multi"]

def ratio(items, key):
    if not items:
        return 0.0
    return sum(int(i[key]) for i in items) / len(items)

single_think = ratio(single, "has_think_tag")
single_balanced = ratio(single, "balanced_think_tags")
single_final = ratio(single, "has_final_after_think")
multi_retention = ratio(multi, "multi_turn_retained")

ok = (
    single_think >= threshold_think
    and single_balanced >= threshold_balanced
    and single_final >= threshold_final
    and multi_retention >= threshold_multi
)

summary_lines = [
    "# Reasoning Validation Report",
    "",
    "## Aggregate Metrics",
    "",
    "| Metric | Value | Threshold |",
    "|---|---:|---:|",
    f"| single_think_tag_rate | {single_think:.4f} | {threshold_think:.4f} |",
    f"| single_balanced_tag_rate | {single_balanced:.4f} | {threshold_balanced:.4f} |",
    f"| single_final_after_think_rate | {single_final:.4f} | {threshold_final:.4f} |",
    f"| multi_turn_retention_rate | {multi_retention:.4f} | {threshold_multi:.4f} |",
    "",
    f"- `status`: `{'PASS' if ok else 'FAIL'}`",
    f"- `runs`: single={len(single)} multi={len(multi)}",
    "",
    "## Artifacts",
    "",
    f"- Metrics CSV: `{metrics_csv}`",
    f"- Raw logs: `{runs_file.parent / 'raw'}`",
]
summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(f"Metrics CSV: {metrics_csv}")
print(f"Summary: {summary_md}")
if not ok:
    sys.exit(2)
PY

echo ""
echo "Reasoning validation finished."
echo "Output directory: ${OUTPUT_DIR}"
