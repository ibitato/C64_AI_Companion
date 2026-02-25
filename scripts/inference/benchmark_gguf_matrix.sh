#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
RESULTS_DIR="${ROOT_DIR}/results/benchmarks"
LLAMA_BIN="${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-completion"
IN_CONTAINER=0

MODELS="F16 Q4_K_M Q6_K Q8_0"
PROMPT="List 4 concise points about the Commodore 64 SID chip."
N_PREDICT=96
CTX_SIZE=2048
TIMEOUT_SEC=240
OUTPUT_CSV=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/inference/benchmark_gguf_matrix.sh [options]

Options:
  --in-container           Internal flag used by docker wrapper.
  --output FILE            Output CSV path. Default: results/benchmarks/gguf_benchmark_<timestamp>.csv
  --models "LIST"          Space-separated quant list. Default: "F16 Q4_K_M Q6_K Q8_0"
  --prompt TEXT            Prompt for benchmark generation.
  --n-predict N            Number of generated tokens. Default: 96
  --ctx-size N             Context size. Default: 2048
  --timeout SEC            Per-model timeout in seconds. Default: 240
  -h, --help               Show this help.

Examples:
  bash scripts/inference/benchmark_gguf_matrix.sh
  bash scripts/inference/benchmark_gguf_matrix.sh --models "Q4_K_M Q6_K Q8_0" --n-predict 128
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-container)
      IN_CONTAINER=1
      shift
      ;;
    --output)
      OUTPUT_CSV="${2:-}"
      shift 2
      ;;
    --models)
      MODELS="${2:-}"
      shift 2
      ;;
    --prompt)
      PROMPT="${2:-}"
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
    --timeout)
      TIMEOUT_SEC="${2:-}"
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
    echo "ERROR: docker is required to run this benchmark reproducibly." >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose is required." >&2
    exit 1
  fi
  compose_cmd=(
    docker compose run --rm trainer
    bash scripts/inference/benchmark_gguf_matrix.sh --in-container
    --models "$MODELS"
    --prompt "$PROMPT"
    --n-predict "$N_PREDICT"
    --ctx-size "$CTX_SIZE"
    --timeout "$TIMEOUT_SEC"
  )
  if [[ -n "${OUTPUT_CSV}" ]]; then
    compose_cmd+=(--output "$OUTPUT_CSV")
  fi
  exec "${compose_cmd[@]}"
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/.cache/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "ERROR: llama-completion not found at '${LLAMA_BIN}'" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

if [[ -z "${OUTPUT_CSV}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_CSV="${RESULTS_DIR}/gguf_benchmark_${TS}.csv"
fi

# Normalize to absolute workspace path for container runs.
if [[ "${OUTPUT_CSV}" != /* ]]; then
  OUTPUT_CSV="${ROOT_DIR}/${OUTPUT_CSV}"
fi
mkdir -p "$(dirname "${OUTPUT_CSV}")"

echo "quant,status,rc,model_path,offload_layers,offload_total,eval_ms,eval_runs,tok_per_s,total_ms,total_tokens,gpu_use_avg_pct,gpu_use_max_pct,vram_max_pct,power_max_w" > "${OUTPUT_CSV}"

parse_line_number() {
  local pattern="$1"
  local file="$2"
  sed -n "s/${pattern}/\\1/p" "${file}" | head -n 1
}

for quant in ${MODELS}; do
  model_path="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${quant}.gguf"
  log_file="$(mktemp /tmp/gguf_bench_${quant}_XXXX.log)"
  smi_file="$(mktemp /tmp/gguf_smi_${quant}_XXXX.csv)"

  status="OK"
  rc=0

  echo "=== Benchmark ${quant} ==="

  if [[ ! -f "${model_path}" ]]; then
    status="MISSING"
    rc=127
    echo "${quant},${status},${rc},${model_path},,,,,,,,,,," >> "${OUTPUT_CSV}"
    rm -f "${log_file}" "${smi_file}"
    continue
  fi

  smi_pid=""
  if command -v rocm-smi >/dev/null 2>&1; then
    (
      for _ in $(seq 1 90); do
        rocm-smi --showuse --showmemuse --showpower --csv 2>/dev/null \
          | awk -F, '/^card[0-9]+,/{print $0}' >> "${smi_file}" || true
        sleep 1
      done
    ) &
    smi_pid=$!
  fi

  set +e
  timeout "${TIMEOUT_SEC}" "${LLAMA_BIN}" \
    -m "${model_path}" \
    -ngl 99 \
    -c "${CTX_SIZE}" \
    -n "${N_PREDICT}" \
    -no-cnv \
    -p "${PROMPT}" \
    > "${log_file}" 2>&1
  rc=$?
  set -e

  if [[ -n "${smi_pid}" ]]; then
    kill "${smi_pid}" >/dev/null 2>&1 || true
    wait "${smi_pid}" 2>/dev/null || true
  fi

  if [[ "${rc}" -eq 124 ]]; then
    status="TIMEOUT"
  elif [[ "${rc}" -ne 0 ]]; then
    status="ERROR"
  fi

  offload_layers="$(parse_line_number '.*offloaded \([0-9][0-9]*\)\/[0-9][0-9]* layers to GPU.*' "${log_file}")"
  offload_total="$(parse_line_number '.*offloaded [0-9][0-9]*\/\([0-9][0-9]*\) layers to GPU.*' "${log_file}")"
  eval_ms="$(parse_line_number '.*eval time = *\([0-9.][0-9.]*\) ms.*' "${log_file}")"
  eval_runs="$(parse_line_number '.*eval time = *[0-9.][0-9.]* ms \/ *\([0-9][0-9]*\) runs.*' "${log_file}")"
  tok_per_s="$(parse_line_number '.*eval time = .*[, ]\([0-9.][0-9.]*\) tokens per second).*' "${log_file}")"
  total_ms="$(parse_line_number '.*total time = *\([0-9.][0-9.]*\) ms.*' "${log_file}")"
  total_tokens="$(parse_line_number '.*total time = *[0-9.][0-9.]* ms \/ *\([0-9][0-9]*\) tokens.*' "${log_file}")"

  gpu_use_avg_pct=""
  gpu_use_max_pct=""
  vram_max_pct=""
  power_max_w=""
  if [[ -s "${smi_file}" ]]; then
    gpu_use_avg_pct="$(awk -F, '{u=$3; gsub(/[^0-9.]/,"",u); if(u!=""){s+=u; n+=1}} END{if(n>0) printf "%.2f", s/n}' "${smi_file}")"
    gpu_use_max_pct="$(awk -F, 'BEGIN{m=0} {u=$3; gsub(/[^0-9.]/,"",u); if((u+0)>m) m=u+0} END{printf "%.2f", m}' "${smi_file}")"
    vram_max_pct="$(awk -F, 'BEGIN{m=0} {v=$4; gsub(/[^0-9.]/,"",v); if((v+0)>m) m=v+0} END{printf "%.2f", m}' "${smi_file}")"
    power_max_w="$(awk -F, 'BEGIN{m=0} {p=$2; gsub(/[^0-9.]/,"",p); if((p+0)>m) m=p+0} END{printf "%.2f", m}' "${smi_file}")"
  fi

  echo "${quant},${status},${rc},${model_path},${offload_layers},${offload_total},${eval_ms},${eval_runs},${tok_per_s},${total_ms},${total_tokens},${gpu_use_avg_pct},${gpu_use_max_pct},${vram_max_pct},${power_max_w}" >> "${OUTPUT_CSV}"

  echo "status=${status} rc=${rc} offload=${offload_layers}/${offload_total} tok/s=${tok_per_s} gpu_max=${gpu_use_max_pct}% vram_max=${vram_max_pct}%"
  rm -f "${log_file}" "${smi_file}"
done

echo ""
echo "Benchmark CSV: ${OUTPUT_CSV}"
