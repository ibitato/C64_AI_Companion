#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
LLAMA_BIN="${LLAMA_BIN:-${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-cli}"

QUANT="${1:-Q8_0}"
PROMPT="${2:-Briefly explain what the Commodore 64 SID chip does.}"
shift || true
shift || true
EXTRA_ARGS=("$@")
SINGLE_TURN=1

# Normalize short aliases to canonical quantization names.
case "${QUANT^^}" in
  Q4|Q4_K_M) QUANT="Q4_K_M" ;;
  Q6|Q6_K) QUANT="Q6_K" ;;
  Q8|Q8_0) QUANT="Q8_0" ;;
  F16) QUANT="F16" ;;
  *)
    echo "ERROR: unsupported quantization '${QUANT}'. Use: Q4_K_M, Q6_K, Q8_0, F16" >&2
    exit 1
    ;;
esac

MODEL_PATH="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${QUANT}.gguf"

if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "ERROR: llama.cpp executable not found at '${LLAMA_BIN}'" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model file not found at '${MODEL_PATH}'" >&2
  exit 1
fi

echo "Using model: ${MODEL_PATH}"
echo "Executable: ${LLAMA_BIN}"

filtered_args=()
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --multi-turn)
      SINGLE_TURN=0
      ;;
    --single-turn)
      SINGLE_TURN=1
      ;;
    *)
      filtered_args+=("${arg}")
      ;;
  esac
done
EXTRA_ARGS=("${filtered_args[@]}")

set_n_predict=1
set_reasoning_format=1
set_reasoning_budget=1
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "-n" || "${arg}" == "--predict" || "${arg}" == "--n-predict" ]]; then
    set_n_predict=0
  fi
  if [[ "${arg}" == "--reasoning-format" ]]; then
    set_reasoning_format=0
  fi
  if [[ "${arg}" == "--reasoning-budget" ]]; then
    set_reasoning_budget=0
  fi
done

cmd=(
  "${LLAMA_BIN}"
  -m "${MODEL_PATH}"
  -ngl 99
  -c 4096
  -p "${PROMPT}"
)

if [[ "${SINGLE_TURN}" -eq 1 ]]; then
  cmd+=(-st)
fi

if [[ "${set_n_predict}" -eq 1 ]]; then
  cmd+=(-n 256)
fi

if [[ "${set_reasoning_format}" -eq 1 ]]; then
  cmd+=(--reasoning-format "${LLAMA_REASONING_FORMAT:-none}")
fi

if [[ "${set_reasoning_budget}" -eq 1 ]]; then
  cmd+=(--reasoning-budget "${LLAMA_REASONING_BUDGET:--1}")
fi

cmd+=("${EXTRA_ARGS[@]}")
exec "${cmd[@]}"
