#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
LLAMA_BIN="${LLAMA_BIN:-${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-cli}"

QUANT="${1:-Q4_K_M}"
PROMPT="${2:-Explica brevemente que es el chip SID del Commodore 64.}"
shift || true
shift || true
EXTRA_ARGS=("$@")

case "${QUANT^^}" in
  Q4|Q4_K_M) QUANT="Q4_K_M" ;;
  Q6|Q6_K) QUANT="Q6_K" ;;
  Q8|Q8_0) QUANT="Q8_0" ;;
  F16) QUANT="F16" ;;
  *)
    echo "ERROR: cuantizacion no soportada '${QUANT}'. Usa: Q4_K_M, Q6_K, Q8_0, F16" >&2
    exit 1
    ;;
esac

MODEL_PATH="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${QUANT}.gguf"

if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "ERROR: no existe ejecutable llama.cpp en '${LLAMA_BIN}'" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: no existe modelo '${MODEL_PATH}'" >&2
  exit 1
fi

echo "Usando modelo: ${MODEL_PATH}"
echo "Binario: ${LLAMA_BIN}"

set_n_predict=1
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "-n" || "${arg}" == "--predict" || "${arg}" == "--n-predict" ]]; then
    set_n_predict=0
    break
  fi
done

cmd=(
  "${LLAMA_BIN}"
  -m "${MODEL_PATH}"
  -ngl 99
  -st
  -c 4096
  -p "${PROMPT}"
)

if [[ "${set_n_predict}" -eq 1 ]]; then
  cmd+=(-n 256)
fi

cmd+=("${EXTRA_ARGS[@]}")
exec "${cmd[@]}"
