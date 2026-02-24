#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
QBIN="${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-quantize"
BASE_F16="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-F16.gguf"

if [[ ! -x "${QBIN}" ]]; then
  echo "ERROR: no existe '${QBIN}'. Compila llama.cpp primero." >&2
  exit 1
fi

if [[ ! -f "${BASE_F16}" ]]; then
  echo "ERROR: no existe '${BASE_F16}'. Exporta primero el GGUF F16." >&2
  exit 1
fi

quantize_if_missing() {
  local quant="$1"
  local out_file="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${quant}.gguf"
  if [[ -f "${out_file}" ]]; then
    echo "SKIP: ${out_file} ya existe"
    return 0
  fi
  echo "Generando ${out_file} ..."
  "${QBIN}" "${BASE_F16}" "${out_file}" "${quant}"
}

quantize_if_missing "Q6_K"
quantize_if_missing "Q8_0"

echo ""
echo "GGUF disponibles:"
ls -lh "${GGUF_DIR}"/*.gguf
