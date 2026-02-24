#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
PREFIX="c64-ministral-3-8b-thinking-c64"

mkdir -p "${GGUF_DIR}"

write_modelfile() {
  local quant="$1"
  local gguf_file="${PREFIX}-${quant}.gguf"
  local modelfile="${GGUF_DIR}/Modelfile.${quant}"

  if [[ ! -f "${GGUF_DIR}/${gguf_file}" ]]; then
    echo "WARN: no existe ${GGUF_DIR}/${gguf_file}, se omite ${modelfile}" >&2
    return 0
  fi

  printf 'FROM ./%s\n' "${gguf_file}" > "${modelfile}"
  echo "OK: ${modelfile}"
}

write_modelfile "Q4_K_M"
write_modelfile "Q6_K"
write_modelfile "Q8_0"
write_modelfile "F16"

if [[ -f "${GGUF_DIR}/Modelfile.Q4_K_M" ]]; then
  cp "${GGUF_DIR}/Modelfile.Q4_K_M" "${GGUF_DIR}/Modelfile"
  echo "OK: ${GGUF_DIR}/Modelfile (alias Q4_K_M)"
fi

echo ""
echo "Listo. Modelfiles disponibles:"
ls -1 "${GGUF_DIR}"/Modelfile* 2>/dev/null || true
