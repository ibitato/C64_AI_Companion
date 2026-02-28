#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
PREFIX="c64-ministral-3-8b-thinking-c64"

cd "${ROOT_DIR}"
SYSTEM_PROMPT="$(python3 scripts/prompt_contract.py --base-model-path models/Ministral-3-8B-Thinking --print-full)"

mkdir -p "${GGUF_DIR}"

write_modelfile() {
  local quant="$1"
  local gguf_file="${PREFIX}-${quant}.gguf"
  local modelfile="${GGUF_DIR}/Modelfile.${quant}"

  if [[ ! -f "${GGUF_DIR}/${gguf_file}" ]]; then
    echo "WARN: ${GGUF_DIR}/${gguf_file} is missing, skipping ${modelfile}" >&2
    return 0
  fi

  cat > "${modelfile}" <<EOF
FROM ./${gguf_file}
SYSTEM \"\"\"${SYSTEM_PROMPT}\"\"\"
EOF
  echo "OK: ${modelfile}"
}

write_modelfile "Q4_K_M"
write_modelfile "Q6_K"
write_modelfile "Q8_0"
write_modelfile "F16"

if [[ -f "${GGUF_DIR}/Modelfile.Q8_0" ]]; then
  cp "${GGUF_DIR}/Modelfile.Q8_0" "${GGUF_DIR}/Modelfile"
  echo "OK: ${GGUF_DIR}/Modelfile (alias Q8_0)"
elif [[ -f "${GGUF_DIR}/Modelfile.Q4_K_M" ]]; then
  cp "${GGUF_DIR}/Modelfile.Q4_K_M" "${GGUF_DIR}/Modelfile"
  echo "OK: ${GGUF_DIR}/Modelfile (fallback alias Q4_K_M)"
fi

echo ""
echo "Done. Available Modelfiles:"
ls -1 "${GGUF_DIR}"/Modelfile* 2>/dev/null || true
