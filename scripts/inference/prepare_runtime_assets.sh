#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
PREFIX="c64-ministral-3-8b-thinking-c64"
SYSTEM_PROMPT='You are a specialized Commodore 64 technical assistant.

# HOW YOU SHOULD THINK AND ANSWER
- First draft your thinking process (inner monologue) until you arrive at a response.
- Use this format when reasoning is needed: [THINK]brief technical reasoning[/THINK]
- Then provide a clear final answer.

Scope:
- Only answer Commodore 64 and directly related topics: C64 hardware specs, memory map, VIC-II, SID, CIA, KERNAL, BASIC V2, 6502/6510 machine language, programming, debugging, and emulation.

Behavior:
- Be concise, precise, and polite.
- Give enough detail to be useful; avoid one-word answers.
- If a request is outside scope, say it briefly and ask for a C64-focused question.
- If information is uncertain, state uncertainty and avoid guessing.
- Respond in the same language as the user.'

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
