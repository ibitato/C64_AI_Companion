#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PROFILE="${MODEL_PROFILE:-8b}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/inference/prepare_runtime_assets.sh [--model-profile 8b|14b]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-profile)
      MODEL_PROFILE="${2:-}"
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

case "${MODEL_PROFILE}" in
  8b)
    GGUF_DIR="${ROOT_DIR}/models/gguf"
    PREFIX="c64-ministral-3-8b-thinking-c64"
    BASE_MODEL_PATH="models/Ministral-3-8B-Thinking"
    ;;
  14b)
    GGUF_DIR="${ROOT_DIR}/models/gguf-14b"
    PREFIX="c64-ministral-3-14b-thinking-c64"
    BASE_MODEL_PATH="models/Ministral-3-14B-Thinking"
    ;;
  *)
    echo "ERROR: unsupported model profile '${MODEL_PROFILE}'. Use: 8b or 14b" >&2
    exit 1
    ;;
esac

cd "${ROOT_DIR}"
SYSTEM_PROMPT="$(python3 scripts/prompt_contract.py --model-profile "${MODEL_PROFILE}" --base-model-path "${BASE_MODEL_PATH}" --print-full)"

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
