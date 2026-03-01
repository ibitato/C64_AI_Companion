#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PROFILE="${MODEL_PROFILE:-8b}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/inference/create_ollama_models.sh [--model-profile 8b|14b]
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
    OLLAMA_BASE="c64-ministral-c64"
    ;;
  14b)
    GGUF_DIR="${ROOT_DIR}/models/gguf-14b"
    OLLAMA_BASE="c64-ministral-c64-14b"
    ;;
  *)
    echo "ERROR: unsupported model profile '${MODEL_PROFILE}'. Use: 8b or 14b" >&2
    exit 1
    ;;
esac

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' is not installed or not in PATH" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/inference/prepare_runtime_assets.sh" --model-profile "${MODEL_PROFILE}"

create_model() {
  local model_name="$1"
  local modelfile="${GGUF_DIR}/$2"
  if [[ ! -f "${modelfile}" ]]; then
    echo "WARN: missing ${modelfile}, skipping ${model_name}" >&2
    return 0
  fi
  echo "Creating ${model_name} from ${modelfile} ..."
  ollama create "${model_name}" -f "${modelfile}"
}

if [[ -f "${GGUF_DIR}/Modelfile.Q8_0" ]]; then
  create_model "${OLLAMA_BASE}" "Modelfile.Q8_0"
else
  create_model "${OLLAMA_BASE}" "Modelfile.Q4_K_M"
fi
create_model "${OLLAMA_BASE}-q4" "Modelfile.Q4_K_M"
create_model "${OLLAMA_BASE}-q6" "Modelfile.Q6_K"
create_model "${OLLAMA_BASE}-q8" "Modelfile.Q8_0"

echo ""
echo "Ollama models available for this project:"
ollama list | grep -E "^${OLLAMA_BASE}(-q4|-q6|-q8)?\\b" || true
