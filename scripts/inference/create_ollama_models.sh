#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' is not installed or not in PATH" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/inference/prepare_runtime_assets.sh"

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

create_model "c64-ministral-c64" "Modelfile.Q4_K_M"
create_model "c64-ministral-c64-q6" "Modelfile.Q6_K"
create_model "c64-ministral-c64-q8" "Modelfile.Q8_0"

echo ""
echo "Ollama models available for this project:"
ollama list | grep -E '^c64-ministral-c64(-q6|-q8)?\b' || true
