#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' no esta instalado o no esta en PATH" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/inference/prepare_runtime_assets.sh"

create_model() {
  local model_name="$1"
  local modelfile="${GGUF_DIR}/$2"
  if [[ ! -f "${modelfile}" ]]; then
    echo "WARN: falta ${modelfile}, se omite ${model_name}" >&2
    return 0
  fi
  echo "Creando ${model_name} con ${modelfile} ..."
  ollama create "${model_name}" -f "${modelfile}"
}

create_model "c64-ministral-c64" "Modelfile.Q4_K_M"
create_model "c64-ministral-c64-q6" "Modelfile.Q6_K"
create_model "c64-ministral-c64-q8" "Modelfile.Q8_0"

echo ""
echo "Modelos Ollama disponibles para este proyecto:"
ollama list | grep -E '^c64-ministral-c64(-q6|-q8)?\b' || true
