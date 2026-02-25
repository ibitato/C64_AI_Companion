#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
QBIN="${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-quantize"
LLAMA_BIN_DIR="${ROOT_DIR}/.cache/llama.cpp/build/bin"
BASE_F16="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-F16.gguf"

if [[ ! -x "${QBIN}" ]]; then
  echo "ERROR: '${QBIN}' is missing. Build llama.cpp first." >&2
  exit 1
fi

# llama.cpp shared libraries are colocated with binaries in build/bin.
# Ensure the dynamic linker can resolve libllama/libggml at runtime.
export LD_LIBRARY_PATH="${LLAMA_BIN_DIR}:${LD_LIBRARY_PATH:-}"
# Add common ROCm runtime locations when present on host.
for rocm_lib_dir in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib /opt/rocm-*/lib64; do
  if [[ -d "${rocm_lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${rocm_lib_dir}:${LD_LIBRARY_PATH}"
  fi
done

if [[ ! -f "${BASE_F16}" ]]; then
  echo "ERROR: '${BASE_F16}' is missing. Export F16 GGUF first." >&2
  exit 1
fi

probe_quantize_runtime() {
  local probe_output
  probe_output="$("${QBIN}" --help 2>&1 || true)"
  if grep -q "error while loading shared libraries" <<< "${probe_output}"; then
    echo "${probe_output}" >&2

    # If host lacks ROCm userland libs, re-run inside canonical project container.
    if [[ ! -f "/.dockerenv" ]] && command -v docker >/dev/null 2>&1; then
      if docker compose version >/dev/null 2>&1; then
        echo "INFO: missing runtime libs on host. Re-running inside 'trainer' container..." >&2
        exec docker compose run --rm trainer bash scripts/inference/quantize_additional_gguf.sh
      fi
    fi

    echo "ERROR: llama-quantize runtime dependencies are missing." >&2
    echo "Hint: run inside the trainer container:" >&2
    echo "  docker compose run --rm trainer bash scripts/inference/quantize_additional_gguf.sh" >&2
    exit 1
  fi
}

probe_quantize_runtime

quantize_if_missing() {
  local quant="$1"
  local out_file="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${quant}.gguf"
  if [[ -f "${out_file}" ]]; then
    echo "SKIP: ${out_file} already exists"
    return 0
  fi
  echo "Generating ${out_file} ..."
  "${QBIN}" "${BASE_F16}" "${out_file}" "${quant}"
}

quantize_if_missing "Q6_K"
quantize_if_missing "Q8_0"

echo ""
echo "Available GGUF files:"
ls -lh "${GGUF_DIR}"/*.gguf
