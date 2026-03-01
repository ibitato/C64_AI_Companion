#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF_DIR="${ROOT_DIR}/models/gguf"
LLAMA_BIN="${LLAMA_BIN:-${ROOT_DIR}/.cache/llama.cpp/build/bin/llama-cli}"

QUANT="${1:-Q8_0}"
PROMPT="${2:-Briefly explain what the Commodore 64 SID chip does.}"
shift || true
shift || true
EXTRA_ARGS=("$@")
SINGLE_TURN=1
set_chat_mode=1
set_jinja_mode=1
set_special_mode=1
set_system_prompt=1
DEFAULT_SYSTEM_PROMPT_FILE="${ROOT_DIR}/.cache/runtime/c64_system_prompt.txt"

ensure_contract_system_prompt_file() {
  local out_file="${DEFAULT_SYSTEM_PROMPT_FILE}"
  mkdir -p "$(dirname "${out_file}")"
  if [[ ! -s "${out_file}" || "${LLAMA_REFRESH_SYSTEM_PROMPT:-0}" == "1" ]]; then
    python3 "${ROOT_DIR}/scripts/prompt_contract.py" --print-full > "${out_file}"
  fi
  if [[ ! -s "${out_file}" ]]; then
    echo "ERROR: failed to prepare system prompt file at '${out_file}'" >&2
    exit 1
  fi
  echo "${out_file}"
}

# Normalize short aliases to canonical quantization names.
case "${QUANT^^}" in
  Q4|Q4_K_M) QUANT="Q4_K_M" ;;
  Q6|Q6_K) QUANT="Q6_K" ;;
  Q8|Q8_0) QUANT="Q8_0" ;;
  F16) QUANT="F16" ;;
  *)
    echo "ERROR: unsupported quantization '${QUANT}'. Use: Q4_K_M, Q6_K, Q8_0, F16" >&2
    exit 1
    ;;
esac

MODEL_PATH="${GGUF_DIR}/c64-ministral-3-8b-thinking-c64-${QUANT}.gguf"

if [[ ! -x "${LLAMA_BIN}" ]]; then
  echo "ERROR: llama.cpp executable not found at '${LLAMA_BIN}'" >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model file not found at '${MODEL_PATH}'" >&2
  exit 1
fi

echo "Using model: ${MODEL_PATH}"
echo "Executable: ${LLAMA_BIN}"

filtered_args=()
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --multi-turn)
      SINGLE_TURN=0
      ;;
    --single-turn)
      SINGLE_TURN=1
      ;;
    -cnv|--conversation|-no-cnv|--no-conversation)
      set_chat_mode=0
      filtered_args+=("${arg}")
      ;;
    --jinja|--no-jinja)
      set_jinja_mode=0
      filtered_args+=("${arg}")
      ;;
    -sp|--special)
      set_special_mode=0
      filtered_args+=("${arg}")
      ;;
    -sys|--system-prompt|-sysf|--system-prompt-file|--system-prompt=*|--system-prompt-file=*)
      set_system_prompt=0
      filtered_args+=("${arg}")
      ;;
    *)
      filtered_args+=("${arg}")
      ;;
  esac
done
EXTRA_ARGS=("${filtered_args[@]}")

set_n_predict=1
set_reasoning_format=1
set_reasoning_budget=1
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "-n" || "${arg}" == "--predict" || "${arg}" == "--n-predict" ]]; then
    set_n_predict=0
  fi
  if [[ "${arg}" == "--reasoning-format" ]]; then
    set_reasoning_format=0
  fi
  if [[ "${arg}" == "--reasoning-budget" ]]; then
    set_reasoning_budget=0
  fi
done

cmd=(
  "${LLAMA_BIN}"
  -m "${MODEL_PATH}"
  -ngl 99
  -c 4096
  -p "${PROMPT}"
)

if [[ "${SINGLE_TURN}" -eq 1 ]]; then
  cmd+=(-st)
fi

if [[ "${set_chat_mode}" -eq 1 ]]; then
  case "${LLAMA_CHAT_MODE:-cnv}" in
    cnv)
      cmd+=(-cnv)
      ;;
    no-cnv)
      cmd+=(-no-cnv)
      ;;
    *)
      echo "ERROR: invalid LLAMA_CHAT_MODE='${LLAMA_CHAT_MODE}'. Use 'cnv' or 'no-cnv'." >&2
      exit 1
      ;;
  esac
fi

if [[ "${set_jinja_mode}" -eq 1 ]]; then
  if [[ "${LLAMA_USE_JINJA:-1}" == "1" ]]; then
    cmd+=(--jinja)
  else
    cmd+=(--no-jinja)
  fi
fi

if [[ "${set_special_mode}" -eq 1 && "${LLAMA_SHOW_SPECIAL:-1}" == "1" ]]; then
  cmd+=(-sp)
fi

if [[ "${set_system_prompt}" -eq 1 && "${LLAMA_USE_CONTRACT_PROMPT:-1}" == "1" ]]; then
  cmd+=(-sysf "$(ensure_contract_system_prompt_file)")
fi

if [[ "${set_n_predict}" -eq 1 ]]; then
  cmd+=(-n 256)
fi

if [[ "${set_reasoning_format}" -eq 1 ]]; then
  cmd+=(--reasoning-format "${LLAMA_REASONING_FORMAT:-none}")
fi

if [[ "${set_reasoning_budget}" -eq 1 ]]; then
  cmd+=(--reasoning-budget "${LLAMA_REASONING_BUDGET:--1}")
fi

cmd+=("${EXTRA_ARGS[@]}")
exec "${cmd[@]}"
