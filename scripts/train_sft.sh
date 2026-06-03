#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOODR1_ROOT="${FOODR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ -z "${FOODR1_ENV:-}" && -f "${FOODR1_ROOT}/local.env" ]]; then
  export FOODR1_ENV="${FOODR1_ROOT}/local.env"
fi

if [[ -z "${FOODR1_ENV:-}" ]]; then
  echo "Set FOODR1_ENV=local.env or create ${FOODR1_ROOT}/local.env." >&2
  exit 1
fi

FOODR1_ROOT="${FOODR1_ROOT}" FOODR1_ENV="${FOODR1_ENV}" bash "${SCRIPT_DIR}/train_sft_megatron.sh"
