#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOODR1_ROOT="${FOODR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  FOODR1_ENV=local.env bash scripts/train_grpo.sh [--caloriebench|--auxiliary]

Default:
  --caloriebench
EOF
}

TARGET="caloriebench"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--caloriebench" ]]; then
  TARGET="caloriebench"
elif [[ "${1:-}" == "--auxiliary" ]]; then
  TARGET="auxiliary"
elif [[ -n "${1:-}" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "${FOODR1_ENV:-}" && -f "${FOODR1_ROOT}/local.env" ]]; then
  export FOODR1_ENV="${FOODR1_ROOT}/local.env"
fi

if [[ -z "${FOODR1_ENV:-}" ]]; then
  echo "Set FOODR1_ENV=local.env or create ${FOODR1_ROOT}/local.env." >&2
  exit 1
fi

case "${TARGET}" in
  caloriebench)
    FOODR1_ROOT="${FOODR1_ROOT}" FOODR1_ENV="${FOODR1_ENV}" bash "${SCRIPT_DIR}/train_grpo_caloriebench80k.sh"
    ;;
  auxiliary)
    FOODR1_ROOT="${FOODR1_ROOT}" FOODR1_ENV="${FOODR1_ENV}" bash "${SCRIPT_DIR}/train_grpo_auxiliary.sh"
    ;;
esac
