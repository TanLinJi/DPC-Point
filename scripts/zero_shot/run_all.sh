#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU="${1:-0}"

bash "${SCRIPT_DIR}/run_ulip.sh" "${GPU}"
bash "${SCRIPT_DIR}/run_openshape.sh" "${GPU}"
bash "${SCRIPT_DIR}/run_uni3d.sh" "${GPU}"
