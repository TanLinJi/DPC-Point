#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_ulip.sh" "${GPU}"
bash "${SCRIPT_DIR}/run_openshape.sh" "${GPU}"
bash "${SCRIPT_DIR}/run_uni3d.sh" "${GPU}"
