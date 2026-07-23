#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU="${1:-0}"
shift || true
DATASETS=("$@")
if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  DATASETS=(modelnet modelnet_c scanobjectnn scanobjectnn_c)
fi

for dataset in "${DATASETS[@]}"; do
  bash "${SCRIPT_DIR}/run_common.sh" uni3d "${dataset}" "${GPU}"
done
