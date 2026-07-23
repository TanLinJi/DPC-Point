#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "$#" -eq 0 ]]; then
  DATASETS=(modelnet_c scanobjectnn_c)
else
  DATASETS=("$@")
fi

run_one() {
  local dataset="$1"
  case "${dataset}" in
    modelnet)
      python scripts/download/download_data_modelnet.py
      ;;
    modelnet_c)
      python scripts/download/download_data_modelnet_c.py
      ;;
    scanobjectnn)
      python scripts/download/download_data_scanobjectnn.py
      ;;
    scanobjectnn_c)
      python scripts/download/download_data_scanobjectnn_c.py
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      echo "Available: modelnet, modelnet_c, scanobjectnn, scanobjectnn_c" >&2
      return 1
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  echo
  echo ">>> Download dataset: ${dataset}"
  run_one "${dataset}"
done

echo
echo "All requested dataset files have been downloaded."
