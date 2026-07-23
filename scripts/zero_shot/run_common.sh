#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/zero_shot/run_common.sh BACKBONE DATASET [GPU] [EXTRA_ARGS...]" >&2
  echo "Backbones: ulip, openshape, uni3d" >&2
  echo "Datasets: modelnet, modelnet_c, scanobjectnn, scanobjectnn_c" >&2
  exit 1
fi

BACKBONE="$1"
DATASET="$2"
GPU="${3:-0}"
shift 2
if [[ "$#" -gt 0 ]]; then
  shift 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${GPU}" != "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  DEVICE="cuda:0"
else
  DEVICE="cpu"
fi

python runners/zero_shot.py \
  --backbone "${BACKBONE}" \
  --dataset "${DATASET}" \
  --device "${DEVICE}" \
  "$@"
