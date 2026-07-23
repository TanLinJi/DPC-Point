#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"

DATASETS=(
  "modelnet"
  "modelnet_c"
  "scanobjectnn"
  "scanobjectnn_c"
)

COMBINATIONS=(
  "3,3,3,6|best:4.4,3.9,0.19"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

severity_for_dataset() {
  case "$1" in
    modelnet_c|scanobjectnn_c) echo "all35" ;;
    *) echo "clean" ;;
  esac
}

variant_for_dataset() {
  case "$1" in
    scanobjectnn|scanobjectnn_c) echo "hardest" ;;
    *) echo "standard" ;;
  esac
}

for DATASET in "${DATASETS[@]}"; do
  SEVERITY_SET="$(severity_for_dataset "${DATASET}")"
  VARIANT="$(variant_for_dataset "${DATASET}")"
  for COMBINATION in "${COMBINATIONS[@]}"; do
    IFS='|' read -r CACHE_PART SCORE_PART <<< "${COMBINATION}"
    IFS=',' read -r ENTROPY_CAP GPA_CAP LOCAL_CAP NEG_CAP <<< "${CACHE_PART}"
    SCORE_NAME="${SCORE_PART%%:*}"
    EXP_NAME="uni3d_${DATASET}_${VARIANT}_${SEVERITY_SET}_e${ENTROPY_CAP}_g${GPA_CAP}_l${LOCAL_CAP}_n${NEG_CAP}_${SCORE_NAME}"

    bash "${SCRIPT_DIR}/run_common.sh" uni3d "${DATASET}" "${GPU}" \
      --severity-set "${SEVERITY_SET}" \
      --entropy-cap "${ENTROPY_CAP}" \
      --gpa-cap "${GPA_CAP}" \
      --local-cap "${LOCAL_CAP}" \
      --neg-cap "${NEG_CAP}" \
      --local-centers 3 \
      --final-score-weights "${SCORE_PART}" \
      --exp-name "${EXP_NAME}"
  done
done
