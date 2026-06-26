#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${DATASET:?Set DATASET, e.g. ropedia or wuxi}"
: "${HAMER_CANDIDATES_JSON:?Set HAMER_CANDIDATES_JSON}"

RECOVER_OUT="${RECOVER_OUT:-tmp/${DATASET}_hamer_ego_filter_${START}_$((START + NUM_FRAMES - 1))_recovered_area0007_temporal_gated}"

python scripts/recover_ego_hand_candidates.py \
  --input_json "${HAMER_CANDIDATES_JSON}" \
  --output_dir "${RECOVER_OUT}" \
  --recovery_min_area_ratio "${RECOVERY_MIN_AREA_RATIO}" \
  --recovery_min_center_y_ratio "${RECOVERY_MIN_CENTER_Y_RATIO}" \
  --recovery_max_median_depth_m "${RECOVERY_MAX_MEDIAN_DEPTH_M}"

echo "RECOVER_OUT=${RECOVER_OUT}"
echo "RECOVERED_CANDIDATES_JSON=${RECOVER_OUT}/ego_hand_candidates_recovered.json"
