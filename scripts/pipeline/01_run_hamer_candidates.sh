#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${DATASET:?Set DATASET, e.g. ropedia or wuxi}"
: "${VIDEO:?Set VIDEO to the left/RGB video path}"
: "${DEPTH_DIR:?Set DEPTH_DIR to FoundationStereo depth output dir}"

HAMER_OUT="${HAMER_OUT:-tmp/${DATASET}_hamer_ego_filter_${START}_$((START + NUM_FRAMES - 1))_tuned}"

ARGS=(
  --depth_dir "${DEPTH_DIR}"
  --output_dir "${HAMER_OUT}"
  --hamer_root "${HAMER_ROOT}"
  --start "${START}"
  --num_frames "${NUM_FRAMES}"
  --body_detector "${HAMER_BODY_DETECTOR}"
  --batch_size "${HAMER_BATCH_SIZE}"
  --save_mesh
  --recovery_min_area_ratio "${RECOVERY_MIN_AREA_RATIO}"
  --recovery_min_center_y_ratio "${RECOVERY_MIN_CENTER_Y_RATIO}"
  --recovery_max_median_depth_m "${RECOVERY_MAX_MEDIAN_DEPTH_M}"
)

if [[ -n "${DATASET_MANIFEST:-}" ]]; then
  ARGS+=(--manifest "${DATASET_MANIFEST}")
else
  ARGS+=(--video "${VIDEO}")
fi

if [[ -n "${DEPTH_QUALITY_REPORT:-}" && -f "${DEPTH_QUALITY_REPORT}" ]]; then
  ARGS+=(--depth_quality_report "${DEPTH_QUALITY_REPORT}")
fi

python scripts/hand/run_hamer_ego_filter.py \
  "${ARGS[@]}"

echo "HAMER_OUT=${HAMER_OUT}"
