#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${DATASET:?Set DATASET, e.g. ropedia or wuxi}"
: "${ALIGNMENT_REPORT:?Set ALIGNMENT_REPORT}"
: "${DEPTH_DIR:?Set DEPTH_DIR}"

TRACK_OUT="${TRACK_OUT:-tmp/${DATASET}_ego_hand_tracking_${START}_$((START + NUM_FRAMES - 1))}"

ARGS=(
  --alignment_report "${ALIGNMENT_REPORT}"
  --depth_dir "${DEPTH_DIR}"
  --output_dir "${TRACK_OUT}"
  --start "${START}"
  --num_frames "${NUM_FRAMES}"
  --max_interp_gap "${MAX_INTERP_GAP}"
  --smooth_alpha "${SMOOTH_ALPHA}"
  --jump_threshold_m "${JUMP_THRESHOLD_M}"
  --mesh_error_threshold_m "${MESH_ERROR_THRESHOLD_M}"
  --min_valid_depth_ratio "${MIN_VALID_DEPTH_RATIO}"
  --no-drop_unstable_recovered
  --unstable_recovered_jump_threshold_m "${JUMP_THRESHOLD_M}"
  --unstable_recovered_max_area_ratio 0.001
  --apply_recovered_depth_overrides
  --scene_point_stride 6
  --max_scene_points 50000
)

if [[ "${TRACK_SAVE_RRD:-0}" == "1" ]]; then
  ARGS+=(--save_rrd "${TRACK_RRD:-${TRACK_OUT}/debug_tracks.rrd}")
else
  ARGS+=(--no_save_rrd)
fi

if [[ -n "${DATASET_MANIFEST:-}" ]]; then
  ARGS+=(--manifest "${DATASET_MANIFEST}")
else
  : "${VIDEO:?Set VIDEO or DATASET_MANIFEST}"
  ARGS+=(--video "${VIDEO}")
fi

python scripts/hand/track_fill_smooth.py "${ARGS[@]}"

echo "TRACK_OUT=${TRACK_OUT}"
echo "TRACKS_JSON=${TRACK_OUT}/ego_hand_tracks.json"
