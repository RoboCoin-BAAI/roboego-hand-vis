#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${DATASET:?Set DATASET, e.g. ropedia or wuxi}"
: "${DATASET_MANIFEST:?Set DATASET_MANIFEST}"
: "${CANDIDATES_JSON:?Set CANDIDATES_JSON}"
: "${DEPTH_DIR:?Set DEPTH_DIR}"

ALIGN_OUT="${ALIGN_OUT:-tmp/${DATASET}_hand_depth_alignment_${START}_$((START + NUM_FRAMES - 1))}"

ORIENTATION_ARGS=(--orientation_aware_compensation)
if [[ "${ORIENTATION_AWARE_COMPENSATION}" == "0" ]]; then
  ORIENTATION_ARGS=(--no-orientation_aware_compensation)
fi

RRD_ARGS=(--no_save_rrd)
if [[ "${ALIGN_SAVE_RRD:-0}" == "1" ]]; then
  RRD_ARGS=(--save_rrd "${ALIGN_RRD:-${ALIGN_OUT}/debug_alignment.rrd}")
fi

HAMER_OVERLAY_ARGS=()
if [[ -n "${HAMER_OVERLAY_DIR:-}" ]]; then
  HAMER_OVERLAY_ARGS+=(--hamer_overlay_dir "${HAMER_OVERLAY_DIR}")
fi

python scripts/hand/align_mesh_depth.py \
  --manifest "${DATASET_MANIFEST}" \
  --candidates_json "${CANDIDATES_JSON}" \
  --depth_dir "${DEPTH_DIR}" \
  --output_dir "${ALIGN_OUT}" \
  --start "${START}" \
  --num_frames "${NUM_FRAMES}" \
  "${HAMER_OVERLAY_ARGS[@]}" \
  "${RRD_ARGS[@]}" \
  --scene_point_stride "${SCENE_POINT_STRIDE}" \
  --max_scene_points "${MAX_SCENE_POINTS}" \
  --mesh_mask_source "${MESH_MASK_SOURCE}" \
  --mesh_mask_padding_px "${MESH_MASK_PADDING_PX}" \
  --overlay_diff_threshold "${OVERLAY_DIFF_THRESHOLD}" \
  --overlay_diff_search_margin_px "${OVERLAY_DIFF_SEARCH_MARGIN_PX}" \
  --mask_depth_band_half_width_m "${MASK_DEPTH_BAND_HALF_WIDTH_M}" \
  --shape_preserved_depth_source "${SHAPE_PRESERVED_DEPTH_SOURCE}" \
  "${ORIENTATION_ARGS[@]}" \
  --mano_model_dir "${MANO_MODEL_DIR}" \
  --orientation_depth_bias_m "${ORIENT_DEPTH_BIAS_M}" \
  --orientation_surface_min_m "${ORIENT_SURFACE_MIN_M}" \
  --orientation_surface_max_m "${ORIENT_SURFACE_MAX_M}" \
  --orientation_ray_blend "${ORIENT_RAY_BLEND}" \
  --depth_lift_window_radius_px "${DEPTH_LIFT_WINDOW_RADIUS_PX}" \
  --depth_lift_max_delta_m "${DEPTH_LIFT_MAX_DELTA_M}"

echo "ALIGN_OUT=${ALIGN_OUT}"
echo "ALIGNMENT_REPORT=${ALIGN_OUT}/hand_depth_mesh_alignment_report.json"
