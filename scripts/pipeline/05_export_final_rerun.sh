#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

: "${TRACKS_JSON:?Set TRACKS_JSON}"
: "${ALIGNMENT_REPORT:?Set ALIGNMENT_REPORT}"
: "${DEPTH_DIR:?Set DEPTH_DIR}"

RRD_OUT="${RRD_OUT:-tmp/${DATASET}_final_hand_vis_${START}_$((START + NUM_FRAMES - 1))}"
SAVE_RRD="${SAVE_RRD:-${RRD_OUT}/${DATASET}_final_rgb_pointcloud_mesh.rrd}"

RRD_START="${RRD_START:-${START}}"
RRD_NUM_FRAMES="${RRD_NUM_FRAMES:-${NUM_FRAMES}}"
FINAL_SMOOTHING_WINDOW="${FINAL_SMOOTHING_WINDOW:-5}"
FINAL_SCENE_POINT_STRIDE="${FINAL_SCENE_POINT_STRIDE:-8}"
FINAL_MAX_SCENE_POINTS="${FINAL_MAX_SCENE_POINTS:-30000}"

python scripts/hand/export_final_rerun.py \
  --tracks_json "${TRACKS_JSON}" \
  --alignment_report "${ALIGNMENT_REPORT}" \
  --depth_dir "${DEPTH_DIR}" \
  --output_rrd "${SAVE_RRD}" \
  --start "${START}" \
  --num_frames "${NUM_FRAMES}" \
  --rrd_start "${RRD_START}" \
  --rrd_num_frames "${RRD_NUM_FRAMES}" \
  --final_smoothing_window "${FINAL_SMOOTHING_WINDOW}" \
  --scene_point_stride "${FINAL_SCENE_POINT_STRIDE}" \
  --max_scene_points "${FINAL_MAX_SCENE_POINTS}" \
  --mano_model_dir "${MANO_MODEL_DIR}"

echo "RRD=${SAVE_RRD}"
