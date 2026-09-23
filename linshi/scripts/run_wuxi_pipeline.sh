#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/pipeline/common.sh"

export DATASET="${DATASET:-wuxi}"
export START="${START:-0}"
export NUM_FRAMES="${NUM_FRAMES:-300}"
export VIDEO="${VIDEO:-./data/wuxi/stereo_left.mp4}"
export RIGHT_VIDEO="${RIGHT_VIDEO:-./data/wuxi/stereo_right.mp4}"
export INTRINSICS_JSON="${INTRINSICS_JSON:-./data/wuxi/camera_intrinsics.json}"
export INTRINSICS_CAMERA="${INTRINSICS_CAMERA:-left}"
export INTRINSICS_RESIZE_SCALE="${INTRINSICS_RESIZE_SCALE:-1.0}"
export DATASET_MANIFEST="${DATASET_MANIFEST:-tmp/wuxi_dataset_manifest/dataset_manifest.json}"
export DEPTH_DIR="${DEPTH_DIR:-tmp/wuxi_foundation_depth_continuous_000000_000299}"
export DEPTH_QUALITY_REPORT="${DEPTH_QUALITY_REPORT:-tmp/wuxi_depth_quality_000000_000299/depth_quality_report.json}"

export CANDIDATES_JSON="${CANDIDATES_JSON:-tmp/wuxi_egohandkit/ego_hand_candidates.json}"
export HAMER_OVERLAY_DIR="${HAMER_OVERLAY_DIR:-}"
export ALIGN_OUT="${ALIGN_OUT:-tmp/wuxi_hand_depth_alignment_000000_000299}"
export ALIGNMENT_REPORT="${ALIGNMENT_REPORT:-${ALIGN_OUT}/hand_depth_mesh_alignment_report.json}"
export TRACK_OUT="${TRACK_OUT:-tmp/wuxi_ego_hand_tracking_000000_000299}"
export TRACKS_JSON="${TRACKS_JSON:-${TRACK_OUT}/ego_hand_tracks.json}"
export RRD_OUT="${RRD_OUT:-tmp/wuxi_hand_vis_000000_000299}"
export SAVE_RRD="${SAVE_RRD:-${RRD_OUT}/wuxi_hand_vis.rrd}"

RUN_FOUNDATION="${RUN_FOUNDATION:-1}"
RUN_ALIGN="${RUN_ALIGN:-1}"
RUN_TRACK="${RUN_TRACK:-1}"
RUN_RRD="${RUN_RRD:-1}"

FOUNDATION_ROOT="${FOUNDATION_ROOT:-../FoundationStereo}"
FOUNDATION_CKPT="${FOUNDATION_CKPT:-${FOUNDATION_ROOT}/pretrained_models/23-51-11/model_best_bp2.pth}"
FOUNDATION_SCALE="${FOUNDATION_SCALE:-0.5}"
FOUNDATION_VALID_ITERS="${FOUNDATION_VALID_ITERS:-32}"
FOUNDATION_MAX_DEPTH_M="${FOUNDATION_MAX_DEPTH_M:-4.0}"
FOUNDATION_OVERWRITE="${FOUNDATION_OVERWRITE:-1}"

if [[ "${RUN_FOUNDATION}" == "1" ]]; then
  FOUNDATION_OVERWRITE_ARG=()
  if [[ "${FOUNDATION_OVERWRITE}" == "1" ]]; then
    FOUNDATION_OVERWRITE_ARG+=(--overwrite)
  fi
  python scripts/run_foundation_stereo_video.py \
    --manifest "${DATASET_MANIFEST}" \
    --foundation_root "${FOUNDATION_ROOT}" \
    --ckpt "${FOUNDATION_CKPT}" \
    --output_dir "${DEPTH_DIR}" \
    --start "${START}" \
    --num_frames "${NUM_FRAMES}" \
    --scale "${FOUNDATION_SCALE}" \
    --valid_iters "${FOUNDATION_VALID_ITERS}" \
    --max_depth_m "${FOUNDATION_MAX_DEPTH_M}" \
    "${FOUNDATION_OVERWRITE_ARG[@]}"
elif [[ ! -d "${DEPTH_DIR}" ]]; then
  echo "Depth dir not found: ${DEPTH_DIR}" >&2
  echo "Set RUN_FOUNDATION=1 to generate Wuxi FoundationStereo depth first." >&2
  exit 2
fi

if [[ "${RUN_ALIGN}" == "1" ]]; then
  scripts/pipeline/03_align_mesh_depth_report.sh
fi

if [[ "${RUN_TRACK}" == "1" ]]; then
  scripts/pipeline/04_track_fill_missing.sh
fi

if [[ "${RUN_RRD}" == "1" ]]; then
  scripts/pipeline/05_export_final_rerun.sh
fi

echo "Final RRD: ${SAVE_RRD}"
