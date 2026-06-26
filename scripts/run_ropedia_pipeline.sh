#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/pipeline/common.sh"

export DATASET="${DATASET:-ropedia}"
export START="${START:-0}"
export NUM_FRAMES="${NUM_FRAMES:-300}"
export VIDEO="${VIDEO:-/home/liuyou/Documents/ropedia_data/stereo_left.mp4}"
export DATASET_MANIFEST="${DATASET_MANIFEST:-tmp/ropedia_dataset_manifest/dataset_manifest.json}"
export DEPTH_DIR="${DEPTH_DIR:-tmp/ropedia_foundation_depth_continuous_000000_000299}"
export DEPTH_QUALITY_REPORT="${DEPTH_QUALITY_REPORT:-tmp/ropedia_depth_quality_000000_000299/depth_quality_report.json}"

export HAMER_OUT="${HAMER_OUT:-tmp/ropedia_hamer_ego_filter_000000_000299_tuned}"
export HAMER_CANDIDATES_JSON="${HAMER_CANDIDATES_JSON:-${HAMER_OUT}/ego_hand_candidates.json}"
export RECOVER_OUT="${RECOVER_OUT:-tmp/ropedia_hamer_ego_filter_000000_000299_recovered_area0007_temporal_gated}"
export CANDIDATES_JSON="${CANDIDATES_JSON:-${RECOVER_OUT}/ego_hand_candidates_recovered.json}"
export HAMER_OVERLAY_DIR="${HAMER_OVERLAY_DIR:-${HAMER_OUT}/hamer_overlay}"
export ALIGN_OUT="${ALIGN_OUT:-tmp/ropedia_hand_depth_alignment_000000_000299}"
export ALIGNMENT_REPORT="${ALIGNMENT_REPORT:-${ALIGN_OUT}/hand_depth_mesh_alignment_report.json}"
export TRACK_OUT="${TRACK_OUT:-tmp/ropedia_ego_hand_tracking_000000_000299}"
export TRACKS_JSON="${TRACKS_JSON:-${TRACK_OUT}/ego_hand_tracks.json}"
export RRD_OUT="${RRD_OUT:-tmp/ropedia_hand_vis_000000_000299}"
export SAVE_RRD="${SAVE_RRD:-${RRD_OUT}/ropedia_hand_vis.rrd}"

RUN_HAMER="${RUN_HAMER:-0}"
RUN_RECOVER="${RUN_RECOVER:-1}"
RUN_ALIGN="${RUN_ALIGN:-1}"
RUN_TRACK="${RUN_TRACK:-1}"
RUN_RRD="${RUN_RRD:-1}"
RUN_ORIENTATION_EVAL="${RUN_ORIENTATION_EVAL:-1}"

if [[ "${RUN_HAMER}" == "1" ]]; then
  scripts/pipeline/01_run_hamer_candidates.sh
fi

if [[ "${RUN_RECOVER}" == "1" ]]; then
  scripts/pipeline/02_recover_candidates.sh
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

if [[ "${RUN_ORIENTATION_EVAL}" == "1" ]]; then
  python scripts/experiments/evaluate_orientation_compensation.py \
    --alignment_report "${ALIGNMENT_REPORT}" \
    --output_dir "tmp/ropedia_orientation_compensation_eval_${START}_$((START + NUM_FRAMES - 1))" \
    --start "${START}" \
    --num_frames "${NUM_FRAMES}" \
    --depth_bias_m "${ORIENT_DEPTH_BIAS_M}" \
    --surface_min_m "${ORIENT_SURFACE_MIN_M}" \
    --surface_max_m "${ORIENT_SURFACE_MAX_M}" \
    --ray_blend "${ORIENT_RAY_BLEND}"
fi

echo "Final RRD: ${SAVE_RRD}"
