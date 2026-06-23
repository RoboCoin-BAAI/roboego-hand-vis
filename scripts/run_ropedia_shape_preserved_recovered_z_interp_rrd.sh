#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=. python scripts/view_ropedia_hamer_depth_alignment_rerun.py \
  --candidates_json tmp/ropedia_hamer_ego_filter_000000_000299_recovered_area0007_temporal_gated/ego_hand_candidates_recovered_temporal_gated.json \
  --tracks_json tmp/ropedia_ego_hand_tracking_000000_000299_area0007_temporal_gated_fixed/ego_hand_tracks.json \
  --tracks_current_best_mode fill_missing \
  --recovered_depth_reference_report tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_temporal_gated_final/hand_depth_mesh_alignment_report.json \
  --depth_dir tmp/ropedia_foundation_depth_continuous_000000_000299 \
  --video /home/liuyou/Documents/ropedia_data/stereo_left.mp4 \
  --hamer_overlay_dir tmp/ropedia_hamer_ego_filter_000000_000299_tuned/hamer_overlay \
  --output_dir tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_candidate \
  --save_rrd tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_candidate/ropedia_hamer_new_method_shape_preserved_area0007_recovered_z_interp.rrd \
  --scene_point_stride 4 \
  --max_scene_points 90000 \
  --display_mesh_rotate_axis none \
  --display_mesh_rotate_degrees 0 \
  --display_mesh_rotate_about camera \
  --display_mesh_translate 0 0 0 \
  --new_method_only \
  --mesh_mask_source hamer_overlay_diff \
  --mesh_mask_padding_px 2 \
  --overlay_diff_threshold 8 \
  --overlay_diff_search_margin_px 24 \
  --mask_depth_band_half_width_m 0.12 \
  --depth_lift_window_radius_px 2 \
  --depth_lift_max_delta_m 0.12
