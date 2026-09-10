from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Ropedia HaMeR ego hands against FoundationStereo hand depth crops."
    )
    parser.add_argument("--candidates_json", default="tmp/ropedia_hamer_ego_filter_000000_000007_tuned/ego_hand_candidates.json")
    parser.add_argument(
        "--tracks_json",
        default="",
        help="Optional ego_hand_tracks.json. If set, current_best_mesh uses tracked/smoothed translation.",
    )
    parser.add_argument(
        "--recovered_depth_reference_report",
        default="",
        help=(
            "Optional previous alignment report. When set, recovered hands use target_z_median "
            "interpolated from neighboring non-recovered hands of the same side."
        ),
    )
    parser.add_argument("--tracks_current_best_mode", choices=["replace", "fill_missing", "log_only"], default="replace")
    parser.add_argument("--depth_dir", default="tmp/ropedia_foundation_depth_continuous_000000_000029")
    parser.add_argument("--video", default="data/ropedia_data/stereo_left.mp4")
    parser.add_argument("--hamer_overlay_dir", default="")
    parser.add_argument("--intrinsics_json", default="", help="Optional camera intrinsics JSON. Uses camera=left.")
    parser.add_argument("--intrinsics_camera", default="left", help="Camera key inside intrinsics_json.")
    parser.add_argument("--intrinsics_resize_scale", type=float, default=1.0)
    parser.add_argument("--output_dir", default="tmp/ropedia_hamer_depth_alignment_000000_000007")
    parser.add_argument("--save_rrd", default="")
    parser.add_argument("--no_save_rrd", action="store_true", help="Write report/debug images without saving an RRD file.")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--bbox_margin_px", type=int, default=14)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--max_points_per_hand", type=int, default=12000)
    parser.add_argument("--max_mesh_vertices_for_error", type=int, default=778)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--rgb_overlay_point_radius", type=float, default=2.0)
    parser.add_argument("--max_projected_mesh_points", type=int, default=778)
    parser.add_argument("--max_projected_depth_points", type=int, default=2000)
    parser.add_argument("--log_raw_mesh", action="store_true")
    parser.add_argument("--log_raw_depth_crop", action="store_true")
    parser.add_argument("--log_full_scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_final_view", action="store_true")
    parser.add_argument("--current_best_only", action="store_true")
    parser.add_argument("--new_method_only", action="store_true")
    parser.add_argument("--scene_point_stride", type=int, default=4)
    parser.add_argument("--max_scene_points", type=int, default=90000)
    parser.add_argument("--display_mesh_rotate_axis", choices=["none", "x", "y", "z"], default="none")
    parser.add_argument("--display_mesh_rotate_degrees", type=float, default=0.0)
    parser.add_argument("--display_mesh_rotate_about", choices=["camera", "centroid"], default="camera")
    parser.add_argument("--display_mesh_translate", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--depth_band_half_width_m", type=float, default=0.16)
    parser.add_argument("--mesh_mask_source", choices=["coarse_fit", "raw_hamer", "hamer_renderer", "hamer_overlay_diff"], default="coarse_fit")
    parser.add_argument("--mesh_mask_padding_px", type=int, default=10)
    parser.add_argument("--overlay_diff_threshold", type=float, default=8.0)
    parser.add_argument("--overlay_diff_search_margin_px", type=int, default=24)
    parser.add_argument("--mask_depth_band_half_width_m", type=float, default=0.12)
    parser.add_argument("--z_anchor_source", choices=["mask", "core_mask"], default="core_mask")
    parser.add_argument("--core_mask_keep_x_ratio", type=float, default=0.6)
    parser.add_argument("--core_mask_keep_y_ratio", type=float, default=0.6)
    parser.add_argument("--core_mask_min_pixels", type=int, default=20)
    parser.add_argument("--finger_anchor_keep_top_ratio", type=float, default=0.65)
    parser.add_argument("--finger_anchor_min_pixels", type=int, default=20)
    parser.add_argument("--shape_preserved_depth_source", choices=["mask", "finger"], default="finger")
    parser.add_argument("--orientation_aware_compensation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mano_model_dir", default="../hamer/_DATA/data/mano")
    parser.add_argument("--orientation_depth_bias_m", type=float, default=0.035)
    parser.add_argument("--orientation_surface_min_m", type=float, default=0.005)
    parser.add_argument("--orientation_surface_max_m", type=float, default=0.020)
    parser.add_argument("--orientation_ray_blend", type=float, default=0.3)
    parser.add_argument("--depth_lift_window_radius_px", type=int, default=2)
    parser.add_argument("--depth_lift_max_delta_m", type=float, default=0.12)
    parser.add_argument("--hamer_base_focal_length", type=float, default=5000.0)
    parser.add_argument("--hamer_model_image_size", type=float, default=256.0)
    parser.add_argument("--smooth_translation_alpha", type=float, default=0.35)
    parser.add_argument("--fit_mode", choices=["translation", "similarity"], default="translation")
    return parser.parse_args()
