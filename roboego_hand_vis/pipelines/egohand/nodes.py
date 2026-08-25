"""Nodes for ego-hand estimation pipelines."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value


def _append_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def _append_many(args: list[str], name: str, values: Any) -> None:
    if not values:
        return
    if isinstance(values, str):
        values = [values]
    values = [str(value) for value in values if _optional_str(value)]
    if values:
        args.append(name)
        args.extend(values)


def _append_flag(args: list[str], name: str, enabled: Any) -> None:
    if bool(enabled):
        args.append(name)


def _default_plan_path(config: dict[str, Any]) -> str:
    output_root = _optional_str(config.get("output_root"))
    if output_root:
        return str(Path(output_root) / "hamer_window_run_plan.json")
    segments_root = _optional_str(config.get("segments_root"))
    if segments_root:
        return str(Path(segments_root) / "hamer_window_run_plan.json")
    return ""


def run_hamer_windows_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "hamer_windows", "status": "skipped", "reason": "disabled"}

    windows_json = _required_value(config, "windows_json", "hamer_windows")
    hamer_root = _required_value(config, "hamer_root", "hamer_windows")
    stage1_actions = _required_value(config, "stage1_actions", "hamer_windows")
    hamer_python = _optional_str(config.get("hamer_python") or config.get("python_for_hamer") or config.get("hamer_env_python"))

    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/run_hamer_windows.py",
        "--windows_json",
        windows_json,
    ]
    _append_arg(args, "--manifest", config.get("manifest"))
    _append_arg(args, "--depth_dir", config.get("depth_dir"))
    _append_arg(args, "--output_root", config.get("output_root"))
    _append_arg(args, "--segments_root", config.get("segments_root"))
    _append_arg(args, "--output_subdir", config.get("output_subdir"))
    args.extend(["--hamer_root", hamer_root, "--stage1_actions", stage1_actions])
    _append_arg(args, "--python", hamer_python)
    _append_arg(args, "--body_detector", config.get("body_detector"))
    _append_arg(args, "--batch_size", config.get("batch_size"))
    _append_arg(args, "--wearer_roi_x_margin_ratio", config.get("wearer_roi_x_margin_ratio"))
    _append_arg(args, "--wearer_roi_y_margin_ratio", config.get("wearer_roi_y_margin_ratio"))
    if bool(config.get("save_mesh", True)):
        args.append("--save_mesh")
    else:
        args.append("--no-save_mesh")
    _append_arg(args, "--refine_mode", config.get("refine_mode"))
    _append_many(args, "--only_window_ids", config.get("only_window_ids"))
    _append_many(args, "--only_segment_ids", config.get("only_segment_ids"))
    _append_arg(args, "--max_windows", config.get("max_windows"))
    if bool(config.get("skip_missing_segments", False)):
        args.append("--skip_missing_segments")
    if bool(config.get("accept_uncertain_refined", False)):
        args.append("--accept_uncertain_refined")
    if bool(config.get("dry_run", False)):
        args.append("--dry_run")
    if bool(config.get("overwrite", False)):
        args.append("--overwrite")

    subprocess.run(args, check=True)
    return {
        "stage": "hamer_windows",
        "status": "complete",
        "windows_json": windows_json,
        "segments_root": _optional_str(config.get("segments_root")),
        "plan_path": _default_plan_path(config),
    }


def run_refined_temporal_gate_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "refined_temporal_gate", "status": "skipped", "reason": "disabled"}

    manifest = _required_value(config, "manifest", "refined_temporal_gate")
    refined_json = _required_value(config, "refined_json", "refined_temporal_gate")
    output_dir = _required_value(config, "output_dir", "refined_temporal_gate")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/track_refined_hands.py",
        "--manifest",
        manifest,
        "--refined_json",
        refined_json,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--max_gap", config.get("max_gap"))
    _append_arg(args, "--max_bbox_jump_px", config.get("max_bbox_jump_px"))
    _append_arg(args, "--max_bbox_area_change_ratio", config.get("max_bbox_area_change_ratio"))
    _append_arg(args, "--max_cam_t_jump", config.get("max_cam_t_jump"))
    _append_arg(args, "--min_cam_t_z", config.get("min_cam_t_z"))
    _append_arg(args, "--max_cam_t_z", config.get("max_cam_t_z"))
    _append_arg(args, "--max_mano_pose_jump", config.get("max_mano_pose_jump"))
    _append_arg(args, "--max_mesh_centroid_jump", config.get("max_mesh_centroid_jump"))
    _append_arg(args, "--max_mesh_vertex_rmse", config.get("max_mesh_vertex_rmse"))
    _append_arg(args, "--local_window_radius", config.get("local_window_radius"))
    if bool(config.get("save_vis", True)):
        args.append("--save_vis")
    else:
        args.append("--no-save_vis")
    _append_arg(args, "--contact_sheet_columns", config.get("contact_sheet_columns"))
    _append_arg(args, "--contact_sheet_thumb_width", config.get("contact_sheet_thumb_width"))

    subprocess.run(args, check=True)
    return {
        "stage": "refined_temporal_gate",
        "status": "complete",
        "manifest": manifest,
        "refined_json": refined_json,
        "output_dir": output_dir,
        "tracks_json": str(Path(output_dir) / "refined_temporal_gate_tracks.json"),
    }


def run_final_hand_result_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "final_hand_result", "status": "skipped", "reason": "disabled"}

    tracks_json = _required_value(config, "tracks_json", "final_hand_result")
    output_dir = _required_value(config, "output_dir", "final_hand_result")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/export_final_hand_result.py",
        "--tracks_json",
        tracks_json,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--temporal_annotated_dir", config.get("temporal_annotated_dir"))
    _append_arg(args, "--hamer_overlay_dir", config.get("hamer_overlay_dir"))
    _append_arg(args, "--source_frames_dir", config.get("source_frames_dir"))
    if bool(config.get("copy_meshes", True)):
        args.append("--copy_meshes")
    else:
        args.append("--no-copy_meshes")
    if bool(config.get("copy_visuals", True)):
        args.append("--copy_visuals")
    else:
        args.append("--no-copy_visuals")

    subprocess.run(args, check=True)
    return {
        "stage": "final_hand_result",
        "status": "complete",
        "tracks_json": tracks_json,
        "output_dir": output_dir,
        "final_tracks_json": str(Path(output_dir) / "final_hand_tracks.json"),
        "final_summary_json": str(Path(output_dir) / "final_hand_summary.json"),
    }


def run_joints21_pose_gate_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "joints21_pose_gate", "status": "skipped", "reason": "disabled"}

    final_tracks_json = _required_value(config, "final_tracks_json", "joints21_pose_gate")
    output_dir = _required_value(config, "output_dir", "joints21_pose_gate")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/gate_hand_pose_joints21.py",
        "--final_tracks_json",
        final_tracks_json,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--mano_model_dir", config.get("mano_model_dir"))
    _append_arg(args, "--max_bone_length_ratio", config.get("max_bone_length_ratio"))
    _append_arg(args, "--max_joint_local_deviation", config.get("max_joint_local_deviation"))
    _append_arg(args, "--max_wrist_local_deviation_m", config.get("max_wrist_local_deviation_m"))
    _append_arg(args, "--max_palm_local_deviation_m", config.get("max_palm_local_deviation_m"))
    _append_arg(args, "--max_fingertip_local_deviation_m", config.get("max_fingertip_local_deviation_m"))
    _append_arg(args, "--min_finger_direction_dot", config.get("min_finger_direction_dot"))
    _append_arg(args, "--local_window_radius", config.get("local_window_radius"))

    subprocess.run(args, check=True)
    return {
        "stage": "joints21_pose_gate",
        "status": "complete",
        "final_tracks_json": final_tracks_json,
        "output_dir": output_dir,
        "pose_quality_report_json": str(Path(output_dir) / "pose_quality_report.json"),
        "final_hand_candidates_gated_json": str(Path(output_dir) / "final_hand_candidates_gated.json"),
        "joints21_compact_npz": str(Path(output_dir) / "joints21_compact.npz"),
    }


def run_depth_anchored_joints21_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "depth_anchored_joints21", "status": "skipped", "reason": "disabled"}

    manifest = _required_value(config, "manifest", "depth_anchored_joints21")
    candidates_json = _required_value(config, "candidates_json", "depth_anchored_joints21")
    depth_dir = _required_value(config, "depth_dir", "depth_anchored_joints21")
    output_dir = _required_value(config, "output_dir", "depth_anchored_joints21")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/export_depth_anchored_joints21.py",
        "--manifest",
        manifest,
        "--candidates_json",
        candidates_json,
        "--depth_dir",
        depth_dir,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--mano_model_dir", config.get("mano_model_dir"))
    _append_arg(args, "--anchor_mode", config.get("anchor_mode"))
    _append_arg(args, "--anchor_joint_index", config.get("anchor_joint_index"))
    _append_arg(args, "--depth_window_radius_px", config.get("depth_window_radius_px"))
    _append_arg(args, "--min_depth_m", config.get("min_depth_m"))
    _append_arg(args, "--max_depth_m", config.get("max_depth_m"))
    _append_arg(args, "--overlay_frames", config.get("overlay_frames"))
    _append_arg(args, "--overlay_thumb_width", config.get("overlay_thumb_width"))
    _append_arg(args, "--overlay_columns", config.get("overlay_columns"))

    subprocess.run(args, check=True)
    return {
        "stage": "depth_anchored_joints21",
        "status": "complete",
        "manifest": manifest,
        "candidates_json": candidates_json,
        "depth_dir": depth_dir,
        "output_dir": output_dir,
        "joints21_depth_anchored_json": str(Path(output_dir) / "joints21_depth_anchored.json"),
        "joints21_depth_anchored_npz": str(Path(output_dir) / "joints21_depth_anchored.npz"),
        "summary_json": str(Path(output_dir) / "joints21_depth_anchored_summary.json"),
    }


def run_legacy_mesh_depth_alignment_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "legacy_mesh_depth_alignment", "status": "skipped", "reason": "disabled"}

    manifest = _required_value(config, "manifest", "legacy_mesh_depth_alignment")
    candidates_json = _required_value(config, "candidates_json", "legacy_mesh_depth_alignment")
    depth_dir = _required_value(config, "depth_dir", "legacy_mesh_depth_alignment")
    output_dir = _required_value(config, "output_dir", "legacy_mesh_depth_alignment")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/align_mesh_depth.py",
        "--manifest",
        manifest,
        "--candidates_json",
        candidates_json,
        "--depth_dir",
        depth_dir,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--hamer_overlay_dir", config.get("hamer_overlay_dir"))
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--save_rrd", config.get("save_rrd"))
    _append_flag(args, "--no_save_rrd", config.get("no_save_rrd"))
    _append_arg(args, "--scene_point_stride", config.get("scene_point_stride"))
    _append_arg(args, "--max_scene_points", config.get("max_scene_points"))
    _append_arg(args, "--mano_model_dir", config.get("mano_model_dir"))
    _append_arg(args, "--mesh_mask_source", config.get("mesh_mask_source"))
    _append_arg(args, "--mesh_mask_padding_px", config.get("mesh_mask_padding_px"))
    _append_arg(args, "--overlay_diff_threshold", config.get("overlay_diff_threshold"))
    _append_arg(args, "--overlay_diff_search_margin_px", config.get("overlay_diff_search_margin_px"))
    _append_arg(args, "--mask_depth_band_half_width_m", config.get("mask_depth_band_half_width_m"))
    _append_arg(args, "--shape_preserved_depth_source", config.get("shape_preserved_depth_source"))
    _append_arg(args, "--orientation_depth_bias_m", config.get("orientation_depth_bias_m"))
    _append_arg(args, "--orientation_surface_min_m", config.get("orientation_surface_min_m"))
    _append_arg(args, "--orientation_surface_max_m", config.get("orientation_surface_max_m"))
    _append_arg(args, "--orientation_ray_blend", config.get("orientation_ray_blend"))
    _append_arg(args, "--depth_lift_window_radius_px", config.get("depth_lift_window_radius_px"))
    _append_arg(args, "--depth_lift_max_delta_m", config.get("depth_lift_max_delta_m"))

    subprocess.run(args, check=True)
    return {
        "stage": "legacy_mesh_depth_alignment",
        "status": "complete",
        "manifest": manifest,
        "candidates_json": candidates_json,
        "depth_dir": depth_dir,
        "output_dir": output_dir,
        "alignment_report_json": str(Path(output_dir) / "hand_depth_mesh_alignment_report.json"),
    }


def run_legacy_track_fill_smooth_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "legacy_track_fill_smooth", "status": "skipped", "reason": "disabled"}

    alignment_report = _required_value(config, "alignment_report", "legacy_track_fill_smooth")
    depth_dir = _required_value(config, "depth_dir", "legacy_track_fill_smooth")
    output_dir = _required_value(config, "output_dir", "legacy_track_fill_smooth")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/track_fill_smooth.py",
    ]
    _append_arg(args, "--manifest", config.get("manifest"))
    args.extend(["--alignment_report", alignment_report, "--depth_dir", depth_dir, "--output_dir", output_dir])
    _append_arg(args, "--video", config.get("video"))
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--max_interp_gap", config.get("max_interp_gap"))
    _append_arg(args, "--smooth_alpha", config.get("smooth_alpha"))
    _append_arg(args, "--jump_threshold_m", config.get("jump_threshold_m"))
    _append_arg(args, "--mesh_error_threshold_m", config.get("mesh_error_threshold_m"))
    _append_arg(args, "--min_valid_depth_ratio", config.get("min_valid_depth_ratio"))
    _append_flag(args, "--enable_lr_identity_gate", config.get("enable_lr_identity_gate"))
    _append_flag(args, "--drop_large_jumps", config.get("drop_large_jumps"))
    _append_arg(args, "--large_jump_threshold_m", config.get("large_jump_threshold_m"))
    _append_arg(args, "--max_bbox_center_jump_px", config.get("max_bbox_center_jump_px"))
    _append_arg(args, "--lr_swap_margin_m", config.get("lr_swap_margin_m"))
    _append_flag(args, "--fill_edge_gaps", config.get("fill_edge_gaps"))
    _append_flag(args, "--apply_recovered_depth_overrides", config.get("apply_recovered_depth_overrides"))
    _append_flag(args, "--drop_unstable_recovered", config.get("drop_unstable_recovered"))
    _append_arg(args, "--unstable_recovered_jump_threshold_m", config.get("unstable_recovered_jump_threshold_m"))
    _append_arg(args, "--unstable_recovered_max_area_ratio", config.get("unstable_recovered_max_area_ratio"))
    _append_arg(args, "--min_depth_m", config.get("min_depth_m"))
    _append_arg(args, "--max_depth_m", config.get("max_depth_m"))
    _append_arg(args, "--scene_point_stride", config.get("scene_point_stride"))
    _append_arg(args, "--max_scene_points", config.get("max_scene_points"))
    _append_arg(args, "--point_radius", config.get("point_radius"))
    _append_arg(args, "--rgb_overlay_point_radius", config.get("rgb_overlay_point_radius"))
    _append_arg(args, "--max_debug_frames", config.get("max_debug_frames"))
    _append_arg(args, "--mano_model_dir", config.get("mano_model_dir"))
    _append_arg(args, "--final_smoothing_window", config.get("final_smoothing_window"))
    _append_arg(args, "--save_rrd", config.get("save_rrd"))
    _append_flag(args, "--no_save_rrd", config.get("no_save_rrd"))
    _append_arg(args, "--rrd_start", config.get("rrd_start"))
    _append_arg(args, "--rrd_num_frames", config.get("rrd_num_frames"))

    subprocess.run(args, check=True)
    return {
        "stage": "legacy_track_fill_smooth",
        "status": "complete",
        "alignment_report": alignment_report,
        "depth_dir": depth_dir,
        "output_dir": output_dir,
        "tracks_json": str(Path(output_dir) / "ego_hand_tracks.json"),
        "tracks_csv": str(Path(output_dir) / "ego_hand_tracks.csv"),
    }


def run_legacy_export_joints21_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "legacy_export_joints21", "status": "skipped", "reason": "disabled"}

    tracks_json = _required_value(config, "tracks_json", "legacy_export_joints21")
    alignment_report = _required_value(config, "alignment_report", "legacy_export_joints21")
    output_dir = _required_value(config, "output_dir", "legacy_export_joints21")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/hand/export_joints21.py",
        "--tracks_json",
        tracks_json,
        "--alignment_report",
        alignment_report,
        "--output_dir",
        output_dir,
    ]
    _append_arg(args, "--start", config.get("start"))
    _append_arg(args, "--num_frames", config.get("num_frames"))
    _append_arg(args, "--final_smoothing_window", config.get("final_smoothing_window"))
    _append_arg(args, "--mano_model_dir", config.get("mano_model_dir"))
    _append_arg(args, "--save_rrd", config.get("save_rrd"))
    _append_flag(args, "--no_save_rrd", config.get("no_save_rrd"))
    _append_arg(args, "--rrd_start", config.get("rrd_start"))
    _append_arg(args, "--rrd_num_frames", config.get("rrd_num_frames"))
    _append_arg(args, "--joint_radius_m", config.get("joint_radius_m"))
    _append_arg(args, "--bone_radius_m", config.get("bone_radius_m"))
    _append_arg(args, "--rgb_joint_radius_px", config.get("rgb_joint_radius_px"))
    _append_arg(args, "--rgb_bone_radius_px", config.get("rgb_bone_radius_px"))

    subprocess.run(args, check=True)
    return {
        "stage": "legacy_export_joints21",
        "status": "complete",
        "tracks_json": tracks_json,
        "alignment_report": alignment_report,
        "output_dir": output_dir,
        "joints21_json": str(Path(output_dir) / "joints_21_camera_opencv.json"),
        "joints21_npz": str(Path(output_dir) / "joints_21_camera_opencv.npz"),
        "summary_json": str(Path(output_dir) / "joints_21_summary.json"),
    }
