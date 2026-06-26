from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlignmentInputs:
    manifest_path: Path
    video_path: Path
    intrinsics_json: Path | None
    intrinsics_camera: str
    dataset_name: str | None
    sequence_name: str | None


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_alignment_inputs_from_manifest(manifest_path: str | Path) -> AlignmentInputs:
    manifest_path = _as_path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_camera = manifest.get("reference_camera", "left")
    if reference_camera != "left":
        raise ValueError(f"Only left reference camera is currently supported, got: {reference_camera}")
    video_path = _as_path(manifest["videos"]["left"]["path"])
    calibration_source = manifest.get("camera_model", {}).get("calibration_source")
    intrinsics_json = _as_path(calibration_source) if calibration_source and Path(calibration_source).suffix == ".json" else None
    return AlignmentInputs(
        manifest_path=manifest_path,
        video_path=video_path,
        intrinsics_json=intrinsics_json,
        intrinsics_camera="left",
        dataset_name=manifest.get("dataset_name"),
        sequence_name=manifest.get("sequence_name"),
    )


def filter_candidates_by_frame_range(src: str | Path, dst: str | Path, *, start: int, num_frames: int) -> int:
    src = _as_path(src)
    dst = Path(dst).expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8"))
    end = start + num_frames
    frames = [frame for frame in data.get("frames", []) if start <= int(frame.get("frame_idx", -1)) < end]
    filtered = dict(data)
    filtered["frames"] = frames
    filtered["frame_range"] = {"start": start, "num_frames": num_frames, "end_exclusive": end}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(frames)


def build_legacy_alignment_argv(
    *,
    candidates_json: str | Path,
    depth_dir: str | Path,
    video_path: str | Path,
    hamer_overlay_dir: str | Path,
    output_dir: str | Path,
    intrinsics_json: str | Path | None = None,
    intrinsics_camera: str = "left",
    intrinsics_resize_scale: float = 1.0,
    save_rrd: str | Path | None = None,
    no_save_rrd: bool = False,
    scene_point_stride: int = 4,
    max_scene_points: int = 90000,
    mano_model_dir: str | Path = "/home/liuyou/Documents/hamer/_DATA/data/mano",
    mesh_mask_source: str = "hamer_overlay_diff",
    mesh_mask_padding_px: int = 2,
    overlay_diff_threshold: float = 8.0,
    overlay_diff_search_margin_px: int = 24,
    mask_depth_band_half_width_m: float = 0.12,
    shape_preserved_depth_source: str = "finger",
    orientation_aware_compensation: bool = True,
    orientation_depth_bias_m: float = 0.035,
    orientation_surface_min_m: float = 0.005,
    orientation_surface_max_m: float = 0.020,
    orientation_ray_blend: float = 0.3,
    depth_lift_window_radius_px: int = 2,
    depth_lift_max_delta_m: float = 0.12,
) -> list[str]:
    argv = [
        "mesh_depth_alignment_impl.py",
        "--candidates_json",
        str(_as_path(candidates_json)),
        "--depth_dir",
        str(_as_path(depth_dir)),
        "--video",
        str(_as_path(video_path)),
        "--hamer_overlay_dir",
        str(_as_path(hamer_overlay_dir)),
        "--output_dir",
        str(_as_path(output_dir)),
        "--scene_point_stride",
        str(scene_point_stride),
        "--max_scene_points",
        str(max_scene_points),
        "--display_mesh_rotate_axis",
        "none",
        "--display_mesh_rotate_degrees",
        "0",
        "--display_mesh_rotate_about",
        "camera",
        "--display_mesh_translate",
        "0",
        "0",
        "0",
        "--new_method_only",
        "--mesh_mask_source",
        mesh_mask_source,
        "--mesh_mask_padding_px",
        str(mesh_mask_padding_px),
        "--overlay_diff_threshold",
        str(overlay_diff_threshold),
        "--overlay_diff_search_margin_px",
        str(overlay_diff_search_margin_px),
        "--mask_depth_band_half_width_m",
        str(mask_depth_band_half_width_m),
        "--shape_preserved_depth_source",
        shape_preserved_depth_source,
        "--mano_model_dir",
        str(_as_path(mano_model_dir)),
        "--orientation_depth_bias_m",
        str(orientation_depth_bias_m),
        "--orientation_surface_min_m",
        str(orientation_surface_min_m),
        "--orientation_surface_max_m",
        str(orientation_surface_max_m),
        "--orientation_ray_blend",
        str(orientation_ray_blend),
        "--depth_lift_window_radius_px",
        str(depth_lift_window_radius_px),
        "--depth_lift_max_delta_m",
        str(depth_lift_max_delta_m),
    ]
    if intrinsics_json is not None:
        argv.extend(
            [
                "--intrinsics_json",
                str(_as_path(intrinsics_json)),
                "--intrinsics_camera",
                intrinsics_camera,
                "--intrinsics_resize_scale",
                str(intrinsics_resize_scale),
            ]
        )
    if orientation_aware_compensation:
        argv.append("--orientation_aware_compensation")
    else:
        argv.append("--no-orientation_aware_compensation")
    if save_rrd:
        argv.extend(["--save_rrd", str(_as_path(save_rrd))])
    if no_save_rrd:
        argv.append("--no_save_rrd")
    return argv


def write_alignment_summary(
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    candidates_json: str | Path,
    filtered_candidates_json: str | Path,
    depth_dir: str | Path,
    video_path: str | Path,
    hamer_overlay_dir: str | Path,
    frame_count: int,
    parameters: dict[str, Any],
) -> Path:
    output_dir = _as_path(output_dir)
    summary_path = output_dir / "hand_depth_mesh_alignment_summary.json"
    payload = {
        "manifest_path": str(_as_path(manifest_path)),
        "candidates_json": str(_as_path(candidates_json)),
        "filtered_candidates_json": str(_as_path(filtered_candidates_json)),
        "depth_dir": str(_as_path(depth_dir)),
        "video": str(_as_path(video_path)),
        "hamer_overlay_dir": str(_as_path(hamer_overlay_dir)),
        "filtered_frame_count": int(frame_count),
        "parameters": parameters,
        "alignment_report": str(output_dir / "hand_depth_mesh_alignment_report.json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
