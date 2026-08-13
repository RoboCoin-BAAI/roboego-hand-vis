#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.orientation import load_mano_regressors  # noqa: E402
from roboego_hand_vis.egohand.temporal_tracking import (  # noqa: E402
    apply_recovered_depth_overrides,
    build_recovered_depth_overrides_from_frames,
    process_tracks,
    track_summary,
)
from roboego_hand_vis.egohand.track_debug import export_debug_contact_sheet  # noqa: E402
from roboego_hand_vis.egohand.track_report import write_tracks_csv, write_tracks_json  # noqa: E402
from roboego_hand_vis.egohand.track_rerun import (  # noqa: E402
    apply_alignment_orientation_compensation,
    build_final_smoothed_meshes,
    fitted_mesh_from_observation,
    hand_mesh_entity_paths,
    hand_rgb_overlay_layers,
    hand_rgb_projected_points_paths,
    iter_rerun_offsets,
    log_rerun,
    project_points_to_image,
)
from roboego_hand_vis.egohand.tracking_interface import load_tracking_inputs_from_manifest  # noqa: E402
from scripts.export_depth_pointcloud import ROPEDIA_K  # noqa: E402


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track, interpolate, smooth, and flag ego hand trajectories.")
    parser.add_argument("--manifest", default="", help="Optional dataset_manifest.json. Overrides --video.")
    parser.add_argument("--alignment_report", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--video", default="/home/liuyou/Documents/ropedia_data/stereo_left.mp4")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--max_interp_gap", type=int, default=5)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--jump_threshold_m", type=float, default=0.08)
    parser.add_argument("--mesh_error_threshold_m", type=float, default=0.05)
    parser.add_argument("--min_valid_depth_ratio", type=float, default=0.5)
    parser.add_argument(
        "--enable_lr_identity_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use cross-frame continuity to swap left/right observations when the per-frame labels are crossed.",
    )
    parser.add_argument(
        "--drop_large_jumps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop unexplained large target-anchor jumps before gap interpolation.",
    )
    parser.add_argument("--large_jump_threshold_m", type=float, default=0.12)
    parser.add_argument(
        "--max_bbox_center_jump_px",
        type=float,
        default=200.0,
        help="Drop a same-side observation when its bbox center jumps by more than this many pixels. Use <=0 to disable.",
    )
    parser.add_argument("--lr_swap_margin_m", type=float, default=0.05)
    parser.add_argument(
        "--fill_edge_gaps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fill short missing gaps at the beginning/end of a track with the nearest observation.",
    )
    parser.add_argument(
        "--apply_recovered_depth_overrides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace recovered hand z with z interpolated from neighboring normal hands.",
    )
    parser.add_argument(
        "--drop_unstable_recovered",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop recovered low-area candidates when their 3D anchor jumps too far from neighboring frames.",
    )
    parser.add_argument("--unstable_recovered_jump_threshold_m", type=float, default=0.08)
    parser.add_argument(
        "--unstable_recovered_max_area_ratio",
        type=float,
        default=0.001,
        help="Only temporal-gate recovered candidates below this bbox area ratio. Use <=0 to gate all recovered candidates.",
    )
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--log_full_scene", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene_point_stride", type=int, default=6)
    parser.add_argument("--max_scene_points", type=int, default=50000)
    parser.add_argument("--point_radius", type=float, default=0.004)
    parser.add_argument("--rgb_overlay_point_radius", type=float, default=1.0)
    parser.add_argument("--max_debug_frames", type=int, default=80)
    parser.add_argument("--mano_model_dir", default="", help="Optional override for MANO model dir used by orientation compensation.")
    parser.add_argument(
        "--final_smoothing_window",
        type=int,
        default=5,
        help="Window for final weighted trajectory smoothing. <=0 disables final layer.",
    )
    parser.add_argument("--save_rrd", default="", help="Optional RRD output path.")
    parser.add_argument("--no_save_rrd", action="store_true", help="Only write JSON/CSV/debug images, skip RRD.")
    parser.add_argument("--rrd_start", type=int, default=None, help="First absolute frame index to write to RRD.")
    parser.add_argument(
        "--rrd_num_frames",
        type=int,
        default=0,
        help="Number of frames to write to RRD. <=0 means all processed frames from --rrd_start.",
    )
    parser.add_argument(
        "--log_layered_meshes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log current plus normal/recovered/smooth mesh layers under camera_opencv/hands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    alignment_report = resolve_path(args.alignment_report, cwd)
    depth_dir = resolve_path(args.depth_dir, cwd)
    manifest_path = resolve_path(args.manifest, cwd) if args.manifest else None
    manifest_inputs = load_tracking_inputs_from_manifest(manifest_path) if manifest_path else None
    video_path = manifest_inputs.video_path if manifest_inputs is not None else resolve_path(args.video, cwd)
    output_dir = resolve_path(args.output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(alignment_report.read_text())
    intrinsics = report.get("intrinsics") or dict(ROPEDIA_K)
    orientation_config = report.get("orientation_aware_compensation") or {"enabled": False}
    candidates_json = resolve_path(report["candidates_json"], cwd) if report.get("candidates_json") else None
    hamer_overlay_dir = resolve_path(report["hamer_overlay_dir"], cwd) if report.get("hamer_overlay_dir") else None
    mano_model_dir = (
        resolve_path(args.mano_model_dir, cwd)
        if args.mano_model_dir
        else Path(orientation_config.get("mano_model_dir") or "").expanduser()
    )
    regressor_by_side: Dict[str, np.ndarray] = {}
    if orientation_config.get("enabled") and str(mano_model_dir):
        regressor_by_side = load_mano_regressors(mano_model_dir)

    report_frames = report["frames"]
    recovered_depth_overrides = {}
    if args.apply_recovered_depth_overrides:
        recovered_depth_overrides = build_recovered_depth_overrides_from_frames(report_frames)
        report_frames = apply_recovered_depth_overrides(report_frames, recovered_depth_overrides)
    tracks = process_tracks(
        report_frames,
        start=args.start,
        num_frames=args.num_frames,
        max_interp_gap=args.max_interp_gap,
        smooth_alpha=args.smooth_alpha,
        jump_threshold_m=args.jump_threshold_m,
        mesh_error_threshold_m=args.mesh_error_threshold_m,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        drop_unstable_recovered=args.drop_unstable_recovered,
        unstable_recovered_jump_threshold_m=args.unstable_recovered_jump_threshold_m,
        unstable_recovered_max_area_ratio=(
            None if args.unstable_recovered_max_area_ratio <= 0 else args.unstable_recovered_max_area_ratio
        ),
        enable_lr_identity_gate=args.enable_lr_identity_gate,
        drop_large_jumps=args.drop_large_jumps,
        large_jump_threshold_m=args.large_jump_threshold_m,
        max_bbox_center_jump_px=args.max_bbox_center_jump_px,
        lr_swap_margin_m=args.lr_swap_margin_m,
        fill_edge_gaps_enabled=args.fill_edge_gaps,
    )
    summary = track_summary(tracks)
    summary["identity_and_jump_gate"] = {
        "enable_lr_identity_gate": bool(args.enable_lr_identity_gate),
        "drop_large_jumps": bool(args.drop_large_jumps),
        "large_jump_threshold_m": float(args.large_jump_threshold_m),
        "max_bbox_center_jump_px": float(args.max_bbox_center_jump_px),
        "lr_swap_margin_m": float(args.lr_swap_margin_m),
        "fill_edge_gaps": bool(args.fill_edge_gaps),
    }
    summary["recovered_depth_overrides"] = {
        "enabled": bool(args.apply_recovered_depth_overrides),
        "count": len(recovered_depth_overrides),
    }
    summary["final_smoothing"] = {
        "enabled": args.final_smoothing_window > 0,
        "window": args.final_smoothing_window,
    }
    write_tracks_json(
        tracks,
        summary,
        output_dir / "ego_hand_tracks.json",
        args,
        manifest_inputs=manifest_inputs,
        video_path=video_path,
        alignment_report=alignment_report,
        depth_dir=depth_dir,
        output_dir=output_dir,
    )
    write_tracks_csv(tracks, output_dir / "ego_hand_tracks.csv", start=args.start)
    export_debug_contact_sheet(tracks, video_path, output_dir, args.start, args.max_debug_frames)
    if not args.no_save_rrd:
        save_rrd = resolve_path(args.save_rrd, cwd) if args.save_rrd else output_dir / "ropedia_ego_hands_tracked_smoothed.rrd"
        final_tracks = build_final_smoothed_meshes(
            tracks,
            window=args.final_smoothing_window,
            regressor_by_side=regressor_by_side,
            orientation_config=orientation_config,
        )
        log_rerun(
            tracks,
            final_tracks,
            video_path=video_path,
            depth_dir=depth_dir,
            save_rrd=save_rrd,
            args=args,
            intrinsics=intrinsics,
            orientation_config=orientation_config,
            regressor_by_side=regressor_by_side,
            candidates_json=candidates_json,
            hamer_overlay_dir=hamer_overlay_dir,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
