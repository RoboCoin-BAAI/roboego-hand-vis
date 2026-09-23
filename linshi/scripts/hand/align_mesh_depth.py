#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_depth_alignment.alignment_interface import (  # noqa: E402
    build_legacy_alignment_argv,
    filter_candidates_by_frame_range,
    load_alignment_inputs_from_manifest,
    write_alignment_summary,
)
from linshi.scripts.hand.mesh_depth_alignment_impl import main as alignment_impl_main  # noqa: E402


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align HaMeR/MANO ego hand meshes to FoundationStereo depth.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates_json", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hamer_overlay_dir", default="", help="Defaults to <candidates_dir>/hamer_overlay.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--save_rrd", default="", help="Optional RRD path. Defaults to output_dir/ropedia_hamer_depth_alignment.rrd.")
    parser.add_argument("--no_save_rrd", action="store_true")
    parser.add_argument("--scene_point_stride", type=int, default=4)
    parser.add_argument("--max_scene_points", type=int, default=90000)
    parser.add_argument("--mano_model_dir", default="../hamer/_DATA/data/mano")
    parser.add_argument("--mesh_mask_source", default="hamer_overlay_diff")
    parser.add_argument("--mesh_mask_padding_px", type=int, default=2)
    parser.add_argument("--overlay_diff_threshold", type=float, default=8.0)
    parser.add_argument("--overlay_diff_search_margin_px", type=int, default=24)
    parser.add_argument("--mask_depth_band_half_width_m", type=float, default=0.12)
    parser.add_argument("--shape_preserved_depth_source", choices=["mask", "finger"], default="finger")
    parser.add_argument("--orientation_aware_compensation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation_depth_bias_m", type=float, default=0.005)
    parser.add_argument("--orientation_surface_min_m", type=float, default=0.003)
    parser.add_argument("--orientation_surface_max_m", type=float, default=0.005)
    parser.add_argument("--orientation_ray_blend", type=float, default=0.3)
    parser.add_argument("--depth_lift_window_radius_px", type=int, default=2)
    parser.add_argument("--depth_lift_max_delta_m", type=float, default=0.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    manifest_path = resolve_path(args.manifest, cwd)
    candidates_json = resolve_path(args.candidates_json, cwd)
    depth_dir = resolve_path(args.depth_dir, cwd)
    output_dir = resolve_path(args.output_dir, cwd)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_alignment_inputs_from_manifest(manifest_path)
    hamer_overlay_dir = (
        resolve_path(args.hamer_overlay_dir, cwd)
        if args.hamer_overlay_dir
        else candidates_json.parent / "hamer_overlay"
    )
    filtered_candidates = output_dir / "inputs" / f"ego_hand_candidates_{args.start:06d}_{args.start + args.num_frames - 1:06d}.json"
    filtered_count = filter_candidates_by_frame_range(
        candidates_json,
        filtered_candidates,
        start=args.start,
        num_frames=args.num_frames,
    )
    if filtered_count == 0:
        raise RuntimeError(
            f"No candidate frames in range [{args.start}, {args.start + args.num_frames - 1}] from {candidates_json}"
        )

    legacy_argv = build_legacy_alignment_argv(
        candidates_json=filtered_candidates,
        depth_dir=depth_dir,
        video_path=inputs.video_path,
        hamer_overlay_dir=hamer_overlay_dir,
        output_dir=output_dir,
        intrinsics_json=inputs.intrinsics_json,
        intrinsics_camera=inputs.intrinsics_camera,
        save_rrd=resolve_path(args.save_rrd, cwd) if args.save_rrd else None,
        no_save_rrd=args.no_save_rrd,
        scene_point_stride=args.scene_point_stride,
        max_scene_points=args.max_scene_points,
        mano_model_dir=args.mano_model_dir,
        mesh_mask_source=args.mesh_mask_source,
        mesh_mask_padding_px=args.mesh_mask_padding_px,
        overlay_diff_threshold=args.overlay_diff_threshold,
        overlay_diff_search_margin_px=args.overlay_diff_search_margin_px,
        mask_depth_band_half_width_m=args.mask_depth_band_half_width_m,
        shape_preserved_depth_source=args.shape_preserved_depth_source,
        orientation_aware_compensation=args.orientation_aware_compensation,
        orientation_depth_bias_m=args.orientation_depth_bias_m,
        orientation_surface_min_m=args.orientation_surface_min_m,
        orientation_surface_max_m=args.orientation_surface_max_m,
        orientation_ray_blend=args.orientation_ray_blend,
        depth_lift_window_radius_px=args.depth_lift_window_radius_px,
        depth_lift_max_delta_m=args.depth_lift_max_delta_m,
    )
    write_alignment_summary(
        output_dir=output_dir,
        manifest_path=manifest_path,
        candidates_json=candidates_json,
        filtered_candidates_json=filtered_candidates,
        depth_dir=depth_dir,
        video_path=inputs.video_path,
        hamer_overlay_dir=hamer_overlay_dir,
        frame_count=filtered_count,
        parameters={
            "start": args.start,
            "num_frames": args.num_frames,
            "mesh_mask_source": args.mesh_mask_source,
            "shape_preserved_depth_source": args.shape_preserved_depth_source,
            "orientation_aware_compensation": args.orientation_aware_compensation,
            "orientation_depth_bias_m": args.orientation_depth_bias_m,
            "orientation_surface_min_m": args.orientation_surface_min_m,
            "orientation_surface_max_m": args.orientation_surface_max_m,
            "orientation_ray_blend": args.orientation_ray_blend,
        },
    )

    old_argv = sys.argv
    try:
        sys.argv = legacy_argv
        alignment_impl_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
