#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.restoration.realvdeblur import (
    build_deblur_manifest,
    compute_output_geometry,
    encode_frames_to_video,
    read_manifest,
    run_realvdeblur_inference,
    video_properties,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RealVDeblur on stereo videos referenced by a dataset_manifest.json.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--realvdeblur_python", default="python")
    parser.add_argument("--realvdeblur_root", default="../RealVDeblur")
    parser.add_argument("--wan_model_dir", default="../RealVDeblur/checkpoints/Wan2.1-T2V-1.3B")
    parser.add_argument("--checkpoint", default="../RealVDeblur/checkpoints/realvdeblur_dmd.safetensors")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--enable_twm", action="store_true")
    parser.add_argument("--temporal_window_size", type=int, default=21)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = read_manifest(manifest_path)
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "videos"
    frames_root = output_dir / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    left_input = Path(manifest["videos"]["left"]["path"])
    right_input = Path(manifest["videos"]["right"]["path"])
    props = video_properties(left_input)
    geometry = compute_output_geometry(
        width=int(props["width"]),
        height=int(props["height"]),
        target_width=args.width,
        target_height=args.height,
    )

    output_videos = {
        "left": videos_dir / "left_rectified.mp4",
        "right": videos_dir / "right_rectified.mp4",
    }
    frame_counts: dict[str, int] = {}
    commands: dict[str, list[str]] = {}
    for camera, input_video in [("left", left_input), ("right", right_input)]:
        frames_dir = frames_root / camera
        output_video = output_videos[camera]
        if output_video.exists() and not args.overwrite:
            frame_counts[camera] = int(video_properties(output_video)["frame_count"])
            continue
        frames_dir.mkdir(parents=True, exist_ok=True)
        commands[camera] = run_realvdeblur_inference(
            python=args.realvdeblur_python,
            realvdeblur_root=Path(args.realvdeblur_root),
            input_video=input_video,
            output_frames_dir=frames_dir,
            wan_model_dir=Path(args.wan_model_dir),
            checkpoint=Path(args.checkpoint),
            height=geometry.height,
            width=geometry.width,
            device=args.device,
            dtype=args.dtype,
            num_inference_steps=args.num_inference_steps,
            enable_twm=args.enable_twm,
            temporal_window_size=args.temporal_window_size,
        )
        frame_counts[camera] = encode_frames_to_video(frames_dir, output_video, fps=float(props["fps"]), codec=args.codec)

    if frame_counts.get("left") != frame_counts.get("right"):
        raise RuntimeError(f"Left/right deblurred frame counts differ: {frame_counts}")

    deblur_manifest = build_deblur_manifest(
        manifest,
        output_dir=output_dir,
        left_video=output_videos["left"],
        right_video=output_videos["right"],
        width=geometry.width,
        height=geometry.height,
        fps=float(props["fps"]),
        frame_count=int(frame_counts["left"]),
        geometry=geometry,
    )
    deblur_manifest_path = write_manifest(deblur_manifest, output_dir)
    meta = {
        "stage": "realvdeblur_stereo_manifest",
        "status": "complete",
        "source_manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "manifest": str(deblur_manifest_path.resolve()),
        "videos": {key: str(value.resolve()) for key, value in output_videos.items()},
        "frame_counts": frame_counts,
        "geometry": {
            "width": geometry.width,
            "height": geometry.height,
            "scale": geometry.scale,
            "crop_left": geometry.crop_left,
            "crop_top": geometry.crop_top,
        },
        "commands": commands,
    }
    meta_path = output_dir / "realvdeblur_stereo_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
