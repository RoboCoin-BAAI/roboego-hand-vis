from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True)
class OutputGeometry:
    width: int
    height: int
    scale: float
    crop_left: float
    crop_top: float


def compute_output_geometry(
    *,
    width: int,
    height: int,
    target_width: int | None,
    target_height: int | None,
) -> OutputGeometry:
    if (target_width is None) != (target_height is None):
        raise ValueError("target_width and target_height must be provided together.")
    if target_width is None or target_height is None:
        out_width = (int(width) // 16) * 16
        out_height = (int(height) // 16) * 16
        if out_width <= 0 or out_height <= 0:
            raise ValueError(f"Input video is too small: {width}x{height}")
        return OutputGeometry(
            width=out_width,
            height=out_height,
            scale=1.0,
            crop_left=(int(width) - out_width) / 2.0,
            crop_top=(int(height) - out_height) / 2.0,
        )

    if target_width % 16 or target_height % 16:
        raise ValueError("target_width and target_height must be divisible by 16.")
    scale = max(float(target_width) / float(width), float(target_height) / float(height))
    resized_width = round(float(width) * scale)
    resized_height = round(float(height) * scale)
    return OutputGeometry(
        width=int(target_width),
        height=int(target_height),
        scale=scale,
        crop_left=(resized_width - int(target_width)) / 2.0,
        crop_top=(resized_height - int(target_height)) / 2.0,
    )


def transform_intrinsics(intrinsics: dict[str, Any], geometry: OutputGeometry) -> dict[str, Any]:
    updated = copy.deepcopy(intrinsics)
    updated["fx"] = float(intrinsics["fx"]) * geometry.scale
    updated["fy"] = float(intrinsics["fy"]) * geometry.scale
    updated["cx"] = float(intrinsics["cx"]) * geometry.scale - geometry.crop_left
    updated["cy"] = float(intrinsics["cy"]) * geometry.scale - geometry.crop_top
    updated["width"] = geometry.width
    updated["height"] = geometry.height
    return updated


def build_deblur_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    left_video: Path,
    right_video: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    geometry: OutputGeometry,
) -> dict[str, Any]:
    updated = copy.deepcopy(manifest)
    output_dir = Path(output_dir).resolve()
    left_video = Path(left_video).resolve()
    right_video = Path(right_video).resolve()
    duration_sec = float(frame_count) / float(fps) if fps else None
    for camera, video_path in [("left", left_video), ("right", right_video)]:
        video_meta = copy.deepcopy(updated["videos"][camera])
        video_meta.update(
            {
                "path": str(video_path),
                "exists": True,
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "frame_count": int(frame_count),
                "duration_sec": duration_sec,
            }
        )
        updated["videos"][camera] = video_meta
        updated["intrinsics"][camera] = transform_intrinsics(updated["intrinsics"][camera], geometry)

    updated["restoration"] = {
        "method": "RealVDeblur",
        "output_dir": str(output_dir),
        "geometry": {
            "width": geometry.width,
            "height": geometry.height,
            "scale": geometry.scale,
            "crop_left": geometry.crop_left,
            "crop_top": geometry.crop_top,
        },
    }
    updated["outputs"] = {
        **updated.get("outputs", {}),
        "manifest_path": str(output_dir / "dataset_manifest" / "dataset_manifest.json"),
    }
    return updated


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    manifest_dir = Path(output_dir) / "dataset_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def video_properties(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()


def encode_frames_to_video(frames_dir: Path, output_video: Path, *, fps: float, codec: str = "mp4v") -> int:
    frames = sorted(
        path
        for path in Path(frames_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not frames:
        raise RuntimeError(f"No restored frames found in: {frames_dir}")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read restored frame: {frames[0]}")
    height, width = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_video}")
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Cannot read restored frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Frame size mismatch in {frame_path}: {frame.shape[:2]} != {(height, width)}")
            writer.write(frame)
    finally:
        writer.release()
    return len(frames)


def run_realvdeblur_inference(
    *,
    python: str,
    realvdeblur_root: Path,
    input_video: Path,
    output_frames_dir: Path,
    wan_model_dir: Path,
    checkpoint: Path,
    height: int | None,
    width: int | None,
    device: str,
    dtype: str,
    num_inference_steps: int,
    enable_twm: bool,
    temporal_window_size: int,
) -> list[str]:
    args = [
        python,
        str(Path(realvdeblur_root) / "inference.py"),
        "--input",
        str(input_video),
        "--output",
        str(output_frames_dir),
        "--wan_model_dir",
        str(wan_model_dir),
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
        "--dtype",
        dtype,
        "--num_inference_steps",
        str(num_inference_steps),
    ]
    if height is not None and width is not None:
        args.extend(["--height", str(height), "--width", str(width)])
    if enable_twm:
        args.extend(["--enable_twm", "--temporal_window_size", str(temporal_window_size)])
    subprocess.run(args, cwd=str(realvdeblur_root), check=True)
    return args
