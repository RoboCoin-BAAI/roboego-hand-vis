from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import as_abs, manifest_summary, probe_video


Roi = tuple[int, int, int, int]


def roi_from_ratios(
    *,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Roi:
    if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
        raise ValueError("ROI ratios must satisfy 0 <= min < max <= 1")
    x0 = int(round(width * x_min))
    x1 = int(round(width * x_max))
    y0 = int(round(height * y_min))
    y1 = int(round(height * y_max))
    return validate_roi((x0, y0, x1, y1), width=width, height=height)


def validate_roi(roi: Roi, *, width: int, height: int) -> Roi:
    x0, y0, x1, y1 = [int(v) for v in roi]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid ROI {roi} for image size {width}x{height}")
    return x0, y0, x1, y1


def crop_intrinsics(intrinsics: dict[str, Any], roi: Roi) -> dict[str, Any]:
    x0, y0, x1, y1 = roi
    out = copy.deepcopy(intrinsics)
    out["cx"] = float(intrinsics["cx"]) - float(x0)
    out["cy"] = float(intrinsics["cy"]) - float(y0)
    out["width"] = int(x1 - x0)
    out["height"] = int(y1 - y0)
    return out


def crop_frame(frame: np.ndarray, roi: Roi) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return frame[y0:y1, x0:x1].copy()


def crop_video(input_video: Path, output_video: Path, roi: Roi, *, codec: str = "mp4v", overwrite: bool = False) -> dict[str, Any]:
    input_video = as_abs(input_video)
    output_video = as_abs(output_video)
    if output_video.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_video}")

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    roi = validate_roi(roi, width=width, height=height)
    x0, y0, x1, y1 = roi

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (x1 - x0, y1 - y0),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video writer: {output_video}")

    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(crop_frame(frame, roi))
            written += 1
    finally:
        cap.release()
        writer.release()

    return {
        "input": str(input_video),
        "output": str(output_video),
        "input_width": width,
        "input_height": height,
        "output_width": int(x1 - x0),
        "output_height": int(y1 - y0),
        "fps": fps,
        "input_frame_count": frame_count,
        "written_frame_count": written,
        "roi_xyxy": [int(v) for v in roi],
        "codec": codec,
    }


def update_manifest_for_crop(
    manifest: dict[str, Any],
    *,
    left_video: Path,
    right_video: Path,
    roi: Roi,
) -> dict[str, Any]:
    out = copy.deepcopy(manifest)
    left_video = as_abs(left_video)
    right_video = as_abs(right_video)
    out["videos"]["left"] = probe_video(left_video)
    out["videos"]["right"] = probe_video(right_video)
    out["intrinsics"]["left"] = crop_intrinsics(manifest["intrinsics"]["left"], roi)
    out["intrinsics"]["right"] = crop_intrinsics(manifest["intrinsics"]["right"], roi)
    out["sequence_name"] = f"{manifest.get('sequence_name', 'default')}_crop"
    out["root_dir"] = str(left_video.parent.parent if left_video.parent.name == "videos" else left_video.parent)
    out["metadata"] = copy.deepcopy(manifest.get("metadata", {}))
    out["metadata"]["crop"] = {
        "source_manifest": manifest.get("outputs", {}).get("manifest_path"),
        "source_left_video": manifest["videos"]["left"]["path"],
        "source_right_video": manifest["videos"]["right"]["path"],
        "roi_xyxy": [int(v) for v in roi],
        "roi_note": "Same crop is applied to left/right rectified videos; cx/cy are shifted by x_min/y_min.",
    }
    return out


def write_cropped_manifest(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = as_abs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest["outputs"] = {
        "manifest_path": str(output_dir / "dataset_manifest.json"),
        "summary_path": str(output_dir / "dataset_summary.txt"),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "dataset_summary.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(manifest_summary(manifest), encoding="utf-8")
    return manifest_path, summary_path
