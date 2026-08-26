from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEPTH_NPY_GLOB = "frame_*_foundation_depth_m.npy"


@dataclass(frozen=True)
class DepthArchiveConfig:
    max_depth_m: float = 4.0
    invalid_value: int = 0
    fps: float = 30.0
    write_png: bool = True
    write_avi: bool = True
    write_preview: bool = True
    preview_codec: str = "mp4v"
    dry_run_avi: bool = False
    delete_npy: bool = False
    delete_png: bool = False
    ffmpeg_bin: str = "ffmpeg"


def depth_m_to_uint16_mm(depth_m: np.ndarray, *, max_depth_m: float, invalid_value: int = 0) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth_m))
    encoded = np.full(depth.shape, int(invalid_value), dtype=np.uint16)
    encoded[valid] = np.rint(depth[valid] * 1000.0).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    return encoded


def depth_uint16_mm_to_preview_bgr(
    depth_mm: np.ndarray,
    *,
    max_depth_m: float,
    invalid_value: int = 0,
) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.uint16)
    max_depth_mm = max(1.0, float(max_depth_m) * 1000.0)
    valid = depth != int(invalid_value)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip((depth[valid].astype(np.float32) / max_depth_mm) * 255.0, 0, 255).astype(np.uint8)
    # OpenCV colormaps place low values at cool colors for TURBO, so invert to make near depths visually warm.
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def archive_depth_cache(*, depth_dir: Path, output_dir: Path, config: DepthArchiveConfig) -> dict[str, Any]:
    depth_dir = Path(depth_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    npy_paths = sorted(depth_dir.glob(DEPTH_NPY_GLOB))
    if not npy_paths:
        raise FileNotFoundError(f"No depth npy files found in {depth_dir} matching {DEPTH_NPY_GLOB}")

    png_dir = output_dir / "depth_uint16_png"
    if config.write_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    preview_writer: cv2.VideoWriter | None = None
    preview_path = output_dir / "depth_preview.mp4"
    for source in npy_paths:
        frame_idx = _frame_idx_from_depth_path(source)
        depth = np.load(source)
        encoded = depth_m_to_uint16_mm(depth, max_depth_m=config.max_depth_m, invalid_value=config.invalid_value)
        png_path = png_dir / f"frame_{frame_idx:06d}_depth_mm.png"
        if config.write_png:
            if not cv2.imwrite(str(png_path), encoded):
                raise RuntimeError(f"Failed to write depth PNG: {png_path}")
        if config.write_preview:
            preview_frame = depth_uint16_mm_to_preview_bgr(
                encoded,
                max_depth_m=config.max_depth_m,
                invalid_value=config.invalid_value,
            )
            if preview_writer is None:
                output_dir.mkdir(parents=True, exist_ok=True)
                height, width = preview_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*config.preview_codec)
                preview_writer = cv2.VideoWriter(str(preview_path), fourcc, float(config.fps), (width, height))
                if not preview_writer.isOpened():
                    raise RuntimeError(f"Failed to open preview video writer: {preview_path}")
            preview_writer.write(preview_frame)
        frames.append(
            {
                "frame_idx": frame_idx,
                "source_npy": str(source),
                "png": str(png_path) if config.write_png else "",
                "valid_pixel_count": int(np.count_nonzero(encoded != int(config.invalid_value))),
            }
        )
    if preview_writer is not None:
        preview_writer.release()

    avi_path = output_dir / "depth_uint16_ffv1.avi"
    ffmpeg_command: list[str] = []
    if config.write_avi:
        ffmpeg_command = build_ffv1_command(
            png_dir=png_dir,
            output_avi=avi_path,
            fps=config.fps,
            ffmpeg_bin=config.ffmpeg_bin,
        )
        if not config.dry_run_avi:
            subprocess.run(ffmpeg_command, check=True)

    can_cleanup = not config.dry_run_avi
    if config.delete_npy and can_cleanup:
        for source in npy_paths:
            source.unlink()
    deleted_png = False
    if config.delete_png and can_cleanup and png_dir.is_dir():
        for png_path in png_dir.glob("frame_*_depth_mm.png"):
            png_path.unlink()
        try:
            png_dir.rmdir()
            deleted_png = True
        except OSError:
            deleted_png = False

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "depth_archive_v1",
        "source_depth_dir": str(depth_dir),
        "output_dir": str(output_dir),
        "unit": "millimeter",
        "encoding": "uint16",
        "invalid_value": int(config.invalid_value),
        "max_depth_m": float(config.max_depth_m),
        "fps": float(config.fps),
        "frame_count": len(frames),
        "deleted_source_npy": bool(config.delete_npy and can_cleanup),
        "deleted_uint16_png": deleted_png,
        "outputs": {
            "png_dir": "" if deleted_png or not config.write_png else str(png_dir),
            "avi_path": str(avi_path) if config.write_avi else "",
            "preview_video": str(preview_path) if config.write_preview else "",
            "metadata_json": str(output_dir / "depth_archive_meta.json"),
        },
        "ffmpeg_command": ffmpeg_command,
        "frames": frames,
    }
    (output_dir / "depth_archive_meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_ffv1_command(
    *,
    png_dir: Path,
    output_avi: Path,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    fps_value = f"{float(fps):g}"
    return [
        ffmpeg_bin,
        "-y",
        "-framerate",
        fps_value,
        "-i",
        str(Path(png_dir) / "frame_%06d_depth_mm.png"),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray16le",
        str(output_avi),
    ]


def _frame_idx_from_depth_path(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer frame index from depth filename: {path}")
