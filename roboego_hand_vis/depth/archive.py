from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roboego_hand_vis.depth.io import DEPTH_UINT16_PNG_DIR, DEPTH_UINT16_PNG_PATTERN


DEPTH_NPY_GLOB = "frame_*_foundation_depth_m.npy"
DEPTH_UINT16_PNG_GLOB = "frame_*_depth_mm.png"


@dataclass(frozen=True)
class DepthArchiveConfig:
    max_depth_m: float = 4.0
    invalid_value: int = 0
    fps: float = 30.0
    write_png: bool = True
    write_avi: bool = True
    write_preview: bool = True
    preview_mode: str = "near"
    preview_near_min_m: float = 0.15
    preview_near_max_m: float = 3.0
    preview_contrast_percentile: float | None = 2.0
    preview_codec: str = "mp4v"
    ffv1_container: str = "avi"
    dry_run_avi: bool = False
    delete_npy: bool = False
    delete_png: bool = False
    ffmpeg_bin: str = "ffmpeg"
    write_hevc12: bool = False
    hevc12_crf: int = 0
    hevc12_preset: str = "slow"
    hevc12_lossless: bool = True
    write_valid_mask: bool = False


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
    contrast_percentile: float | None = 2.0,
    preview_mode: str = "near",
    near_min_m: float = 0.15,
    near_max_m: float = 3.0,
) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.uint16)
    max_depth_mm = max(1.0, float(max_depth_m) * 1000.0)
    valid = depth != int(invalid_value)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid].astype(np.float32)
    if values.size and preview_mode == "near":
        lo = max(0.0, float(near_min_m) * 1000.0)
        hi = max(lo + 1.0, float(near_max_m) * 1000.0)
        normalized[valid] = np.clip((values - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    elif values.size and contrast_percentile is not None and contrast_percentile > 0:
        pct = float(contrast_percentile)
        lo = float(np.percentile(values, pct))
        hi = float(np.percentile(values, 100.0 - pct))
        if hi > lo:
            normalized[valid] = np.clip((values - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            normalized[valid] = np.clip((values / max_depth_mm) * 255.0, 0, 255).astype(np.uint8)
    elif values.size:
        normalized[valid] = np.clip((values / max_depth_mm) * 255.0, 0, 255).astype(np.uint8)
    # OpenCV colormaps place low values at cool colors for TURBO, so invert to make near depths visually warm.
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def archive_depth_cache(*, depth_dir: Path, output_dir: Path, config: DepthArchiveConfig) -> dict[str, Any]:
    depth_dir = Path(depth_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    npy_paths = sorted(depth_dir.glob(DEPTH_NPY_GLOB))
    source_png_dir = depth_dir / DEPTH_UINT16_PNG_DIR
    source_png_paths = sorted(source_png_dir.glob(DEPTH_UINT16_PNG_GLOB))
    if npy_paths:
        sources = npy_paths
        source_format = "float32_npy"
    elif source_png_paths:
        sources = source_png_paths
        source_format = "uint16_png"
    else:
        raise FileNotFoundError(
            f"No depth files found in {depth_dir}; expected {DEPTH_NPY_GLOB} or {source_png_dir / DEPTH_UINT16_PNG_GLOB}"
        )
    if source_format == "float32_npy" and config.write_avi and not config.write_png:
        raise ValueError("write_png must be enabled when archiving NPY depth to AVI.")
    if config.ffv1_container not in {"avi", "mkv"}:
        raise ValueError("ffv1_container must be one of: avi, mkv")
    max_depth_mm = int(round(float(config.max_depth_m) * 1000.0))
    if config.write_hevc12 and max_depth_mm > 4095:
        raise ValueError("HEVC12 direct millimeter encoding requires max_depth_m <= 4.095")

    png_dir = output_dir / "depth_uint16_png"
    if config.write_png and (source_format == "float32_npy" or png_dir != source_png_dir):
        png_dir.mkdir(parents=True, exist_ok=True)
    avi_input_png_dir = png_dir if config.write_png else source_png_dir

    frames: list[dict[str, Any]] = []
    preview_writer: cv2.VideoWriter | None = None
    preview_path = output_dir / "depth_preview.mp4"
    hevc12_path = output_dir / "depth_hevc12.mkv"
    hevc12_command: list[str] = []
    hevc12_proc: subprocess.Popen | None = None
    valid_mask_path = output_dir / "valid_mask_ffv1.mkv"
    valid_mask_command: list[str] = []
    valid_mask_proc: subprocess.Popen | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    for source in sources:
        frame_idx = _frame_idx_from_depth_path(source)
        if source_format == "float32_npy":
            depth = np.load(source)
            encoded = depth_m_to_uint16_mm(depth, max_depth_m=config.max_depth_m, invalid_value=config.invalid_value)
        else:
            encoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if encoded is None:
                raise RuntimeError(f"Failed to read depth PNG: {source}")
            if encoded.dtype != np.uint16:
                raise ValueError(f"Depth PNG must be uint16 millimeters: {source}")
        if frame_width is None or frame_height is None:
            frame_height, frame_width = encoded.shape[:2]
        png_path = png_dir / f"frame_{frame_idx:06d}_depth_mm.png"
        if config.write_png:
            if source_format == "uint16_png" and source != png_path:
                shutil.copy2(source, png_path)
            elif source_format == "float32_npy":
                if not cv2.imwrite(str(png_path), encoded):
                    raise RuntimeError(f"Failed to write depth PNG: {png_path}")
        if config.write_preview:
            preview_frame = depth_uint16_mm_to_preview_bgr(
                encoded,
                max_depth_m=config.max_depth_m,
                invalid_value=config.invalid_value,
                contrast_percentile=config.preview_contrast_percentile,
                preview_mode=config.preview_mode,
                near_min_m=config.preview_near_min_m,
                near_max_m=config.preview_near_max_m,
            )
            if preview_writer is None:
                output_dir.mkdir(parents=True, exist_ok=True)
                height, width = preview_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*config.preview_codec)
                preview_writer = cv2.VideoWriter(str(preview_path), fourcc, float(config.fps), (width, height))
                if not preview_writer.isOpened():
                    raise RuntimeError(f"Failed to open preview video writer: {preview_path}")
            preview_writer.write(preview_frame)
        if config.write_hevc12:
            hevc12_frame = np.clip(encoded, 0, max_depth_mm).astype(np.uint16)
            if hevc12_proc is None:
                output_dir.mkdir(parents=True, exist_ok=True)
                height, width = hevc12_frame.shape[:2]
                hevc12_command = build_hevc12_command(
                    width=width,
                    height=height,
                    output_video=hevc12_path,
                    fps=config.fps,
                    crf=config.hevc12_crf,
                    preset=config.hevc12_preset,
                    lossless=config.hevc12_lossless,
                    ffmpeg_bin=config.ffmpeg_bin,
                )
                hevc12_proc = subprocess.Popen(
                    hevc12_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if hevc12_proc.stdin is None:
                raise RuntimeError("Failed to open HEVC12 encoder stdin")
            hevc12_proc.stdin.write(hevc12_frame.tobytes())
        if config.write_valid_mask:
            valid_mask_frame = np.where(encoded != int(config.invalid_value), 255, 0).astype(np.uint8)
            if valid_mask_proc is None:
                output_dir.mkdir(parents=True, exist_ok=True)
                height, width = valid_mask_frame.shape[:2]
                valid_mask_command = build_valid_mask_command(
                    width=width,
                    height=height,
                    output_video=valid_mask_path,
                    fps=config.fps,
                    ffmpeg_bin=config.ffmpeg_bin,
                )
                valid_mask_proc = subprocess.Popen(
                    valid_mask_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            if valid_mask_proc.stdin is None:
                raise RuntimeError("Failed to open valid mask encoder stdin")
            valid_mask_proc.stdin.write(valid_mask_frame.tobytes())
        frames.append(
            {
                "frame_idx": frame_idx,
                "source_npy": str(source) if source_format == "float32_npy" else "",
                "source_png": str(source) if source_format == "uint16_png" else "",
                "png": str(png_path) if config.write_png else "",
                "valid_pixel_count": int(np.count_nonzero(encoded != int(config.invalid_value))),
            }
        )
    if preview_writer is not None:
        preview_writer.release()
    if hevc12_proc is not None:
        if hevc12_proc.stdin is not None:
            hevc12_proc.stdin.close()
        return_code = hevc12_proc.wait()
        if return_code != 0:
            raise RuntimeError(f"HEVC12 encoder failed with exit code {return_code}: {' '.join(hevc12_command)}")
    if valid_mask_proc is not None:
        if valid_mask_proc.stdin is not None:
            valid_mask_proc.stdin.close()
        return_code = valid_mask_proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Valid mask encoder failed with exit code {return_code}: {' '.join(valid_mask_command)}"
            )

    ffv1_path = output_dir / ("depth_ffv1.mkv" if config.ffv1_container == "mkv" else "depth_uint16_ffv1.avi")
    ffmpeg_command: list[str] = []
    if config.write_avi:
        output_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_command = build_ffv1_command(
            png_dir=avi_input_png_dir,
            output_avi=ffv1_path,
            fps=config.fps,
            ffmpeg_bin=config.ffmpeg_bin,
        )
        if not config.dry_run_avi:
            subprocess.run(ffmpeg_command, check=True)

    can_cleanup = not config.dry_run_avi
    if config.delete_npy and can_cleanup and source_format == "float32_npy":
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

    png_report_dir = _reported_png_dir(
        source_format=source_format,
        source_png_dir=source_png_dir,
        png_dir=png_dir,
        write_png=config.write_png,
        deleted_png=deleted_png,
    )
    compression = _compression_report(
        png_dir=Path(png_report_dir) if png_report_dir else None,
        ffv1_path=ffv1_path if config.write_avi else None,
        hevc12_path=hevc12_path if config.write_hevc12 else None,
        valid_mask_path=valid_mask_path if config.write_valid_mask else None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "depth_archive_v1",
        "source_depth_dir": str(depth_dir),
        "source_depth_format": source_format,
        "output_dir": str(output_dir),
        "unit": "millimeter",
        "encoding": "uint16",
        "invalid_value": int(config.invalid_value),
        "max_depth_m": float(config.max_depth_m),
        "fps": float(config.fps),
        "preview_mode": config.preview_mode,
        "preview_near_min_m": float(config.preview_near_min_m),
        "preview_near_max_m": float(config.preview_near_max_m),
        "preview_contrast_percentile": config.preview_contrast_percentile,
        "frame_count": len(frames),
        "deleted_source_npy": bool(config.delete_npy and can_cleanup and source_format == "float32_npy"),
        "deleted_uint16_png": deleted_png,
        "outputs": {
            "png_dir": png_report_dir,
            "avi_path": str(ffv1_path) if config.write_avi and config.ffv1_container == "avi" else "",
            "ffv1_video": str(ffv1_path) if config.write_avi else "",
            "ffv1_container": config.ffv1_container,
            "hevc12_video": str(hevc12_path) if config.write_hevc12 else "",
            "valid_mask_video": str(valid_mask_path) if config.write_valid_mask else "",
            "preview_video": str(preview_path) if config.write_preview else "",
            "metadata_json": str(output_dir / "depth_archive_meta.json"),
        },
        "ffmpeg_command": ffmpeg_command,
        "ffv1": {
            "enabled": bool(config.write_avi),
            "codec": "ffv1",
            "container": config.ffv1_container,
            "pixel_format": "gray16le",
            "bit_depth": 16,
            "depth_unit": "millimeter",
            "invalid_value": int(config.invalid_value),
            "lossless": True,
            "source_frame_pattern": f"{DEPTH_UINT16_PNG_DIR}/{DEPTH_UINT16_PNG_PATTERN}",
            "frame_count": len(frames),
            "fps": float(config.fps),
            "width": int(frame_width or 0),
            "height": int(frame_height or 0),
        },
        "hevc12_command": hevc12_command,
        "hevc12": {
            "enabled": bool(config.write_hevc12),
            "codec": "libx265",
            "container": "mkv",
            "pixel_format": "gray12le",
            "bit_depth": 12,
            "depth_unit": "millimeter",
            "invalid_value": int(config.invalid_value),
            "valid_range_mm": [1, max_depth_mm],
            "reserved_range_mm": [max_depth_mm + 1, 4095] if max_depth_mm < 4095 else [],
            "crf": int(config.hevc12_crf),
            "preset": config.hevc12_preset,
            "lossless": bool(config.hevc12_lossless),
            "requires_valid_mask": bool(config.write_valid_mask),
            "valid_mask_video": str(valid_mask_path) if config.write_valid_mask else "",
            "restore_rule": "depth_mm[valid_mask == 0] = 0" if config.write_valid_mask else "",
        },
        "valid_mask_command": valid_mask_command,
        "valid_mask": {
            "enabled": bool(config.write_valid_mask),
            "codec": "ffv1",
            "container": "mkv",
            "pixel_format": "gray",
            "bit_depth": 8,
            "lossless": True,
            "valid_value": 255,
            "invalid_value": 0,
            "restore_rule": "depth_mm[valid_mask == 0] = 0",
            "frame_count": len(frames),
            "fps": float(config.fps),
            "width": int(frame_width or 0),
            "height": int(frame_height or 0),
        },
        "compression": compression,
        "frames": frames,
    }
    (output_dir / "depth_archive_meta.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _reported_png_dir(
    *,
    source_format: str,
    source_png_dir: Path,
    png_dir: Path,
    write_png: bool,
    deleted_png: bool,
) -> str:
    if deleted_png:
        return ""
    if source_format == "uint16_png" and not write_png:
        return str(source_png_dir)
    if write_png:
        return str(png_dir)
    return ""


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
        str(Path(png_dir) / DEPTH_UINT16_PNG_PATTERN),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray16le",
        str(output_avi),
    ]


def build_hevc12_command(
    *,
    width: int,
    height: int,
    output_video: Path,
    fps: float,
    crf: int = 0,
    preset: str = "slow",
    lossless: bool = True,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    fps_value = f"{float(fps):g}"
    keyint = str(max(1, int(round(float(fps)))))
    x265_params = "lossless=1:bframes=0" if lossless else "bframes=0"
    return [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{int(width)}x{int(height)}",
        "-pix_fmt",
        "gray12le",
        "-r",
        fps_value,
        "-i",
        "-",
        "-r",
        fps_value,
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        "90000",
        "-c:v",
        "libx265",
        "-crf",
        str(int(crf)),
        "-preset",
        preset,
        "-pix_fmt",
        "gray12le",
        "-g",
        keyint,
        "-keyint_min",
        keyint,
        "-x265-params",
        x265_params,
        str(output_video),
    ]


def build_valid_mask_command(
    *,
    width: int,
    height: int,
    output_video: Path,
    fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    fps_value = f"{float(fps):g}"
    return [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{int(width)}x{int(height)}",
        "-pix_fmt",
        "gray",
        "-r",
        fps_value,
        "-i",
        "-",
        "-r",
        fps_value,
        "-fps_mode",
        "cfr",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray",
        str(output_video),
    ]


def _compression_report(
    *,
    png_dir: Path | None,
    ffv1_path: Path | None,
    hevc12_path: Path | None,
    valid_mask_path: Path | None,
) -> dict[str, Any]:
    png_bytes = _png_sequence_size_bytes(png_dir) if png_dir is not None else 0
    ffv1_bytes = ffv1_path.stat().st_size if ffv1_path is not None and ffv1_path.is_file() else 0
    hevc12_bytes = hevc12_path.stat().st_size if hevc12_path is not None and hevc12_path.is_file() else 0
    valid_mask_bytes = valid_mask_path.stat().st_size if valid_mask_path is not None and valid_mask_path.is_file() else 0
    hevc12_with_mask_bytes = hevc12_bytes + valid_mask_bytes
    return {
        "png_sequence_bytes": int(png_bytes),
        "ffv1_avi_bytes": int(ffv1_bytes),
        "ffv1_bytes": int(ffv1_bytes),
        "hevc12_bytes": int(hevc12_bytes),
        "valid_mask_bytes": int(valid_mask_bytes),
        "hevc12_with_valid_mask_bytes": int(hevc12_with_mask_bytes),
        "ffv1_vs_png_ratio": _ratio(png_bytes, ffv1_bytes),
        "hevc12_vs_png_ratio": _ratio(png_bytes, hevc12_bytes),
        "hevc12_with_valid_mask_vs_png_ratio": _ratio(png_bytes, hevc12_with_mask_bytes),
        "hevc12_vs_ffv1_ratio": _ratio(ffv1_bytes, hevc12_bytes),
        "hevc12_with_valid_mask_vs_ffv1_ratio": _ratio(ffv1_bytes, hevc12_with_mask_bytes),
    }


def _png_sequence_size_bytes(png_dir: Path | None) -> int:
    if png_dir is None or not png_dir.is_dir():
        return 0
    return sum(path.stat().st_size for path in png_dir.glob("frame_*_depth_mm.png") if path.is_file())


def _ratio(reference_bytes: int, encoded_bytes: int) -> float | None:
    if reference_bytes <= 0 or encoded_bytes <= 0:
        return None
    return float(reference_bytes) / float(encoded_bytes)


def _frame_idx_from_depth_path(path: Path) -> int:
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer frame index from depth filename: {path}")
