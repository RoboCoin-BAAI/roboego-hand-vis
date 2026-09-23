from __future__ import annotations

from pathlib import Path
from typing import overload

import cv2
import numpy as np

DEPTH_NPY_PATTERN = "frame_%06d_foundation_depth_m.npy"
DEPTH_LEGACY_NPY_PATTERN = "frame_%06d_depth_m.npy"
DEPTH_UINT16_PNG_DIR = "depth_uint16_png"
DEPTH_UINT16_PNG_PATTERN = "frame_%06d_depth_mm.png"


def depth_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    return Path(depth_dir) / (DEPTH_NPY_PATTERN % int(frame_idx))


def legacy_depth_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    return Path(depth_dir) / (DEPTH_LEGACY_NPY_PATTERN % int(frame_idx))


def depth_uint16_png_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    return Path(depth_dir) / DEPTH_UINT16_PNG_DIR / (DEPTH_UINT16_PNG_PATTERN % int(frame_idx))


def find_depth_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    depth_dir = Path(depth_dir)
    candidates = [
        depth_path_for_frame(depth_dir, frame_idx),
        legacy_depth_path_for_frame(depth_dir, frame_idx),
        depth_uint16_png_path_for_frame(depth_dir, frame_idx),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def depth_m_to_uint16_mm(depth_m: np.ndarray, *, max_depth_m: float, invalid_value: int = 0) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_depth_m))
    encoded = np.full(depth.shape, int(invalid_value), dtype=np.uint16)
    encoded[valid] = np.rint(depth[valid] * 1000.0).clip(0, np.iinfo(np.uint16).max).astype(np.uint16)
    return encoded


def uint16_mm_to_depth_m(depth_mm: np.ndarray, *, invalid_value: int = 0) -> np.ndarray:
    encoded = np.asarray(depth_mm, dtype=np.uint16)
    depth = encoded.astype(np.float32) / 1000.0
    depth[encoded == int(invalid_value)] = np.nan
    return depth


def save_depth_uint16_png(path: Path, depth_m: np.ndarray, *, max_depth_m: float, invalid_value: int = 0) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), depth_m_to_uint16_mm(depth_m, max_depth_m=max_depth_m, invalid_value=invalid_value)))


@overload
def load_depth_frame(depth_dir: Path, frame_idx: int, *, return_path: bool = False, invalid_value: int = 0) -> np.ndarray: ...


@overload
def load_depth_frame(depth_dir: Path, frame_idx: int, *, return_path: bool, invalid_value: int = 0) -> tuple[np.ndarray, Path]: ...


def load_depth_frame(
    depth_dir: Path,
    frame_idx: int,
    *,
    return_path: bool = False,
    invalid_value: int = 0,
) -> np.ndarray | tuple[np.ndarray, Path]:
    path = find_depth_path_for_frame(depth_dir, frame_idx)
    if not path.exists():
        raise FileNotFoundError(f"No depth frame found for frame {frame_idx}: {path}")
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".png":
        encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if encoded is None:
            raise RuntimeError(f"Cannot read depth PNG: {path}")
        if encoded.dtype != np.uint16:
            raise ValueError(f"Depth PNG must be uint16 millimeters: {path}")
        depth = uint16_mm_to_depth_m(encoded, invalid_value=invalid_value)
    else:
        raise ValueError(f"Unsupported depth frame format: {path}")
    if return_path:
        return depth, path
    return depth
