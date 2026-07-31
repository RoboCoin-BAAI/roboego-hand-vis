from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MaskDepthStats:
    mask_pixel_count: int
    valid_depth_count: int
    depth_valid_ratio: float
    depth_median_m: float | None
    depth_p10_m: float | None
    depth_p90_m: float | None


def depth_path_for_frame(depth_dir: Path, frame_idx: int) -> Path:
    return depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"


def load_depth_frame(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_path_for_frame(depth_dir, frame_idx)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def mask_depth_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    *,
    intrinsics: dict[str, float],
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    max_points: int = 20000,
) -> tuple[np.ndarray, MaskDepthStats]:
    if depth_m.ndim == 2 and mask.ndim == 2 and depth_m.shape != mask.shape:
        raise ValueError(f"mask/depth shape mismatch: mask {mask.shape}, depth {depth_m.shape}")

    stride = max(1, int(stride))
    mask_pixel_count = int((mask > 0).sum())
    all_valid_z = depth_m[mask > 0]
    all_valid_z = all_valid_z[
        np.isfinite(all_valid_z) & (all_valid_z >= min_depth_m) & (all_valid_z <= max_depth_m)
    ]
    stats = MaskDepthStats(
        mask_pixel_count=mask_pixel_count,
        valid_depth_count=int(len(all_valid_z)),
        depth_valid_ratio=float(len(all_valid_z) / max(1, mask_pixel_count)),
        depth_median_m=float(np.median(all_valid_z)) if len(all_valid_z) else None,
        depth_p10_m=float(np.percentile(all_valid_z, 10)) if len(all_valid_z) else None,
        depth_p90_m=float(np.percentile(all_valid_z, 90)) if len(all_valid_z) else None,
    )

    ys, xs = np.nonzero(mask[::stride, ::stride] > 0)
    ys = ys * stride
    xs = xs * stride
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float32), stats
    z = depth_m[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32), stats
    if max_points > 0 and len(z) > max_points:
        keep = np.linspace(0, len(z) - 1, max_points).astype(np.int64)
        xs = xs[keep]
        ys = ys[keep]
        z = z[keep]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x3 = (xs.astype(np.float32) - cx) * z / fx
    y3 = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x3, y3, z], axis=1).astype(np.float32)
    return points, stats
