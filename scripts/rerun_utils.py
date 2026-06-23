from __future__ import annotations

import cv2
import numpy as np
import rerun as rr


def depth_to_vis(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    scaled = np.zeros_like(depth_m, dtype=np.float32)
    scaled[valid] = np.clip((depth_m[valid] - min_m) / max(max_m - min_m, 1e-6), 0.0, 1.0)
    vis_bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis_bgr[~valid] = 0
    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)


def log_pinhole(entity: str, intrinsics: dict, width: int, height: int) -> None:
    k = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rr.log(entity, rr.Pinhole(image_from_camera=k, resolution=[width, height], image_plane_distance=0.08))
