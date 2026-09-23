from __future__ import annotations

import cv2
import numpy as np


def depth_to_vis(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m)
    out = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, 0.0, max_depth_m)
        out[valid] = (255.0 * (1.0 - clipped[valid] / max_depth_m)).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)


def make_depth_preview_panel_rgb(left_rgb: np.ndarray, depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_vis_rgb = cv2.cvtColor(depth_to_vis(depth_m, max_depth_m), cv2.COLOR_BGR2RGB)
    return np.concatenate([left_rgb, depth_vis_rgb], axis=1)

