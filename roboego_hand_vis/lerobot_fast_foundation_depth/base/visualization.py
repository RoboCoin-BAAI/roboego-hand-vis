"""Display-only contrast controls, independent of metric depth storage."""
import cv2
import numpy as np


def make_preview(left_rgb, depth_m, *, near=.2, far=1., mode="linear"):
    if not np.isfinite([near, far]).all() or not 0 <= near < far:
        raise ValueError("Preview range must satisfy 0 <= near < far")
    valid = np.isfinite(depth_m) & (depth_m > 0)
    clipped = np.clip(np.where(valid, depth_m, far), near, far)
    if mode == "inverse":
        if near <= 0:
            raise ValueError("Inverse depth preview requires near > 0")
        intensity = (1 / clipped - 1 / far) / (1 / near - 1 / far)
    elif mode == "linear":
        intensity = (far - clipped) / (far - near)
    else:
        raise ValueError("Preview mode must be linear or inverse")
    colors = cv2.applyColorMap(np.rint(intensity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colors[~valid] = 0
    rgb = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)
    # Fixed limits avoid per-frame auto-contrast flicker and permit comparison.
    label = f"Near {near:g}m (warm) | Far {far:g}m (cool) | {mode}"
    if rgb.shape[1] >= 400 and rgb.shape[0] >= 100:
        cv2.putText(rgb, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rgb, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([left_rgb, rgb], axis=1)
