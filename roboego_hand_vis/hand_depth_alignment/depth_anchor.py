from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthAnchorSelection:
    points: np.ndarray
    colors: np.ndarray | None
    source: str
    fallback_reason: str | None = None

    @property
    def count(self) -> int:
        return int(len(self.points))

    @property
    def median_z(self) -> float | None:
        if len(self.points) == 0:
            return None
        z = self.points[:, 2].astype(np.float32)
        z = z[np.isfinite(z)]
        if len(z) == 0:
            return None
        return float(np.median(z))

    def report(self) -> dict:
        return {
            "source": self.source,
            "point_count": self.count,
            "median_z": self.median_z,
            "fallback_reason": self.fallback_reason,
        }


def choose_z_anchor(
    *,
    z_anchor_source: str,
    core_points: np.ndarray,
    core_colors: np.ndarray,
    mask_points: np.ndarray,
    mask_colors: np.ndarray,
    bbox_points: np.ndarray,
    bbox_colors: np.ndarray,
) -> DepthAnchorSelection:
    if z_anchor_source == "core_mask" and len(core_points) > 0:
        return DepthAnchorSelection(core_points, core_colors, "core_mask_points")
    if len(mask_points) > 0:
        reason = "core_mask_empty" if z_anchor_source == "core_mask" else None
        return DepthAnchorSelection(mask_points, mask_colors, "mask_hand_points", fallback_reason=reason)
    return DepthAnchorSelection(bbox_points, bbox_colors, "bbox_hand_points", fallback_reason="mask_empty")


def choose_finger_anchor(
    *,
    finger_points: np.ndarray,
    finger_colors: np.ndarray,
    z_anchor: DepthAnchorSelection,
) -> DepthAnchorSelection:
    if len(finger_points) > 0:
        return DepthAnchorSelection(finger_points, finger_colors, "finger_upper_mask_points")
    return DepthAnchorSelection(
        z_anchor.points,
        z_anchor.colors,
        z_anchor.source,
        fallback_reason="finger_empty",
    )


def choose_shape_preserved_depth_anchor(
    *,
    shape_preserved_depth_source: str,
    finger_points: np.ndarray,
    finger_colors: np.ndarray,
    z_anchor: DepthAnchorSelection,
    mask_points: np.ndarray,
    bbox_points: np.ndarray,
) -> DepthAnchorSelection:
    finger_anchor = choose_finger_anchor(
        finger_points=finger_points,
        finger_colors=finger_colors,
        z_anchor=z_anchor,
    )
    if shape_preserved_depth_source == "finger" and len(finger_anchor.points) > 0:
        return finger_anchor
    if len(mask_points) > 0:
        reason = None if shape_preserved_depth_source != "finger" else "finger_empty"
        return DepthAnchorSelection(mask_points, None, "mask_hand_points", fallback_reason=reason)
    return DepthAnchorSelection(bbox_points, None, "bbox_hand_points", fallback_reason="mask_empty")
