from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .segmentation import ObjectObservation


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def sam3_mask_path(masks_root: Path, frame_idx: int, object_id: str) -> Path:
    return masks_root / f"frame_{frame_idx:06d}_masks" / f"{object_id}.png"


def load_sam3_mask_observation(
    masks_root: Path,
    *,
    frame_idx: int,
    object_id: str,
) -> tuple[ObjectObservation, np.ndarray]:
    path = sam3_mask_path(masks_root, frame_idx, object_id)
    if not path.exists():
        return ObjectObservation(frame_idx, object_id, None, 0, "sam3_mask_missing"), np.zeros((0, 0), dtype=np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    mask = (mask > 0).astype(np.uint8) * 255
    bbox = _bbox_from_mask(mask)
    area = int((mask > 0).sum())
    return ObjectObservation(frame_idx, object_id, bbox, area, "sam3_mask"), mask
