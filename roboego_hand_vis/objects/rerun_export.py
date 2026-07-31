from __future__ import annotations

import cv2
import numpy as np
import rerun as rr

from .segmentation import ObjectObservation


def draw_object_bbox_overlay(
    image_rgb: np.ndarray,
    obs: ObjectObservation,
    *,
    label: str,
    contact_text: str | None = None,
) -> np.ndarray:
    overlay = image_rgb.copy()
    if obs.bbox_xywh is None:
        return overlay
    x, y, w, h = [int(v) for v in obs.bbox_xywh]
    color = (255, 40, 20)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 4)
    text = label if contact_text is None else f"{label} {contact_text}"
    cv2.putText(overlay, text, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return overlay


def log_object_overlay(entity_path: str, image_rgb: np.ndarray, *, jpeg_quality: int = 85) -> None:
    rr.log(entity_path, rr.Image(image_rgb).compress(jpeg_quality=jpeg_quality))
