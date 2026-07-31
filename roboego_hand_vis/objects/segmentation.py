from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ObjectObservation:
    frame_idx: int
    object_id: str
    bbox_xywh: Optional[list[int]]
    mask_area_px: int
    source: str
    mask_path: Optional[Path] = None
    crop_path: Optional[Path] = None
    depth_median_m: Optional[float] = None
    depth_valid_ratio: Optional[float] = None


def bbox_xywh_to_xyxy(bbox_xywh: list[int]) -> list[int]:
    x, y, w, h = [int(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def crop_with_padding(image: np.ndarray, bbox_xywh: list[int], padding_px: int) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox_xywh]
    height, width = image.shape[:2]
    x1 = max(0, x - padding_px)
    y1 = max(0, y - padding_px)
    x2 = min(width, x + w + padding_px)
    y2 = min(height, y + h + padding_px)
    return image[y1:y2, x1:x2].copy()
