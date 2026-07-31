from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .segmentation import ObjectObservation, crop_with_padding


@dataclass
class YellowSegmentationConfig:
    hsv_lower: tuple[int, int, int] = (16, 55, 70)
    hsv_upper: tuple[int, int, int] = (45, 255, 255)
    roi_x_min_ratio: float = 0.20
    roi_x_max_ratio: float = 0.80
    roi_y_min_ratio: float = 0.38
    min_area_px: int = 900
    open_kernel_px: int = 5
    close_kernel_px: int = 17
    gap_fill_frames: int = 15


def detect_yellow_object(
    image_rgb: np.ndarray,
    *,
    frame_idx: int,
    object_id: str,
    config: YellowSegmentationConfig,
) -> tuple[ObjectObservation, np.ndarray]:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    height, width = image_rgb.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[
        int(height * config.roi_y_min_ratio):height,
        int(width * config.roi_x_min_ratio):int(width * config.roi_x_max_ratio),
    ] = 255
    mask = cv2.inRange(hsv, np.array(config.hsv_lower), np.array(config.hsv_upper))
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((config.open_kernel_px, config.open_kernel_px), np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((config.close_kernel_px, config.close_kernel_px), np.uint8),
    )

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[tuple[int, list[int], list[float], int]] = []
    for label in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[label]]
        cx, cy = [float(v) for v in centroids[label]]
        if area < config.min_area_px:
            continue
        components.append((area, [x, y, w, h], [cx, cy], label))
    if not components:
        return ObjectObservation(frame_idx, object_id, None, 0, "missing"), np.zeros_like(mask)
    components.sort(reverse=True)
    area, bbox, _centroid, label = components[0]
    object_mask = (labels == label).astype(np.uint8) * 255
    return ObjectObservation(frame_idx, object_id, bbox, int(area), "yellow_hsv_component"), object_mask


def fill_missing_observations(frames: list[ObjectObservation], max_gap: int) -> list[ObjectObservation]:
    valid = [idx for idx, obs in enumerate(frames) if obs.bbox_xywh is not None]
    out = list(frames)
    for idx, obs in enumerate(frames):
        if obs.bbox_xywh is not None:
            continue
        prev = max((i for i in valid if i < idx), default=None)
        nxt = min((i for i in valid if i > idx), default=None)
        if prev is None or nxt is None:
            continue
        gap_len = nxt - prev - 1
        if gap_len > max_gap:
            continue
        ratio = (idx - prev) / (nxt - prev)
        a = np.asarray(frames[prev].bbox_xywh, dtype=np.float32)
        b = np.asarray(frames[nxt].bbox_xywh, dtype=np.float32)
        bbox = np.round(a * (1.0 - ratio) + b * ratio).astype(int).tolist()
        out[idx] = ObjectObservation(
            obs.frame_idx,
            obs.object_id,
            bbox,
            bbox[2] * bbox[3],
            "interpolated_yellow_bbox",
        )
    return out


def save_mask_and_crop(
    *,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    obs: ObjectObservation,
    output_dir: Path,
    crop_padding_px: int,
) -> ObjectObservation:
    if obs.bbox_xywh is None:
        return obs
    masks_dir = output_dir / "masks"
    crops_dir = output_dir / "crops"
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{obs.frame_idx:06d}_{obs.object_id}"
    mask_path = masks_dir / f"{stem}.png"
    crop_path = crops_dir / f"{stem}.png"
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(
        str(crop_path),
        cv2.cvtColor(
            crop_with_padding(image_rgb, obs.bbox_xywh, crop_padding_px),
            cv2.COLOR_RGB2BGR,
        ),
    )
    obs.mask_path = mask_path
    obs.crop_path = crop_path
    return obs
