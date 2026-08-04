from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .manipulated_tracker import MaskCandidate, project_joints_to_pixels


@dataclass
class FastSamPromptConfig:
    min_mask_area_px: int = 100
    max_hand_bbox_distance_px: float = 80.0
    max_hand_points_inside_ratio: float = 0.35
    max_mask_frame_area_ratio: float = 0.35
    area_weight: float = 0.00008
    distance_weight: float = 1.0


@dataclass
class SamPointPrompt:
    frame_idx: int
    candidate_id: str
    point_xy: list[int]
    point_labels: list[int]
    bbox_xywh: list[int]
    mask_area_px: int
    score: float
    source: str


def choose_sam_point_from_candidates(
    *,
    candidates: list[MaskCandidate],
    left_joints: np.ndarray | None,
    right_joints: np.ndarray | None,
    intrinsics: dict[str, float],
    config: FastSamPromptConfig | None = None,
) -> SamPointPrompt | None:
    cfg = config or FastSamPromptConfig()
    hand_pixels = np.concatenate(
        [
            project_joints_to_pixels(left_joints, intrinsics),
            project_joints_to_pixels(right_joints, intrinsics),
        ],
        axis=0,
    )
    if len(hand_pixels) == 0:
        return None

    scored: list[tuple[float, MaskCandidate, list[int]]] = []
    for cand in candidates:
        if cand.mask_area_px < cfg.min_mask_area_px:
            continue
        if _mask_frame_area_ratio(cand.mask) > cfg.max_mask_frame_area_ratio:
            continue
        distance = _bbox_distance_to_points_px(cand.bbox_xywh, hand_pixels)
        if distance is None or distance > cfg.max_hand_bbox_distance_px:
            continue
        inside_ratio = _hand_points_inside_mask_ratio(cand.mask, hand_pixels)
        if inside_ratio > cfg.max_hand_points_inside_ratio:
            continue
        point_xy = _mask_inner_point(cand.mask)
        if point_xy is None:
            continue
        score = -distance * cfg.distance_weight + cand.mask_area_px * cfg.area_weight
        scored.append((score, cand, point_xy))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    score, cand, point_xy = scored[0]
    return SamPointPrompt(
        frame_idx=int(cand.frame_idx),
        candidate_id=cand.candidate_id,
        point_xy=point_xy,
        point_labels=[1],
        bbox_xywh=list(cand.bbox_xywh),
        mask_area_px=int(cand.mask_area_px),
        score=float(score),
        source=f"fastsam_hand_roi:{cand.source}",
    )


def _bbox_distance_to_points_px(bbox_xywh: list[int], points_xy: np.ndarray) -> float | None:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.size == 0:
        return None
    x, y, w, h = [float(v) for v in bbox_xywh]
    x2 = x + w
    y2 = y + h
    dx = np.maximum(np.maximum(x - points[:, 0], 0.0), points[:, 0] - x2)
    dy = np.maximum(np.maximum(y - points[:, 1], 0.0), points[:, 1] - y2)
    return float(np.sqrt(dx * dx + dy * dy).min())


def _hand_points_inside_mask_ratio(mask: np.ndarray, points_xy: np.ndarray) -> float:
    if len(points_xy) == 0:
        return 0.0
    height, width = mask.shape[:2]
    inside = 0
    valid = 0
    for x, y in np.round(points_xy).astype(int):
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        valid += 1
        if mask[y, x] > 0:
            inside += 1
    return inside / valid if valid else 0.0


def _mask_frame_area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float((mask > 0).sum()) / float(mask.shape[0] * mask.shape[1])


def _mask_inner_point(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return [int(np.floor(float(np.median(xs)))), int(np.floor(float(np.median(ys))))]
