from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class EgoHandFilterConfig:
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.35
    preferred_area_ratio: float = 0.035
    min_center_y_ratio: float = 0.12
    bottom_start_y_ratio: float = 0.35
    min_confidence: float = 0.35
    min_valid_depth_m: float = 0.05
    near_depth_m: float = 0.25
    far_depth_m: float = 1.8
    max_median_depth_m: float = 3.0
    confidence_weight: float = 0.45
    bottom_weight: float = 0.25
    area_weight: float = 0.15
    depth_weight: float = 0.15
    recovery_min_area_ratio: float = 0.001
    recovery_min_center_y_ratio: float = 0.35
    recovery_max_median_depth_m: float = 1.2
    wearer_roi_x_margin_ratio: float = 0.0
    wearer_roi_y_margin_ratio: float = 0.05
    semantic_hand_visible_y_margin_ratio: float = 0.25
    semantic_hand_visible_min_area_ratio: float = 0.001
    semantic_hand_visible_recovery_min_area_ratio: float = 0.0005


@dataclass
class EgoHandSelection:
    candidates: List[Dict[str, Any]]
    selected: Dict[str, Optional[int]]


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def bbox_median_depth(
    depth_m: np.ndarray,
    bbox_xyxy: Iterable[float],
    image_width: int,
    image_height: int,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 10.0,
) -> Optional[float]:
    if depth_m is None or depth_m.size == 0:
        return None

    depth_h, depth_w = depth_m.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    scale_x = depth_w / float(image_width)
    scale_y = depth_h / float(image_height)

    dx1 = int(np.floor(x1 * scale_x))
    dy1 = int(np.floor(y1 * scale_y))
    dx2 = int(np.ceil(x2 * scale_x))
    dy2 = int(np.ceil(y2 * scale_y))

    dx1 = max(0, min(depth_w, dx1))
    dx2 = max(0, min(depth_w, dx2))
    dy1 = max(0, min(depth_h, dy1))
    dy2 = max(0, min(depth_h, dy2))
    if dx2 <= dx1 or dy2 <= dy1:
        return None

    crop = depth_m[dy1:dy2, dx1:dx2]
    valid = crop[np.isfinite(crop)]
    valid = valid[(valid >= min_valid_depth_m) & (valid <= max_valid_depth_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def compute_candidate_features(
    candidate: Dict[str, Any],
    image_width: int,
    image_height: int,
    depth_m: Optional[np.ndarray] = None,
    config: EgoHandFilterConfig = EgoHandFilterConfig(),
    wearer_bbox_xyxy: Optional[Iterable[float]] = None,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in candidate["bbox_xyxy"]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    image_area = float(image_width * image_height)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    center_x_ratio = center_x / float(image_width)
    center_y_ratio = center_y / float(image_height)
    area_ratio = area / image_area if image_area > 0 else 0.0
    bottom_score = clamp(
        (center_y_ratio - config.bottom_start_y_ratio)
        / max(1e-6, 1.0 - config.bottom_start_y_ratio)
    )
    area_score = clamp(
        (area_ratio - config.min_area_ratio)
        / max(1e-6, config.preferred_area_ratio - config.min_area_ratio)
    )

    confidence = float(candidate.get("confidence", 0.0))
    median_depth_m = None
    if depth_m is not None:
        median_depth_m = bbox_median_depth(
            depth_m,
            candidate["bbox_xyxy"],
            image_width=image_width,
            image_height=image_height,
            min_valid_depth_m=config.min_valid_depth_m,
            max_valid_depth_m=max(10.0, config.max_median_depth_m),
        )

    if median_depth_m is None:
        depth_score = 0.5
    else:
        depth_score = clamp(
            (config.far_depth_m - median_depth_m)
            / max(1e-6, config.far_depth_m - config.near_depth_m)
        )

    reject_reasons: List[str] = []
    if confidence < config.min_confidence:
        reject_reasons.append("low_confidence")
    if area_ratio < config.min_area_ratio:
        reject_reasons.append("too_small")
    if area_ratio > config.max_area_ratio:
        reject_reasons.append("too_large")
    if center_y_ratio < config.min_center_y_ratio:
        reject_reasons.append("too_high_in_image")
    if median_depth_m is not None and median_depth_m > config.max_median_depth_m:
        reject_reasons.append("too_far")
    if wearer_bbox_xyxy is not None and not bbox_center_inside_wearer_region(
        [x1, y1, x2, y2],
        wearer_bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
        x_margin_ratio=config.wearer_roi_x_margin_ratio,
        y_margin_ratio=config.wearer_roi_y_margin_ratio,
    ):
        reject_reasons.append("outside_wearer_region")

    score = (
        config.confidence_weight * clamp(confidence)
        + config.bottom_weight * bottom_score
        + config.area_weight * area_score
        + config.depth_weight * depth_score
    )

    enriched = dict(candidate)
    enriched.update(
        {
            "bbox_xyxy": [x1, y1, x2, y2],
            "center_xy": [center_x, center_y],
            "center_x_ratio": center_x_ratio,
            "center_y_ratio": center_y_ratio,
            "bbox_area": area,
            "bbox_area_ratio": area_ratio,
            "bottom_score": bottom_score,
            "area_score": area_score,
            "median_depth_m": median_depth_m,
            "depth_score": depth_score,
            "ego_score": float(score),
            "reject_reasons": reject_reasons,
            "status": "rejected" if reject_reasons else "other_hands",
        }
    )
    return enriched


def select_ego_hands(
    candidates: Iterable[Dict[str, Any]],
    image_width: int,
    image_height: int,
    depth_m: Optional[np.ndarray] = None,
    config: EgoHandFilterConfig = EgoHandFilterConfig(),
    wearer_bbox_xyxy: Optional[Iterable[float]] = None,
    semantic_hand_visible: bool = False,
) -> EgoHandSelection:
    effective_config = semantic_hand_visible_config(config) if semantic_hand_visible else config
    enriched = [
        compute_candidate_features(
            candidate,
            image_width=image_width,
            image_height=image_height,
            depth_m=depth_m,
            config=effective_config,
            wearer_bbox_xyxy=wearer_bbox_xyxy,
        )
        for candidate in candidates
    ]

    selected: Dict[str, Optional[int]] = {"ego_left": None, "ego_right": None}
    for side, selected_key in (("left", "ego_left"), ("right", "ego_right")):
        valid = [
            (idx, candidate)
            for idx, candidate in enumerate(enriched)
            if candidate["status"] != "rejected"
            and str(candidate.get("handedness", "")).lower() == side
        ]
        if not valid:
            continue
        best_idx, _ = max(valid, key=lambda item: item[1]["ego_score"])
        enriched[best_idx]["status"] = selected_key
        selected[selected_key] = best_idx

    recover_missing_ego_hands(enriched, selected, effective_config)

    return EgoHandSelection(candidates=enriched, selected=selected)


def semantic_hand_visible_config(config: EgoHandFilterConfig) -> EgoHandFilterConfig:
    return replace(
        config,
        wearer_roi_y_margin_ratio=max(
            float(config.wearer_roi_y_margin_ratio),
            float(config.semantic_hand_visible_y_margin_ratio),
        ),
        min_area_ratio=min(
            float(config.min_area_ratio),
            float(config.semantic_hand_visible_min_area_ratio),
        ),
        recovery_min_area_ratio=min(
            float(config.recovery_min_area_ratio),
            float(config.semantic_hand_visible_recovery_min_area_ratio),
        ),
    )


def normalize_wearer_bbox_xyxy(
    bbox_xyxy: Iterable[float],
    *,
    image_width: int,
    image_height: int,
    x_margin_ratio: float = 0.0,
    y_margin_ratio: float = 0.0,
) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= float(image_width)
        x2 *= float(image_width)
        y1 *= float(image_height)
        y2 *= float(image_height)
    x_margin_px = max(0.0, float(x_margin_ratio)) * float(image_width)
    y_margin_px = max(0.0, float(y_margin_ratio)) * float(image_height)
    x1 -= x_margin_px
    x2 += x_margin_px
    y1 -= y_margin_px
    y2 += y_margin_px
    return [
        max(0.0, min(float(image_width), x1)),
        max(0.0, min(float(image_height), y1)),
        max(0.0, min(float(image_width), x2)),
        max(0.0, min(float(image_height), y2)),
    ]


def bbox_center_inside_wearer_region(
    bbox_xyxy: Iterable[float],
    wearer_bbox_xyxy: Iterable[float],
    *,
    image_width: int,
    image_height: int,
    x_margin_ratio: float = 0.0,
    y_margin_ratio: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    wx1, wy1, wx2, wy2 = normalize_wearer_bbox_xyxy(
        wearer_bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
        x_margin_ratio=x_margin_ratio,
        y_margin_ratio=y_margin_ratio,
    )
    if wx2 <= wx1 or wy2 <= wy1:
        return True
    return wx1 <= cx <= wx2 and wy1 <= cy <= wy2


def is_recoverable_ego_candidate(
    candidate: Dict[str, Any],
    config: EgoHandFilterConfig = EgoHandFilterConfig(),
) -> bool:
    if candidate.get("status") != "rejected":
        return False
    if float(candidate.get("confidence", 0.0)) < config.min_confidence:
        return False
    if float(candidate.get("bbox_area_ratio", 0.0)) < config.recovery_min_area_ratio:
        return False
    if float(candidate.get("center_y_ratio", 0.0)) < config.recovery_min_center_y_ratio:
        return False
    median_depth = candidate.get("median_depth_m")
    if median_depth is not None and float(median_depth) > config.recovery_max_median_depth_m:
        return False

    reject_reasons = set(candidate.get("reject_reasons", []))
    allowed_reasons = {"too_small"}
    return bool(reject_reasons) and reject_reasons.issubset(allowed_reasons)


def recover_missing_ego_hands(
    candidates: List[Dict[str, Any]],
    selected: Dict[str, Optional[int]],
    config: EgoHandFilterConfig = EgoHandFilterConfig(),
) -> Dict[str, Optional[int]]:
    for side, selected_key, recovered_status in (
        ("left", "ego_left", "ego_left_recovered"),
        ("right", "ego_right", "ego_right_recovered"),
    ):
        if selected.get(selected_key) is not None:
            continue
        fallback = [
            (idx, candidate)
            for idx, candidate in enumerate(candidates)
            if str(candidate.get("handedness", "")).lower() == side
            and is_recoverable_ego_candidate(candidate, config)
        ]
        if not fallback:
            continue
        best_idx, best_candidate = max(fallback, key=lambda item: item[1].get("ego_score", 0.0))
        best_candidate["status"] = recovered_status
        best_candidate["recovery"] = {
            "reason": "missing_side_fallback",
            "original_reject_reasons": list(best_candidate.get("reject_reasons", [])),
        }
        selected[selected_key] = best_idx
    return selected
