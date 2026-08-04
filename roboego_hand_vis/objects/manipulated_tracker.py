from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import ObjectObservation


@dataclass
class MaskCandidate:
    frame_idx: int
    candidate_id: str
    bbox_xywh: list[int]
    mask_area_px: int
    mask: np.ndarray
    source: str
    object_points: np.ndarray | None = None


@dataclass
class ManipulatedTrackerConfig:
    min_mask_area_px: int = 100
    max_hand_bbox_distance_px: float = 80.0
    max_hand_object_distance_m: float = 0.08
    max_missing_frames: int = 8
    iou_bonus: float = 35.0
    distance_weight: float = 1.0
    area_weight: float = 0.002


def project_joints_to_pixels(joints: np.ndarray | None, intrinsics: dict[str, float]) -> np.ndarray:
    if joints is None:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
    z = arr[:, 2]
    valid = np.isfinite(arr).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)
    pts = arr[valid]
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    pixels = np.empty((len(pts), 2), dtype=np.float32)
    pixels[:, 0] = pts[:, 0] * fx / pts[:, 2] + cx
    pixels[:, 1] = pts[:, 1] * fy / pts[:, 2] + cy
    return pixels


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


def _bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _min_distance_3d(points: np.ndarray | None, joints: np.ndarray | None) -> float | None:
    if points is None or joints is None:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    js = np.asarray(joints, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    js = js[np.isfinite(js).all(axis=1)]
    if len(pts) == 0 or len(js) == 0:
        return None
    distances = np.linalg.norm(pts[:, None, :] - js[None, :, :], axis=2)
    return float(distances.min())


class ManipulatedObjectTracker:
    def __init__(self, *, object_id: str, config: ManipulatedTrackerConfig | None = None):
        self.object_id = object_id
        self.config = config or ManipulatedTrackerConfig()
        self._last_bbox: list[int] | None = None
        self._last_mask_area_px = 0
        self._missed_frames = 0

    def _score(
        self,
        candidate: MaskCandidate,
        *,
        hand_pixels: np.ndarray,
        hand_joints: np.ndarray | None,
    ) -> float | None:
        if candidate.mask_area_px < self.config.min_mask_area_px or candidate.bbox_xywh is None:
            return None

        pixel_distance = _bbox_distance_to_points_px(candidate.bbox_xywh, hand_pixels)
        distance_3d = _min_distance_3d(candidate.object_points, hand_joints)
        near_in_image = (
            pixel_distance is not None
            and pixel_distance <= self.config.max_hand_bbox_distance_px
        )
        near_in_depth = (
            distance_3d is not None
            and distance_3d <= self.config.max_hand_object_distance_m
        )
        if not near_in_image and not near_in_depth:
            return None

        distance_term = 0.0 if pixel_distance is None else -pixel_distance * self.config.distance_weight
        area_term = float(candidate.mask_area_px) * self.config.area_weight
        iou_term = _bbox_iou(self._last_bbox, candidate.bbox_xywh) * self.config.iou_bonus
        depth_term = 20.0 if near_in_depth else 0.0
        return distance_term + area_term + iou_term + depth_term

    def update(
        self,
        *,
        frame_idx: int,
        candidates: list[MaskCandidate],
        left_joints: np.ndarray | None,
        right_joints: np.ndarray | None,
        intrinsics: dict[str, float],
    ) -> tuple[ObjectObservation, MaskCandidate | None]:
        left_pixels = project_joints_to_pixels(left_joints, intrinsics)
        right_pixels = project_joints_to_pixels(right_joints, intrinsics)
        hand_pixels = np.concatenate([left_pixels, right_pixels], axis=0)
        hand_joints = _concat_joints(left_joints, right_joints)

        scored: list[tuple[float, MaskCandidate]] = []
        for cand in candidates:
            score = self._score(cand, hand_pixels=hand_pixels, hand_joints=hand_joints)
            if score is not None:
                scored.append((score, cand))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = scored[0][1]
            self._last_bbox = list(selected.bbox_xywh)
            self._last_mask_area_px = int(selected.mask_area_px)
            self._missed_frames = 0
            return (
                ObjectObservation(
                    frame_idx=frame_idx,
                    object_id=self.object_id,
                    bbox_xywh=list(selected.bbox_xywh),
                    mask_area_px=int(selected.mask_area_px),
                    source=f"tracked:{selected.source}:{selected.candidate_id}",
                ),
                selected,
            )

        if self._last_bbox is not None and self._missed_frames < self.config.max_missing_frames:
            self._missed_frames += 1
            return (
                ObjectObservation(
                    frame_idx=frame_idx,
                    object_id=self.object_id,
                    bbox_xywh=list(self._last_bbox),
                    mask_area_px=int(self._last_mask_area_px),
                    source="tracked_gap_fill",
                ),
                None,
            )

        self._last_bbox = None
        self._last_mask_area_px = 0
        return ObjectObservation(frame_idx, self.object_id, None, 0, "tracked_missing"), None


def _concat_joints(left_joints: np.ndarray | None, right_joints: np.ndarray | None) -> np.ndarray | None:
    arrays = []
    if left_joints is not None:
        arrays.append(np.asarray(left_joints, dtype=np.float32).reshape(-1, 3))
    if right_joints is not None:
        arrays.append(np.asarray(right_joints, dtype=np.float32).reshape(-1, 3))
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)
