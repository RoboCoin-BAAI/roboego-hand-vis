from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


SIDES = ("ego_left", "ego_right")


def build_temporal_tracks_from_refined(
    refined_payload: dict[str, Any],
    *,
    start: int,
    num_frames: int,
    max_gap: int = 10,
    max_bbox_jump_px: float = 220.0,
    max_bbox_area_change_ratio: float = 3.0,
    max_cam_t_jump: float = 0.0,
    min_cam_t_z: float = 0.0,
    max_cam_t_z: float = 200.0,
    max_mano_pose_jump: float = 0.0,
    max_mesh_centroid_jump: float = 0.0,
    max_mesh_vertex_rmse: float = 0.0,
    local_window_radius: int = 2,
) -> dict[str, Any]:
    frames = refined_payload.get("frames") or refined_payload.get("results") or []
    by_frame = {_frame_index(frame): frame for frame in frames if isinstance(frame, dict)}
    tracks: dict[str, list[dict[str, Any] | None]] = {side: [] for side in SIDES}

    for frame_idx in range(start, start + num_frames):
        frame = by_frame.get(frame_idx, {})
        for side in SIDES:
            hand = frame.get(f"{side}_refined")
            tracks[side].append(_observation_from_refined(frame_idx, side, hand))

    gated = {
        side: _apply_temporal_quality_gate(
            track,
            max_bbox_jump_px=max_bbox_jump_px,
            max_bbox_area_change_ratio=max_bbox_area_change_ratio,
            max_cam_t_jump=max_cam_t_jump,
            min_cam_t_z=min_cam_t_z,
            max_cam_t_z=max_cam_t_z,
            max_mano_pose_jump=max_mano_pose_jump,
            max_mesh_centroid_jump=max_mesh_centroid_jump,
            max_mesh_vertex_rmse=max_mesh_vertex_rmse,
            local_window_radius=local_window_radius,
        )
        for side, track in tracks.items()
    }
    output_frames = _frames_from_tracks(gated, start=start, num_frames=num_frames)
    return {
        "stage": "refined_temporal_gate_v1",
        "parameters": {
            "start": int(start),
            "num_frames": int(num_frames),
            "max_gap": int(max_gap),
            "interpolation_enabled": False,
            "max_bbox_jump_px": float(max_bbox_jump_px),
            "max_bbox_area_change_ratio": float(max_bbox_area_change_ratio),
            "max_cam_t_jump": float(max_cam_t_jump),
            "min_cam_t_z": float(min_cam_t_z),
            "max_cam_t_z": float(max_cam_t_z),
            "max_mano_pose_jump": float(max_mano_pose_jump),
            "max_mesh_centroid_jump": float(max_mesh_centroid_jump),
            "max_mesh_vertex_rmse": float(max_mesh_vertex_rmse),
            "local_window_radius": int(local_window_radius),
        },
        "tracks": {side: items["track"] for side, items in gated.items()},
        "frames": output_frames,
        "summary": _summary({side: items["track"] for side, items in gated.items()}, gated),
    }


def _observation_from_refined(frame_idx: int, side: str, hand: Any) -> dict[str, Any] | None:
    if not isinstance(hand, dict):
        return None
    bbox = _bbox_xyxy(hand)
    if bbox is None:
        return None
    return {
        "frame_idx": int(frame_idx),
        "side": side,
        "candidate_id": hand.get("candidate_id", hand.get("id")),
        "status": "normal",
        "source": "strict_refine_reliable",
        "bbox_xyxy": bbox,
        "bbox_center_xy": _bbox_center(bbox),
        "handedness": hand.get("handedness"),
        "is_right": hand.get("is_right"),
        "confidence": hand.get("confidence"),
        "detector_score": hand.get("detector_score"),
        "keypoint_mean_conf": hand.get("keypoint_mean_conf"),
        "valid_keypoints": hand.get("valid_keypoints"),
        "mesh_obj": hand.get("mesh_obj"),
        "pred_cam_t_full": _float_vector(hand.get("pred_cam_t_full"), 3),
        "mano_params": hand.get("mano_params"),
        "bbox_area": _bbox_area(bbox),
        "bbox_area_ratio": hand.get("bbox_area_ratio"),
        "refine_quality": hand.get("refine_quality"),
        "refine_reasons": hand.get("refine_reasons", []),
        "refine_risks": hand.get("refine_risks", []),
        "anomaly_flags": [],
        "source_hand": hand,
    }


def _apply_temporal_quality_gate(
    track: list[dict[str, Any] | None],
    *,
    max_bbox_jump_px: float,
    max_bbox_area_change_ratio: float,
    max_cam_t_jump: float,
    min_cam_t_z: float,
    max_cam_t_z: float,
    max_mano_pose_jump: float,
    max_mesh_centroid_jump: float,
    max_mesh_vertex_rmse: float,
    local_window_radius: int,
) -> dict[str, list[dict[str, Any] | None] | list[dict[str, Any]]]:
    out: list[dict[str, Any] | None] = []
    dropped: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    local_references = _local_references(track, radius=local_window_radius)
    for offset, obs in enumerate(track):
        if obs is None:
            out.append(None)
            continue
        item = deepcopy(obs)
        drop_reasons = _intrinsic_drop_reasons(
            item,
            min_cam_t_z=min_cam_t_z,
            max_cam_t_z=max_cam_t_z,
        )
        if previous is not None:
            jump = _distance(item["bbox_center_xy"], previous["bbox_center_xy"])
            item["bbox_center_jump_px"] = float(jump)
            if jump > max_bbox_jump_px:
                item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_bbox_center_jump"]))
                item["temporal_quality"] = "jump_warning"
            else:
                item["temporal_quality"] = "continuous"
            area_ratio = _bbox_area_change_ratio(previous["bbox_xyxy"], item["bbox_xyxy"])
            item["bbox_area_change_ratio"] = float(area_ratio)
            if area_ratio > max_bbox_area_change_ratio:
                item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_bbox_area_change"]))
                drop_reasons.append("large_bbox_area_change")
            cam_jump = _cam_t_jump(previous, item)
            if max_cam_t_jump > 0 and cam_jump is not None:
                item["cam_t_jump"] = float(cam_jump)
                if cam_jump > max_cam_t_jump:
                    item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_cam_t_jump"]))
                    drop_reasons.append("large_cam_t_jump")
            if jump > max_bbox_jump_px and "large_bbox_area_change" in drop_reasons:
                drop_reasons.append("large_bbox_center_jump")
            _apply_pose_jump_gate(
                previous=previous,
                item=item,
                local_reference=local_references[offset],
                max_mano_pose_jump=max_mano_pose_jump,
                drop_reasons=drop_reasons,
            )
            _apply_mesh_jump_gate(
                previous=previous,
                item=item,
                local_reference=local_references[offset],
                max_mesh_centroid_jump=max_mesh_centroid_jump,
                max_mesh_vertex_rmse=max_mesh_vertex_rmse,
                drop_reasons=drop_reasons,
            )
        else:
            item["temporal_quality"] = "track_start"
        if drop_reasons:
            item["status"] = "dropped"
            item["drop_reasons"] = sorted(set(drop_reasons))
            dropped.append(item)
            out.append(None)
            continue
        out.append(item)
        previous = item
    return {"track": out, "dropped": dropped}


def _apply_pose_jump_gate(
    *,
    previous: dict[str, Any],
    item: dict[str, Any],
    local_reference: dict[str, Any],
    max_mano_pose_jump: float,
    drop_reasons: list[str],
) -> None:
    if max_mano_pose_jump <= 0:
        return
    current = _mano_pose_vector(item)
    if current is None:
        return
    previous_pose = _mano_pose_vector(previous)
    jump = _vector_distance(previous_pose, current)
    if jump is not None:
        item["mano_pose_jump"] = float(jump)
    local_pose = local_reference.get("mano_pose")
    local_deviation = _vector_distance(local_pose, current)
    if local_deviation is not None:
        item["mano_pose_local_deviation"] = float(local_deviation)
    if _exceeds_temporal_threshold(
        jump,
        local_deviation,
        max_mano_pose_jump,
        local_count=int(local_reference.get("mano_pose_count", 0)),
    ):
        item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_mano_pose_jump"]))
        drop_reasons.append("large_mano_pose_jump")


def _apply_mesh_jump_gate(
    *,
    previous: dict[str, Any],
    item: dict[str, Any],
    local_reference: dict[str, Any],
    max_mesh_centroid_jump: float,
    max_mesh_vertex_rmse: float,
    drop_reasons: list[str],
) -> None:
    current_mesh = _mesh_features(item)
    if current_mesh is None:
        return
    previous_mesh = _mesh_features(previous)
    if max_mesh_centroid_jump > 0:
        centroid_jump = None
        if previous_mesh is not None:
            centroid_jump = _vector_distance(previous_mesh.get("centroid"), current_mesh.get("centroid"))
            if centroid_jump is not None:
                item["mesh_centroid_jump"] = float(centroid_jump)
        centroid_deviation = _vector_distance(local_reference.get("mesh_centroid"), current_mesh.get("centroid"))
        if centroid_deviation is not None:
            item["mesh_centroid_local_deviation"] = float(centroid_deviation)
        if _exceeds_temporal_threshold(
            centroid_jump,
            centroid_deviation,
            max_mesh_centroid_jump,
            local_count=int(local_reference.get("mesh_centroid_count", 0)),
        ):
            item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_mesh_centroid_jump"]))
            drop_reasons.append("large_mesh_centroid_jump")
    if max_mesh_vertex_rmse > 0:
        vertex_rmse = None
        if previous_mesh is not None:
            vertex_rmse = _mesh_vertex_rmse(previous_mesh.get("vertices"), current_mesh.get("vertices"))
            if vertex_rmse is not None:
                item["mesh_vertex_rmse"] = float(vertex_rmse)
        vertex_rmse_local = _mesh_vertex_rmse(local_reference.get("mesh_vertices"), current_mesh.get("vertices"))
        if vertex_rmse_local is not None:
            item["mesh_vertex_rmse_local"] = float(vertex_rmse_local)
        if _exceeds_temporal_threshold(
            vertex_rmse,
            vertex_rmse_local,
            max_mesh_vertex_rmse,
            local_count=int(local_reference.get("mesh_vertices_count", 0)),
        ):
            item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_mesh_vertex_rmse"]))
            drop_reasons.append("large_mesh_vertex_rmse")


def _exceeds_temporal_threshold(
    jump: float | None,
    local_deviation: float | None,
    threshold: float,
    *,
    local_count: int,
) -> bool:
    if local_count >= 2 and local_deviation is not None:
        return local_deviation > threshold
    return jump is not None and jump > threshold


def _local_references(track: list[dict[str, Any] | None], *, radius: int) -> list[dict[str, Any]]:
    if radius <= 0:
        return [{} for _ in track]
    refs: list[dict[str, Any]] = []
    for idx, _obs in enumerate(track):
        neighbors = [
            other
            for other_idx in range(max(0, idx - radius), min(len(track), idx + radius + 1))
            if other_idx != idx
            for other in [track[other_idx]]
            if other is not None
        ]
        pose_values = [_mano_pose_vector(item) for item in neighbors]
        mesh_values = [features for item in neighbors for features in [_mesh_features(item)] if features is not None]
        refs.append(
            {
                "mano_pose": _median_vector(pose_values),
                "mano_pose_count": sum(1 for value in pose_values if value is not None),
                "mesh_centroid": _median_vector(features["centroid"] for features in mesh_values),
                "mesh_centroid_count": len(mesh_values),
                "mesh_vertices": _median_vertices(features["vertices"] for features in mesh_values),
                "mesh_vertices_count": len(mesh_values),
            }
        )
    return refs


def _frames_from_tracks(
    tracks: dict[str, list[dict[str, Any] | None]],
    *,
    start: int,
    num_frames: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for offset in range(num_frames):
        item: dict[str, Any] = {"frame_idx": int(start + offset), "hands": {}}
        dropped_hands: dict[str, Any] = {}
        for side in SIDES:
            obs = tracks[side]["track"][offset]
            item["hands"][side] = obs
            for dropped in tracks[side]["dropped"]:
                if int(dropped["frame_idx"]) == int(start + offset):
                    dropped_hands[side] = dropped
                    break
        item["dropped_hands"] = dropped_hands
        item["visibility_state"] = _visibility_state(item["hands"])
        out.append(item)
    return out


def _summary(
    tracks: dict[str, list[dict[str, Any] | None]],
    gated: dict[str, dict[str, list[dict[str, Any] | None] | list[dict[str, Any]]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for side, track in tracks.items():
        normal = sum(1 for obs in track if obs is not None and obs.get("status") == "normal")
        interpolated = sum(1 for obs in track if obs is not None and obs.get("status") == "interpolated")
        missing = sum(1 for obs in track if obs is None)
        jumps = sum(1 for obs in track if obs is not None and "large_bbox_center_jump" in obs.get("anomaly_flags", []))
        dropped = [obs for obs in gated[side]["dropped"] if isinstance(obs, dict)]
        drop_reasons: dict[str, int] = {}
        for obs in dropped:
            for reason in obs.get("drop_reasons", []):
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
        summary[side] = {
            "normal": normal,
            "interpolated": interpolated,
            "missing": missing,
            "dropped": len(dropped),
            "drop_reasons": drop_reasons,
            "large_bbox_center_jump": jumps,
            "total": len(track),
        }
    return summary


def _visibility_state(hands: dict[str, dict[str, Any] | None]) -> str:
    left = hands.get("ego_left") is not None
    right = hands.get("ego_right") is not None
    if left and right:
        return "both_hands"
    if left:
        return "only_left"
    if right:
        return "only_right"
    return "none"


def _bbox_xyxy(hand: dict[str, Any]) -> list[float] | None:
    bbox = hand.get("bbox_xyxy") or hand.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    values = [float(v) for v in bbox[:4]]
    if values[2] <= values[0] or values[3] <= values[1]:
        return [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    return values


def _bbox_center(bbox: list[float]) -> list[float]:
    return [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]


def _distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_area_change_ratio(a: list[float], b: list[float]) -> float:
    area_a = max(_bbox_area(a), 1e-6)
    area_b = max(_bbox_area(b), 1e-6)
    return float(max(area_a, area_b) / min(area_a, area_b))


def _mano_pose_vector(obs: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(obs, dict):
        return None
    mano_params = obs.get("mano_params")
    if not isinstance(mano_params, dict):
        return None
    arrays = []
    for key in ("global_orient_rotmat", "hand_pose_rotmat"):
        value = mano_params.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    out = np.concatenate(arrays)
    if not np.isfinite(out).all():
        return None
    return out


def _vector_distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=np.float32).reshape(-1)
    arr_b = np.asarray(b, dtype=np.float32).reshape(-1)
    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return None
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return None
    return float(np.linalg.norm(arr_a - arr_b))


def _median_vector(values: Any) -> np.ndarray | None:
    arrays = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values if value is not None]
    arrays = [arr for arr in arrays if arr.size and np.isfinite(arr).all()]
    if not arrays:
        return None
    first_shape = arrays[0].shape
    arrays = [arr for arr in arrays if arr.shape == first_shape]
    if not arrays:
        return None
    return np.median(np.stack(arrays, axis=0), axis=0)


def _median_vertices(values: Any) -> np.ndarray | None:
    arrays = [np.asarray(value, dtype=np.float32) for value in values if value is not None]
    arrays = [arr for arr in arrays if arr.ndim == 2 and arr.shape[1] == 3 and arr.size and np.isfinite(arr).all()]
    if not arrays:
        return None
    first_shape = arrays[0].shape
    arrays = [arr for arr in arrays if arr.shape == first_shape]
    if not arrays:
        return None
    return np.median(np.stack(arrays, axis=0), axis=0)


def _mesh_features(obs: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
    if not isinstance(obs, dict):
        return None
    mesh_obj = obs.get("mesh_obj")
    if not mesh_obj:
        return None
    vertices = _read_obj_vertices(str(mesh_obj))
    if vertices is None:
        return None
    return {"vertices": vertices, "centroid": vertices.mean(axis=0)}


@lru_cache(maxsize=8192)
def _read_obj_vertices(path_str: str) -> np.ndarray | None:
    path = Path(path_str).expanduser()
    if not path.is_file():
        return None
    vertices: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except (OSError, ValueError):
        return None
    if not vertices:
        return None
    out = np.asarray(vertices, dtype=np.float32)
    if not np.isfinite(out).all():
        return None
    return out


def _mesh_vertex_rmse(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 2 or arr_a.shape[1] != 3 or arr_a.size == 0:
        return None
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return None
    return float(np.linalg.norm(arr_a - arr_b) / np.sqrt(arr_a.size))


def _float_vector(values: Any, length: int) -> list[float] | None:
    if not isinstance(values, list) or len(values) < length:
        return None
    try:
        out = [float(v) for v in values[:length]]
    except (TypeError, ValueError):
        return None
    return out


def _intrinsic_drop_reasons(
    obs: dict[str, Any],
    *,
    min_cam_t_z: float,
    max_cam_t_z: float,
) -> list[str]:
    reasons: list[str] = []
    cam_t = obs.get("pred_cam_t_full")
    if cam_t is not None:
        arr = np.asarray(cam_t, dtype=np.float32)
        if not np.isfinite(arr).all():
            reasons.append("nonfinite_pred_cam_t_full")
        elif arr.shape[0] >= 3 and not (min_cam_t_z <= float(arr[2]) <= max_cam_t_z):
            reasons.append("pred_cam_t_z_out_of_range")
    mano_params = obs.get("mano_params")
    if isinstance(mano_params, dict):
        pose_arrays = []
        for key in ("global_orient_rotmat", "hand_pose_rotmat", "pred_cam"):
            value = mano_params.get(key)
            if value is not None:
                pose_arrays.append(np.asarray(value, dtype=np.float32))
        if any(arr.size and not np.isfinite(arr).all() for arr in pose_arrays):
            reasons.append("nonfinite_mano_pose")
    return reasons


def _cam_t_jump(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    prev = previous.get("pred_cam_t_full")
    cur = current.get("pred_cam_t_full")
    if prev is None or cur is None:
        return None
    prev_arr = np.asarray(prev, dtype=np.float32)
    cur_arr = np.asarray(cur, dtype=np.float32)
    if prev_arr.shape[0] < 3 or cur_arr.shape[0] < 3:
        return None
    if not np.isfinite(prev_arr).all() or not np.isfinite(cur_arr).all():
        return None
    return float(np.linalg.norm(cur_arr[:3] - prev_arr[:3]))


def _frame_index(frame: dict[str, Any]) -> int:
    for key in ("frame_idx", "frame_index", "frame_id", "frame", "index", "idx"):
        value = frame.get(key)
        if value is not None:
            return int(value)
    return 0
