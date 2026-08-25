from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .depth_mesh import hamer_rendered_obj_to_camera_opencv, parse_obj_mesh
from .joints21 import HAND21_BONES, HAND21_NAMES
from .orientation import vertices_to_hamer21_joints
from .temporal_tracking import SIDES


FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def build_joints21_pose_quality(
    final_tracks_payload: dict[str, Any],
    *,
    start: int,
    num_frames: int,
    regressor_by_side: dict[str, np.ndarray] | None = None,
    max_bone_length_ratio: float = 8.0,
    max_joint_local_deviation: float = 7.0,
    max_wrist_local_deviation_m: float = 0.0,
    max_palm_local_deviation_m: float = 0.0,
    max_fingertip_local_deviation_m: float = 0.0,
    min_finger_direction_dot: float = -0.15,
    local_window_radius: int = 2,
) -> dict[str, Any]:
    frames = final_tracks_payload.get("frames") or []
    by_frame = {_frame_index(frame): frame for frame in frames if isinstance(frame, dict)}
    joints_by_side = {side: np.full((num_frames, 21, 3), np.nan, dtype=np.float32) for side in SIDES}
    original_by_side: dict[str, list[dict[str, Any] | None]] = {side: [] for side in SIDES}

    for offset, frame_idx in enumerate(range(start, start + num_frames)):
        frame = by_frame.get(frame_idx, {})
        hands = frame.get("hands") or {}
        for side in SIDES:
            hand = hands.get(side)
            original_by_side[side].append(hand if isinstance(hand, dict) else None)
            joints = _extract_hand_joints21(hand, side=side, regressor_by_side=regressor_by_side)
            if joints is not None:
                joints_by_side[side][offset] = joints

    local_refs = {
        side: _local_joint_references(joints_by_side[side], original_by_side[side], radius=local_window_radius)
        for side in SIDES
    }
    result_frames: list[dict[str, Any]] = []
    gated_tracks: dict[str, list[dict[str, Any] | None]] = {side: [] for side in SIDES}
    candidates_frames: list[dict[str, Any]] = []
    summary = {
        side: {"kept": 0, "missing": 0, "dropped": 0, "drop_reasons": {}, "total": int(num_frames)}
        for side in SIDES
    }

    for offset, frame_idx in enumerate(range(start, start + num_frames)):
        frame_out: dict[str, Any] = {"frame_idx": int(frame_idx), "hands": {}}
        candidates: list[dict[str, Any]] = []
        for side in SIDES:
            hand = original_by_side[side][offset]
            joints = joints_by_side[side][offset]
            if hand is None or not _valid_joints(joints):
                summary[side]["missing"] += 1
                frame_out["hands"][side] = {"valid": False, "drop_reasons": ["missing_joints21"]}
                gated_tracks[side].append(None)
                continue
            metrics = _quality_metrics(joints, local_refs[side][offset])
            reasons = _drop_reasons(
                metrics,
                max_bone_length_ratio=max_bone_length_ratio,
                max_joint_local_deviation=max_joint_local_deviation,
                max_wrist_local_deviation_m=max_wrist_local_deviation_m,
                max_palm_local_deviation_m=max_palm_local_deviation_m,
                max_fingertip_local_deviation_m=max_fingertip_local_deviation_m,
                min_finger_direction_dot=min_finger_direction_dot,
            )
            if reasons:
                summary[side]["dropped"] += 1
                for reason in reasons:
                    summary[side]["drop_reasons"][reason] = summary[side]["drop_reasons"].get(reason, 0) + 1
                frame_out["hands"][side] = {
                    "valid": False,
                    "candidate_id": hand.get("candidate_id"),
                    "drop_reasons": reasons,
                    "metrics": _jsonable(metrics),
                }
                gated_tracks[side].append(None)
                continue
            kept = deepcopy(hand)
            kept["joints21_pose_quality"] = {"valid": True, "metrics": _jsonable(metrics)}
            frame_out["hands"][side] = {
                "valid": True,
                "candidate_id": hand.get("candidate_id"),
                "metrics": _jsonable(metrics),
            }
            gated_tracks[side].append(kept)
            candidates.append(_candidate_from_final_hand(kept, side=side))
            summary[side]["kept"] += 1
        result_frames.append(frame_out)
        candidates_frames.append({"frame_idx": int(frame_idx), "candidates": candidates})

    return {
        "stage": "joints21_pose_quality_gate_v1",
        "parameters": {
            "start": int(start),
            "num_frames": int(num_frames),
            "max_bone_length_ratio": float(max_bone_length_ratio),
            "max_joint_local_deviation": float(max_joint_local_deviation),
            "max_wrist_local_deviation_m": float(max_wrist_local_deviation_m),
            "max_palm_local_deviation_m": float(max_palm_local_deviation_m),
            "max_fingertip_local_deviation_m": float(max_fingertip_local_deviation_m),
            "min_finger_direction_dot": float(min_finger_direction_dot),
            "local_window_radius": int(local_window_radius),
        },
        "joint_names": HAND21_NAMES,
        "bones": HAND21_BONES,
        "frames": result_frames,
        "tracks": gated_tracks,
        "summary": summary,
        "joints_by_side": joints_by_side,
        "final_hand_candidates_gated": {
            "stage": "final_hand_candidates_joints21_pose_gated_v1",
            "frame_range": {"start": int(start), "num_frames": int(num_frames), "end_exclusive": int(start + num_frames)},
            "frames": candidates_frames,
        },
    }


def write_joints21_pose_quality_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "joints21_compact.npz"
    np.savez_compressed(
        npz_path,
        left_joints21_camera_opencv=result["joints_by_side"]["ego_left"],
        right_joints21_camera_opencv=result["joints_by_side"]["ego_right"],
        joint_names=np.asarray(HAND21_NAMES),
        bones=np.asarray(HAND21_BONES, dtype=np.int32),
    )
    report_path = output_dir / "pose_quality_report.json"
    report = {key: value for key, value in result.items() if key not in {"joints_by_side", "tracks", "final_hand_candidates_gated"}}
    report["outputs"] = {
        "joints21_compact_npz": str(npz_path),
        "pose_quality_report_json": str(report_path),
        "final_hand_candidates_gated_json": str(output_dir / "final_hand_candidates_gated.json"),
    }
    report_path.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    candidates_path = output_dir / "final_hand_candidates_gated.json"
    candidates_path.write_text(
        json.dumps(_jsonable(result["final_hand_candidates_gated"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report["outputs"]


def _extract_hand_joints21(
    hand: Any,
    *,
    side: str,
    regressor_by_side: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    if not isinstance(hand, dict):
        return None
    direct = hand.get("joints21_camera_opencv") or hand.get("joints21")
    if direct is not None:
        joints = np.asarray(direct, dtype=np.float32)
        return joints if _valid_joints(joints) else None
    if regressor_by_side is None or side not in regressor_by_side:
        return None
    mesh_obj = hand.get("mesh_obj") or hand.get("source_mesh_obj")
    if not mesh_obj:
        return None
    mesh_path = Path(str(mesh_obj)).expanduser()
    if not mesh_path.is_file():
        return None
    vertices_obj, _faces, _colors = parse_obj_mesh(mesh_path)
    if vertices_obj.shape != (778, 3):
        return None
    vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
    return vertices_to_hamer21_joints(vertices_camera, regressor_by_side[side])


def _quality_metrics(joints: np.ndarray, local_ref: dict[str, Any]) -> dict[str, Any]:
    bone_lengths = _bone_lengths(joints)
    finite_lengths = bone_lengths[np.isfinite(bone_lengths) & (bone_lengths > 1e-6)]
    median_length = float(np.median(finite_lengths)) if len(finite_lengths) else None
    max_ratio = float(np.max(finite_lengths) / max(float(np.median(finite_lengths)), 1e-6)) if len(finite_lengths) else None
    normalized = _normalized_joints(joints)
    local_deviation = _joint_rmse(normalized, local_ref.get("normalized_joints"))
    finger_dots = _finger_direction_dots(joints, local_ref.get("finger_directions"))
    wrist_deviation = _point_distance(joints[0], local_ref.get("wrist"))
    palm_deviation = _point_distance(_palm_points(joints).mean(axis=0), local_ref.get("palm"))
    fingertip_deviations = _pointwise_distances(_fingertip_points(joints), local_ref.get("fingertips"))
    return {
        "median_bone_length_m": median_length,
        "max_bone_length_ratio": max_ratio,
        "joint_local_deviation": local_deviation,
        "wrist_local_deviation_m": wrist_deviation,
        "palm_local_deviation_m": palm_deviation,
        "fingertip_local_deviation_max_m": max(fingertip_deviations) if fingertip_deviations else None,
        "fingertip_local_deviation_mean_m": float(np.mean(fingertip_deviations)) if fingertip_deviations else None,
        "finger_direction_min_dot": min(finger_dots) if finger_dots else None,
        "finger_direction_mean_dot": float(np.mean(finger_dots)) if finger_dots else None,
        "local_reference_count": int(local_ref.get("count", 0)),
    }


def _drop_reasons(
    metrics: dict[str, Any],
    *,
    max_bone_length_ratio: float,
    max_joint_local_deviation: float,
    max_wrist_local_deviation_m: float,
    max_palm_local_deviation_m: float,
    max_fingertip_local_deviation_m: float,
    min_finger_direction_dot: float,
) -> list[str]:
    reasons: list[str] = []
    bone_ratio = metrics.get("max_bone_length_ratio")
    if max_bone_length_ratio > 0 and bone_ratio is not None and float(bone_ratio) > max_bone_length_ratio:
        reasons.append("bone_length_outlier")
    local_count = int(metrics.get("local_reference_count", 0))
    local_deviation = metrics.get("joint_local_deviation")
    if (
        local_count >= 2
        and max_joint_local_deviation > 0
        and local_deviation is not None
        and float(local_deviation) > max_joint_local_deviation
    ):
        reasons.append("large_joint_local_deviation")
    wrist_deviation = metrics.get("wrist_local_deviation_m")
    if (
        local_count >= 2
        and max_wrist_local_deviation_m > 0
        and wrist_deviation is not None
        and float(wrist_deviation) > max_wrist_local_deviation_m
    ):
        reasons.append("large_wrist_trajectory_deviation")
    palm_deviation = metrics.get("palm_local_deviation_m")
    if (
        local_count >= 2
        and max_palm_local_deviation_m > 0
        and palm_deviation is not None
        and float(palm_deviation) > max_palm_local_deviation_m
    ):
        reasons.append("large_palm_trajectory_deviation")
    fingertip_deviation = metrics.get("fingertip_local_deviation_max_m")
    if (
        local_count >= 2
        and max_fingertip_local_deviation_m > 0
        and fingertip_deviation is not None
        and float(fingertip_deviation) > max_fingertip_local_deviation_m
    ):
        reasons.append("large_fingertip_trajectory_deviation")
    finger_dot = metrics.get("finger_direction_min_dot")
    if local_count >= 2 and finger_dot is not None and float(finger_dot) < min_finger_direction_dot:
        reasons.append("finger_direction_flip")
    return reasons


def _local_joint_references(
    joints_seq: np.ndarray,
    hands: list[dict[str, Any] | None],
    *,
    radius: int,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx in range(len(joints_seq)):
        neighbors = []
        for other_idx in range(max(0, idx - radius), min(len(joints_seq), idx + radius + 1)):
            if other_idx == idx or hands[other_idx] is None:
                continue
            joints = joints_seq[other_idx]
            if _valid_joints(joints):
                neighbors.append(joints)
        normalized = [_normalized_joints(joints) for joints in neighbors]
        directions = [_finger_directions(joints) for joints in neighbors]
        wrists = [joints[0] for joints in neighbors]
        palms = [_palm_points(joints).mean(axis=0) for joints in neighbors]
        fingertips = [_fingertip_points(joints) for joints in neighbors]
        refs.append(
            {
                "count": len(neighbors),
                "normalized_joints": _median_stack(normalized),
                "finger_directions": _median_stack(directions),
                "wrist": _median_stack(wrists),
                "palm": _median_stack(palms),
                "fingertips": _median_stack(fingertips),
            }
        )
    return refs


def _bone_lengths(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    return np.asarray([np.linalg.norm(arr[end] - arr[start]) for start, end in HAND21_BONES], dtype=np.float32)


def _normalized_joints(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    centered = arr - arr[0:1]
    lengths = _bone_lengths(arr)
    finite = lengths[np.isfinite(lengths) & (lengths > 1e-6)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    return centered / max(scale, 1e-6)


def _finger_directions(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    vectors = []
    for chain in FINGER_CHAINS:
        vec = arr[chain[-1]] - arr[chain[0]]
        norm = float(np.linalg.norm(vec))
        vectors.append(vec / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32))
    return np.asarray(vectors, dtype=np.float32)


def _finger_direction_dots(joints: np.ndarray, reference: Any) -> list[float]:
    if reference is None:
        return []
    current = _finger_directions(joints)
    ref = np.asarray(reference, dtype=np.float32)
    if current.shape != ref.shape:
        return []
    dots = np.sum(current * ref, axis=1)
    return [float(value) for value in dots if np.isfinite(value)]


def _palm_points(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    return arr[[0, 5, 9, 13, 17]]


def _fingertip_points(joints: np.ndarray) -> np.ndarray:
    arr = np.asarray(joints, dtype=np.float32)
    return arr[[4, 8, 12, 16, 20]]


def _point_distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return None
    return float(np.linalg.norm(arr_a - arr_b))


def _pointwise_distances(a: Any, b: Any) -> list[float]:
    if a is None or b is None:
        return []
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 2:
        return []
    if not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return []
    distances = np.linalg.norm(arr_a - arr_b, axis=1)
    return [float(value) for value in distances if np.isfinite(value)]


def _joint_rmse(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    if arr_a.shape != arr_b.shape or not np.isfinite(arr_a).all() or not np.isfinite(arr_b).all():
        return None
    return float(np.linalg.norm(arr_a - arr_b) / np.sqrt(arr_a.size))


def _median_stack(values: list[np.ndarray]) -> np.ndarray | None:
    values = [np.asarray(value, dtype=np.float32) for value in values if value is not None and np.isfinite(value).all()]
    if not values:
        return None
    shape = values[0].shape
    values = [value for value in values if value.shape == shape]
    if not values:
        return None
    return np.median(np.stack(values, axis=0), axis=0).astype(np.float32)


def _candidate_from_final_hand(hand: dict[str, Any], *, side: str) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    for key in (
        "frame_idx",
        "candidate_id",
        "bbox_xyxy",
        "bbox_center_xy",
        "confidence",
        "detector_score",
        "keypoint_mean_conf",
        "valid_keypoints",
        "median_depth_m",
        "mesh_obj",
        "temporal_quality",
        "anomaly_flags",
    ):
        if hand.get(key) is not None:
            candidate[key] = hand.get(key)
    candidate["status"] = side
    candidate["side"] = side
    candidate["joints21_pose_quality"] = hand.get("joints21_pose_quality", {"valid": True})
    return candidate


def _valid_joints(joints: Any) -> bool:
    arr = np.asarray(joints, dtype=np.float32)
    return arr.shape == (21, 3) and np.isfinite(arr).all()


def _frame_index(frame: dict[str, Any]) -> int:
    return int(frame.get("frame_idx", frame.get("frame_index", frame.get("idx", 0))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
