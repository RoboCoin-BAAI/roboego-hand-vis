from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .temporal_tracking import SIDES


def polish_depth_anchored_joints(
    payload: dict[str, Any],
    *,
    max_interp_gap: int = 10,
    wrist_jump_m: float = 0.0,
    max_local_wrist_deviation_m: float = 0.0,
    local_window_radius: int = 5,
    smooth_window: int = 9,
    smooth_relative_fingers: bool = True,
) -> dict[str, Any]:
    frames = payload.get("frames") or []
    frame_indices = [int(frame.get("frame_idx", idx)) for idx, frame in enumerate(frames)]
    output_frames = [{"frame_idx": frame_idx, "hands": {side: {"valid": False, "reason": "missing_candidate"} for side in SIDES}} for frame_idx in frame_indices]
    report: dict[str, Any] = {}

    for side in SIDES:
        joints, meta, initial_valid = _extract_side(frames, side)
        gated, kept, drop_reasons = _drop_wrist_outliers(
            joints,
            initial_valid,
            max_interp_gap=max_interp_gap,
            wrist_jump_m=wrist_jump_m,
            max_local_wrist_deviation_m=max_local_wrist_deviation_m,
            local_window_radius=local_window_radius,
        )
        filled, interpolated = _interpolate_small_gaps(gated, kept, max_gap=max_interp_gap)
        smoothed = _smooth_joints(filled, kept | interpolated, window=smooth_window, smooth_relative_fingers=smooth_relative_fingers)
        final_valid = kept | interpolated
        for offset, valid in enumerate(final_valid):
            if not bool(valid):
                reason = "dropped_by_trajectory_polish" if bool(initial_valid[offset]) else "missing_candidate"
                output_frames[offset]["hands"][side] = {"valid": False, "reason": reason}
                continue
            item = deepcopy(meta[offset]) if isinstance(meta[offset], dict) else {}
            item.update(
                {
                    "valid": True,
                    "status": "interpolated" if bool(interpolated[offset]) else "smoothed",
                    "joints21_camera_opencv": smoothed[offset].astype(float).tolist(),
                }
            )
            item["trajectory_polish"] = {
                "valid": True,
                "interpolated": bool(interpolated[offset]),
                "smoothed": bool(smooth_window > 1),
            }
            output_frames[offset]["hands"][side] = item
        report[side] = {
            "initial_valid": int(initial_valid.sum()),
            "dropped": int((initial_valid & ~kept).sum()),
            "interpolated": int(interpolated.sum()),
            "final_valid": int(final_valid.sum()),
            "smoothed": int(final_valid.sum()) if smooth_window > 1 else 0,
            "drop_reasons": drop_reasons,
            "intervals": _valid_intervals(frame_indices, final_valid),
        }

    return {
        "stage": "final_hand_trajectory_polish_v1",
        "coordinate_frame": payload.get("coordinate_frame", "camera_opencv"),
        "unit": payload.get("unit", "meter"),
        "joint_names": payload.get("joint_names"),
        "bones": payload.get("bones"),
        "parameters": {
            "max_interp_gap": int(max_interp_gap),
            "wrist_jump_m": float(wrist_jump_m),
            "max_local_wrist_deviation_m": float(max_local_wrist_deviation_m),
            "local_window_radius": int(local_window_radius),
            "smooth_window": int(smooth_window),
            "smooth_relative_fingers": bool(smooth_relative_fingers),
        },
        "report": report,
        "frames": output_frames,
    }


def _extract_side(frames: list[dict[str, Any]], side: str) -> tuple[np.ndarray, list[dict[str, Any] | None], np.ndarray]:
    joints = np.full((len(frames), 21, 3), np.nan, dtype=np.float32)
    meta: list[dict[str, Any] | None] = []
    valid = np.zeros((len(frames),), dtype=bool)
    for idx, frame in enumerate(frames):
        hand = (frame.get("hands") or {}).get(side)
        meta.append(hand if isinstance(hand, dict) else None)
        if not isinstance(hand, dict) or not hand.get("valid"):
            continue
        arr = np.asarray(hand.get("joints21_camera_opencv"), dtype=np.float32)
        if arr.shape == (21, 3) and np.isfinite(arr).all():
            joints[idx] = arr
            valid[idx] = True
    return joints, meta, valid


def _drop_wrist_outliers(
    joints: np.ndarray,
    valid: np.ndarray,
    *,
    max_interp_gap: int,
    wrist_jump_m: float,
    max_local_wrist_deviation_m: float,
    local_window_radius: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    out = joints.copy()
    kept = valid.copy()
    drop_reasons: dict[str, int] = {}
    previous_wrist: np.ndarray | None = None
    previous_idx: int | None = None
    wrists = joints[:, 0, :]
    for idx in range(len(joints)):
        if not bool(kept[idx]):
            continue
        reasons = []
        wrist = wrists[idx]
        if previous_wrist is not None and previous_idx is not None and wrist_jump_m > 0 and idx - previous_idx <= max_interp_gap + 1:
            if float(np.linalg.norm(wrist - previous_wrist)) > wrist_jump_m:
                reasons.append("large_wrist_jump")
        if max_local_wrist_deviation_m > 0:
            ref = _local_median_wrist(wrists, valid, idx, radius=local_window_radius)
            if ref is not None and float(np.linalg.norm(wrist - ref)) > max_local_wrist_deviation_m:
                reasons.append("large_local_wrist_deviation")
        if reasons:
            kept[idx] = False
            out[idx] = np.nan
            for reason in reasons:
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        previous_wrist = wrist
        previous_idx = idx
    return out, kept, drop_reasons


def _local_median_wrist(wrists: np.ndarray, valid: np.ndarray, idx: int, *, radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    start = max(0, idx - radius)
    end = min(len(wrists), idx + radius + 1)
    neighbors = [wrists[i] for i in range(start, end) if i != idx and bool(valid[i]) and np.isfinite(wrists[i]).all()]
    if len(neighbors) < 2:
        return None
    return np.median(np.stack(neighbors, axis=0), axis=0).astype(np.float32)


def _interpolate_small_gaps(joints: np.ndarray, valid: np.ndarray, *, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    out = joints.copy()
    interpolated = np.zeros_like(valid)
    if max_gap <= 0:
        return out, interpolated
    idx = 0
    while idx < len(valid):
        if bool(valid[idx]):
            idx += 1
            continue
        start = idx
        while idx < len(valid) and not bool(valid[idx]):
            idx += 1
        end = idx
        gap = end - start
        left = start - 1
        right = end
        if gap <= max_gap and left >= 0 and right < len(valid) and bool(valid[left]) and bool(valid[right]):
            for pos in range(start, end):
                alpha = (pos - left) / float(right - left)
                out[pos] = (1.0 - alpha) * out[left] + alpha * out[right]
                interpolated[pos] = True
    return out, interpolated


def _smooth_joints(joints: np.ndarray, valid: np.ndarray, *, window: int, smooth_relative_fingers: bool) -> np.ndarray:
    if window <= 1:
        return joints.copy()
    out = joints.copy()
    for start, end in _valid_offset_intervals(valid):
        segment = joints[start : end + 1]
        if len(segment) < 3:
            continue
        if smooth_relative_fingers:
            wrist = _moving_average(segment[:, 0:1, :], window)
            offsets = segment - segment[:, 0:1, :]
            out[start : end + 1] = wrist + _moving_average(offsets, max(3, min(window, 5)))
        else:
            out[start : end + 1] = _moving_average(segment, window)
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    radius = win // 2
    out = values.copy()
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        out[idx] = np.mean(values[start:end], axis=0)
    return out


def _valid_offset_intervals(valid: np.ndarray) -> list[tuple[int, int]]:
    return [(start, end) for start, end in _intervals(range(len(valid)), valid)]


def _valid_intervals(frame_indices: list[int], valid: np.ndarray) -> list[dict[str, int]]:
    return [{"start": int(start), "end": int(end)} for start, end in _intervals(frame_indices, valid)]


def _intervals(indices: Any, valid: np.ndarray) -> list[tuple[int, int]]:
    values = list(indices)
    out: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for value, is_valid in zip(values, valid):
        if bool(is_valid) and start is None:
            start = int(value)
        if not bool(is_valid) and start is not None:
            out.append((start, int(prev)))
            start = None
        prev = int(value)
    if start is not None and prev is not None:
        out.append((start, prev))
    return out
