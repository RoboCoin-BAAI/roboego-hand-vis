from __future__ import annotations

from copy import deepcopy
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
) -> dict[str, Any]:
    frames = refined_payload.get("frames") or refined_payload.get("results") or []
    by_frame = {_frame_index(frame): frame for frame in frames if isinstance(frame, dict)}
    tracks: dict[str, list[dict[str, Any] | None]] = {side: [] for side in SIDES}

    for frame_idx in range(start, start + num_frames):
        frame = by_frame.get(frame_idx, {})
        for side in SIDES:
            hand = frame.get(f"{side}_refined")
            tracks[side].append(_observation_from_refined(frame_idx, side, hand))

    flagged = {side: _flag_large_jumps(track, max_bbox_jump_px=max_bbox_jump_px) for side, track in tracks.items()}
    filled = {side: _interpolate_short_gaps(track, max_gap=max_gap) for side, track in flagged.items()}
    output_frames = _frames_from_tracks(filled, start=start, num_frames=num_frames)
    return {
        "stage": "refined_temporal_tracking_v1",
        "parameters": {
            "start": int(start),
            "num_frames": int(num_frames),
            "max_gap": int(max_gap),
            "max_bbox_jump_px": float(max_bbox_jump_px),
        },
        "tracks": filled,
        "frames": output_frames,
        "summary": _summary(filled),
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
        "mesh_obj": hand.get("mesh_obj"),
        "refine_quality": hand.get("refine_quality"),
        "refine_reasons": hand.get("refine_reasons", []),
        "refine_risks": hand.get("refine_risks", []),
        "anomaly_flags": [],
        "source_hand": hand,
    }


def _flag_large_jumps(
    track: list[dict[str, Any] | None],
    *,
    max_bbox_jump_px: float,
) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    previous: dict[str, Any] | None = None
    for obs in track:
        if obs is None:
            out.append(None)
            continue
        item = deepcopy(obs)
        if previous is not None:
            jump = _distance(item["bbox_center_xy"], previous["bbox_center_xy"])
            item["bbox_center_jump_px"] = float(jump)
            if jump > max_bbox_jump_px:
                item["anomaly_flags"] = sorted(set(item.get("anomaly_flags", []) + ["large_bbox_center_jump"]))
                item["temporal_quality"] = "jump_warning"
            else:
                item["temporal_quality"] = "continuous"
        else:
            item["temporal_quality"] = "track_start"
        out.append(item)
        previous = item
    return out


def _interpolate_short_gaps(
    track: list[dict[str, Any] | None],
    *,
    max_gap: int,
) -> list[dict[str, Any] | None]:
    out = [deepcopy(item) if item is not None else None for item in track]
    idx = 0
    while idx < len(out):
        if out[idx] is not None:
            idx += 1
            continue
        gap_start = idx
        while idx < len(out) and out[idx] is None:
            idx += 1
        gap_end = idx - 1
        prev_idx = gap_start - 1
        next_idx = idx
        gap_len = gap_end - gap_start + 1
        if prev_idx < 0 or next_idx >= len(out) or gap_len > max_gap:
            continue
        prev_obs = out[prev_idx]
        next_obs = out[next_idx]
        if prev_obs is None or next_obs is None:
            continue
        for fill_idx in range(gap_start, gap_end + 1):
            ratio = (fill_idx - prev_idx) / float(next_idx - prev_idx)
            source = prev_obs if ratio <= 0.5 else next_obs
            obs = deepcopy(source)
            obs["frame_idx"] = int(prev_obs["frame_idx"] + (fill_idx - prev_idx))
            obs["status"] = "interpolated"
            obs["source"] = "interpolated_between_refined_neighbors"
            obs["bbox_xyxy"] = _lerp(prev_obs["bbox_xyxy"], next_obs["bbox_xyxy"], ratio)
            obs["bbox_center_xy"] = _bbox_center(obs["bbox_xyxy"])
            obs["anomaly_flags"] = sorted(set(obs.get("anomaly_flags", []) + ["interpolated"]))
            obs["temporal_quality"] = "short_gap_interpolated"
            obs["interpolation"] = {
                "prev_frame_idx": int(prev_obs["frame_idx"]),
                "next_frame_idx": int(next_obs["frame_idx"]),
                "ratio": float(ratio),
                "shape_source_frame_idx": int(source["frame_idx"]),
            }
            out[fill_idx] = obs
    return out


def _frames_from_tracks(
    tracks: dict[str, list[dict[str, Any] | None]],
    *,
    start: int,
    num_frames: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for offset in range(num_frames):
        item: dict[str, Any] = {"frame_idx": int(start + offset), "hands": {}}
        for side in SIDES:
            obs = tracks[side][offset]
            item["hands"][side] = obs
        item["visibility_state"] = _visibility_state(item["hands"])
        out.append(item)
    return out


def _summary(tracks: dict[str, list[dict[str, Any] | None]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for side, track in tracks.items():
        normal = sum(1 for obs in track if obs is not None and obs.get("status") == "normal")
        interpolated = sum(1 for obs in track if obs is not None and obs.get("status") == "interpolated")
        missing = sum(1 for obs in track if obs is None)
        jumps = sum(1 for obs in track if obs is not None and "large_bbox_center_jump" in obs.get("anomaly_flags", []))
        summary[side] = {
            "normal": normal,
            "interpolated": interpolated,
            "missing": missing,
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


def _lerp(a: list[float], b: list[float], ratio: float) -> list[float]:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return (arr_a * (1.0 - ratio) + arr_b * ratio).astype(float).tolist()


def _frame_index(frame: dict[str, Any]) -> int:
    for key in ("frame_idx", "frame_index", "frame_id", "frame", "index", "idx"):
        value = frame.get(key)
        if value is not None:
            return int(value)
    return 0
