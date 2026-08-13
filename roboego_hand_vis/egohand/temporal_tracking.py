from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SIDES = ("ego_left", "ego_right")


def as_float_list(values: Iterable[float], length: int) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float32).reshape(length)
    return arr.astype(float).tolist()


def build_recovered_depth_overrides_from_frames(
    report_frames: List[Dict[str, Any]],
) -> Dict[tuple[int, str], Dict[str, Any]]:
    normal_z_by_side: Dict[str, List[tuple[int, float]]] = {side: [] for side in SIDES}
    recovered_frames: List[tuple[int, str, float]] = []
    for frame in report_frames:
        frame_idx = int(frame["frame_idx"])
        for hand in frame.get("hands", []):
            side = hand.get("canonical_status") or hand.get("status")
            if side not in normal_z_by_side:
                continue
            transform = hand.get("shape_preserved_fit_transform") or {}
            target_z = transform.get("target_z_median")
            if target_z is None:
                target_anchor = transform.get("target_anchor")
                if target_anchor is not None:
                    target_z = np.asarray(target_anchor, dtype=np.float32).reshape(3)[2]
            if target_z is None:
                continue
            if hand.get("is_recovered"):
                recovered_frames.append((frame_idx, side, float(target_z)))
            else:
                normal_z_by_side[side].append((frame_idx, float(target_z)))

    overrides: Dict[tuple[int, str], Dict[str, Any]] = {}
    for frame_idx, side, observed_z in recovered_frames:
        normal = normal_z_by_side.get(side, [])
        if not normal:
            continue
        prev_item = None
        next_item = None
        for item in normal:
            if item[0] < frame_idx:
                prev_item = item
            elif item[0] > frame_idx:
                next_item = item
                break
        if prev_item is not None and next_item is not None:
            prev_frame, prev_z = prev_item
            next_frame, next_z = next_item
            ratio = (frame_idx - prev_frame) / float(next_frame - prev_frame)
            target_z = prev_z * (1.0 - ratio) + next_z * ratio
            source = {
                "mode": "interpolated_between_normal_hands",
                "prev_frame_idx": int(prev_frame),
                "next_frame_idx": int(next_frame),
                "ratio": float(ratio),
            }
        elif prev_item is not None:
            prev_frame, target_z = prev_item
            source = {"mode": "previous_normal_hand", "prev_frame_idx": int(prev_frame)}
        else:
            next_frame, target_z = next_item
            source = {"mode": "next_normal_hand", "next_frame_idx": int(next_frame)}
        overrides[(frame_idx, side)] = {
            "frame_idx": int(frame_idx),
            "side": side,
            "observed_target_z_m": float(observed_z),
            "target_z_m": float(target_z),
            "delta_from_observed_m": float(target_z - observed_z),
            **source,
        }
    return overrides


def apply_recovered_depth_overrides(
    report_frames: List[Dict[str, Any]],
    overrides: Dict[tuple[int, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = deepcopy(report_frames)
    if not overrides:
        return out
    for frame in out:
        frame_idx = int(frame["frame_idx"])
        for hand in frame.get("hands", []):
            side = hand.get("canonical_status") or hand.get("status")
            if not hand.get("is_recovered") or side not in SIDES:
                continue
            override = overrides.get((frame_idx, side))
            if override is None:
                continue
            transform = hand.setdefault("shape_preserved_fit_transform", {})
            target_z = float(override["target_z_m"])
            target_anchor = transform.get("target_anchor")
            if target_anchor is None:
                source_anchor = transform.get("source_anchor") or [0.0, 0.0, 0.0]
                translation = transform.get("translation") or [0.0, 0.0, 0.0]
                target_anchor = (
                    np.asarray(source_anchor, dtype=np.float32).reshape(3)
                    + np.asarray(translation, dtype=np.float32).reshape(3)
                ).astype(float).tolist()
            target_anchor = as_float_list(target_anchor, 3)
            observed_z = float(target_anchor[2])
            target_anchor[2] = target_z
            transform["target_anchor"] = target_anchor
            transform["target_z_observed_median"] = transform.get("target_z_median", observed_z)
            transform["target_z_median"] = target_z
            transform["target_z_source"] = "recovered_depth_override"
            transform["recovered_depth_override"] = override
            source_anchor = transform.get("source_anchor")
            if source_anchor is not None:
                source_arr = np.asarray(source_anchor, dtype=np.float32).reshape(3)
                target_arr = np.asarray(target_anchor, dtype=np.float32).reshape(3)
                transform["translation"] = (target_arr - source_arr).astype(float).tolist()
    return out


def hand_translation(hand: Dict[str, Any]) -> Optional[List[float]]:
    transform = hand.get("shape_preserved_fit_transform") or hand.get("smooth_fit_transform")
    if not transform:
        return None
    translation = transform.get("translation")
    if translation is None:
        return None
    return as_float_list(translation, 3)


def observation_from_hand(frame_idx: int, hand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    side = hand.get("canonical_status") or hand.get("status")
    if side not in SIDES:
        return None
    translation = hand_translation(hand)
    if translation is None:
        return None
    shape_transform = hand.get("shape_preserved_fit_transform") or {}
    source_anchor = shape_transform.get("source_anchor") or shape_transform.get("source_centroid")
    target_anchor = shape_transform.get("target_anchor") or shape_transform.get("target_centroid")
    if source_anchor is None:
        source_anchor = [0.0, 0.0, 0.0]
    if target_anchor is None:
        source = np.asarray(source_anchor, dtype=np.float32).reshape(3)
        target_anchor = (source + np.asarray(translation, dtype=np.float32).reshape(3)).astype(float).tolist()
    status = "recovered" if hand.get("is_recovered") else "detected"
    mesh_to_depth = hand.get("shape_preserved_mesh_to_depth_error") or {}
    depth_to_mesh = hand.get("shape_preserved_depth_to_mesh_error") or {}
    bbox = as_float_list(hand.get("bbox_xyxy", [0, 0, 0, 0]), 4)
    bbox_area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return {
        "frame_idx": int(frame_idx),
        "side": side,
        "status": status,
        "source": status,
        "candidate_id": hand.get("candidate_id"),
        "bbox_xyxy": bbox,
        "bbox_area_ratio": float(bbox_area / (512.0 * 512.0)),
        "mesh_obj": hand.get("mesh_obj"),
        "translation": translation,
        "smoothed_translation": translation,
        "source_anchor": as_float_list(source_anchor, 3),
        "target_anchor": as_float_list(target_anchor, 3),
        "smoothed_target_anchor": as_float_list(target_anchor, 3),
        "mesh_to_depth_median_m": mesh_to_depth.get("median_m"),
        "depth_to_mesh_median_m": depth_to_mesh.get("median_m"),
        "valid_depth_ratio": (hand.get("mask_stats") or {}).get("valid_depth_ratio"),
        "anomaly_flags": [],
        "source_hand": hand,
    }


def build_side_tracks(
    report_frames: List[Dict[str, Any]],
    start: int,
    num_frames: int,
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    tracks: Dict[str, List[Optional[Dict[str, Any]]]] = {
        side: [None for _ in range(num_frames)] for side in SIDES
    }
    frame_to_offset = {start + i: i for i in range(num_frames)}
    for frame in report_frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx not in frame_to_offset:
            continue
        offset = frame_to_offset[frame_idx]
        best_by_side: Dict[str, Dict[str, Any]] = {}
        for hand in frame.get("hands", []):
            obs = observation_from_hand(frame_idx, hand)
            if obs is None:
                continue
            side = obs["side"]
            current = best_by_side.get(side)
            if current is None:
                best_by_side[side] = obs
                continue
            current_error = current.get("mesh_to_depth_median_m")
            obs_error = obs.get("mesh_to_depth_median_m")
            if current_error is None or (obs_error is not None and obs_error < current_error):
                best_by_side[side] = obs
        for side, obs in best_by_side.items():
            tracks[side][offset] = obs
    return tracks


def lerp(a: Iterable[float], b: Iterable[float], ratio: float) -> List[float]:
    arr_a = np.asarray(list(a), dtype=np.float32)
    arr_b = np.asarray(list(b), dtype=np.float32)
    return (arr_a * (1.0 - ratio) + arr_b * ratio).astype(float).tolist()


def interpolate_short_gaps(
    track: List[Optional[Dict[str, Any]]],
    max_gap: int,
) -> List[Optional[Dict[str, Any]]]:
    out = [deepcopy(item) if item is not None else None for item in track]
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        gap_start = i
        while i < n and out[i] is None:
            i += 1
        gap_end = i - 1
        prev_idx = gap_start - 1
        next_idx = i
        gap_len = gap_end - gap_start + 1
        if prev_idx < 0 or next_idx >= n or gap_len > max_gap:
            continue
        prev_obs = out[prev_idx]
        next_obs = out[next_idx]
        if prev_obs is None or next_obs is None:
            continue
        for j in range(gap_start, gap_end + 1):
            ratio = (j - prev_idx) / float(next_idx - prev_idx)
            use_prev_shape = (j - prev_idx) <= (next_idx - j)
            shape_source = prev_obs if use_prev_shape else next_obs
            obs = deepcopy(shape_source)
            obs["frame_idx"] = int(j + int(prev_obs["frame_idx"]) - prev_idx)
            obs["status"] = "interpolated"
            obs["source"] = "interpolated_between_neighbors"
            obs["interpolation"] = {
                "prev_frame_idx": int(prev_obs["frame_idx"]),
                "next_frame_idx": int(next_obs["frame_idx"]),
                "ratio": float(ratio),
                "shape_source_frame_idx": int(shape_source["frame_idx"]),
            }
            obs["target_anchor"] = lerp(prev_obs["target_anchor"], next_obs["target_anchor"], ratio)
            # The target pose is temporal, but the MANO mesh comes from a
            # concrete neighboring frame. Keep that mesh's own anchor as the
            # translation reference instead of interpolating anchors from two
            # different meshes.
            obs["source_anchor"] = list(shape_source["source_anchor"])
            translation = (
                np.asarray(obs["target_anchor"], dtype=np.float32)
                - np.asarray(obs["source_anchor"], dtype=np.float32)
            )
            obs["translation"] = translation.astype(float).tolist()
            obs["smoothed_translation"] = list(obs["translation"])
            obs["smoothed_target_anchor"] = list(obs["target_anchor"])
            obs["bbox_xyxy"] = lerp(prev_obs["bbox_xyxy"], next_obs["bbox_xyxy"], ratio)
            obs["anomaly_flags"] = ["interpolated"]
            out[j] = obs
    return out


def make_edge_interpolation_observation(obs: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    out = deepcopy(obs)
    out["frame_idx"] = int(frame_idx)
    out["status"] = "interpolated"
    out["source"] = "interpolated_from_edge_neighbor"
    out["interpolation"] = {
        "mode": "edge_nearest_neighbor",
        "source_frame_idx": int(obs["frame_idx"]),
    }
    flags = list(out.get("anomaly_flags", []))
    flags.append("interpolated")
    flags.append("interpolated_edge")
    out["anomaly_flags"] = sorted(set(flags))
    return out


def fill_edge_gaps(
    track: List[Optional[Dict[str, Any]]],
    max_gap: int,
) -> List[Optional[Dict[str, Any]]]:
    out = [deepcopy(item) if item is not None else None for item in track]
    if not out or max_gap <= 0:
        return out

    first_present = next((idx for idx, obs in enumerate(out) if obs is not None), None)
    if first_present is not None and 0 < first_present <= max_gap:
        source = out[first_present]
        for idx in range(first_present):
            out[idx] = make_edge_interpolation_observation(source, idx)

    last_present = next((idx for idx in range(len(out) - 1, -1, -1) if out[idx] is not None), None)
    if last_present is not None:
        trailing_gap = len(out) - 1 - last_present
        if 0 < trailing_gap <= max_gap:
            source = out[last_present]
            for idx in range(last_present + 1, len(out)):
                out[idx] = make_edge_interpolation_observation(source, idx)
    return out


def translation_distance(a: Dict[str, Any], b: Dict[str, Any], key: str = "target_anchor") -> float:
    va = np.asarray(a[key], dtype=np.float32)
    vb = np.asarray(b[key], dtype=np.float32)
    return float(np.linalg.norm(vb - va))


def target_anchor_distance(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.get("target_anchor") is None or b.get("target_anchor") is None:
        return None
    return translation_distance(a, b)


def bbox_center_distance_px(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Optional[float]:
    if a is None or b is None:
        return None
    bbox_a = a.get("bbox_xyxy")
    bbox_b = b.get("bbox_xyxy")
    if bbox_a is None or bbox_b is None:
        return None
    arr_a = np.asarray(bbox_a, dtype=np.float32).reshape(4)
    arr_b = np.asarray(bbox_b, dtype=np.float32).reshape(4)
    center_a = np.array([(arr_a[0] + arr_a[2]) * 0.5, (arr_a[1] + arr_a[3]) * 0.5], dtype=np.float32)
    center_b = np.array([(arr_b[0] + arr_b[2]) * 0.5, (arr_b[1] + arr_b[3]) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(center_b - center_a))


def relabel_observation_side(obs: Dict[str, Any], side: str, reason: str) -> Dict[str, Any]:
    out = deepcopy(obs)
    out["side"] = side
    flags = list(out.get("anomaly_flags", []))
    flags.append(reason)
    out["anomaly_flags"] = sorted(set(flags))
    return out


def apply_identity_and_jump_gate(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    *,
    enable_lr_identity_gate: bool = False,
    drop_large_jumps: bool = False,
    large_jump_threshold_m: float = 0.12,
    max_bbox_center_jump_px: float = 200.0,
    swap_margin_m: float = 0.05,
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    out: Dict[str, List[Optional[Dict[str, Any]]]] = {
        side: [deepcopy(item) if item is not None else None for item in tracks[side]] for side in SIDES
    }
    if not out[SIDES[0]]:
        return out

    previous_by_side: Dict[str, Optional[Dict[str, Any]]] = {side: None for side in SIDES}
    for idx in range(len(out[SIDES[0]])):
        left = out["ego_left"][idx]
        right = out["ego_right"][idx]

        if enable_lr_identity_gate and left is not None and right is not None:
            prev_left = previous_by_side["ego_left"]
            prev_right = previous_by_side["ego_right"]
            if prev_left is not None and prev_right is not None:
                keep_left = target_anchor_distance(prev_left, left)
                keep_right = target_anchor_distance(prev_right, right)
                swap_left = target_anchor_distance(prev_left, right)
                swap_right = target_anchor_distance(prev_right, left)
                if None not in (keep_left, keep_right, swap_left, swap_right):
                    keep_cost = float(keep_left + keep_right)
                    swap_cost = float(swap_left + swap_right)
                    if swap_cost + float(swap_margin_m) < keep_cost:
                        out["ego_left"][idx] = relabel_observation_side(right, "ego_left", "lr_identity_swapped")
                        out["ego_right"][idx] = relabel_observation_side(left, "ego_right", "lr_identity_swapped")
                        left = out["ego_left"][idx]
                        right = out["ego_right"][idx]

        if drop_large_jumps:
            for side in SIDES:
                obs = out[side][idx]
                prev_obs = previous_by_side[side]
                jump = target_anchor_distance(prev_obs, obs)
                bbox_jump = bbox_center_distance_px(prev_obs, obs)
                has_large_3d_jump = jump is not None and jump > large_jump_threshold_m
                has_large_bbox_jump = (
                    max_bbox_center_jump_px > 0
                    and bbox_jump is not None
                    and bbox_jump > max_bbox_center_jump_px
                )
                if has_large_3d_jump or has_large_bbox_jump:
                    out[side][idx] = None

        for side in SIDES:
            if out[side][idx] is not None:
                previous_by_side[side] = out[side][idx]
    return out


def nearest_present(track: List[Optional[Dict[str, Any]]], start: int, step: int) -> Optional[Dict[str, Any]]:
    i = start
    while 0 <= i < len(track):
        if track[i] is not None:
            return track[i]
        i += step
    return None


def is_unstable_recovered_candidate(
    track: List[Optional[Dict[str, Any]]],
    index: int,
    jump_threshold_m: float,
    max_area_ratio: Optional[float],
) -> bool:
    obs = track[index]
    if obs is None or obs.get("status") != "recovered":
        return False
    if max_area_ratio is not None and float(obs.get("bbox_area_ratio", 1.0)) > max_area_ratio:
        return False

    prev_obs = nearest_present(track, index - 1, -1)
    next_obs = nearest_present(track, index + 1, 1)
    jumps = []
    if prev_obs is not None:
        jumps.append(translation_distance(prev_obs, obs))
    if next_obs is not None:
        jumps.append(translation_distance(obs, next_obs))
    if not jumps:
        return False
    return min(jumps) > jump_threshold_m


def filter_unstable_recovered_candidates(
    track: List[Optional[Dict[str, Any]]],
    jump_threshold_m: float,
    max_area_ratio: Optional[float] = None,
    max_passes: int = 8,
) -> List[Optional[Dict[str, Any]]]:
    out = [deepcopy(item) if item is not None else None for item in track]
    for _ in range(max(1, max_passes)):
        dropped = []
        for idx in range(len(out)):
            if is_unstable_recovered_candidate(
                out,
                idx,
                jump_threshold_m=jump_threshold_m,
                max_area_ratio=max_area_ratio,
            ):
                dropped.append(idx)
        if not dropped:
            break
        for idx in dropped:
            out[idx] = None
    return out


def add_detection_anomaly_flags(
    obs: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    jump_threshold_m: float,
    mesh_error_threshold_m: float,
    min_valid_depth_ratio: float,
) -> None:
    flags = list(obs.get("anomaly_flags", []))
    if previous is not None:
        jump = translation_distance(previous, obs)
        obs["raw_translation_jump_m"] = jump
        if jump > jump_threshold_m:
            flags.append("large_translation_jump")
    else:
        obs["raw_translation_jump_m"] = None

    mesh_error = obs.get("mesh_to_depth_median_m")
    if mesh_error is not None and float(mesh_error) > mesh_error_threshold_m:
        flags.append("large_mesh_depth_error")
    valid_depth_ratio = obs.get("valid_depth_ratio")
    if valid_depth_ratio is not None and float(valid_depth_ratio) < min_valid_depth_ratio:
        flags.append("low_valid_depth_ratio")
    obs["anomaly_flags"] = sorted(set(flags))


def smooth_track_translations(
    track: List[Optional[Dict[str, Any]]],
    alpha: float,
    jump_threshold_m: float = 0.08,
    mesh_error_threshold_m: float = 0.05,
    min_valid_depth_ratio: float = 0.5,
) -> List[Optional[Dict[str, Any]]]:
    out = [deepcopy(item) if item is not None else None for item in track]
    previous_obs: Optional[Dict[str, Any]] = None
    previous_anchor: Optional[np.ndarray] = None
    for obs in out:
        if obs is None:
            continue
        add_detection_anomaly_flags(
            obs,
            previous_obs,
            jump_threshold_m=jump_threshold_m,
            mesh_error_threshold_m=mesh_error_threshold_m,
            min_valid_depth_ratio=min_valid_depth_ratio,
        )
        current_anchor = np.asarray(obs["target_anchor"], dtype=np.float32)
        source_anchor = np.asarray(obs["source_anchor"], dtype=np.float32)
        smoothed_translation = current_anchor - source_anchor
        obs["smoothed_target_anchor"] = current_anchor.astype(float).tolist()
        obs["smoothed_translation"] = smoothed_translation.astype(float).tolist()
        if previous_anchor is not None:
            obs["smoothed_translation_jump_m"] = float(np.linalg.norm(current_anchor - previous_anchor))
        else:
            obs["smoothed_translation_jump_m"] = None
        previous_anchor = current_anchor
        previous_obs = obs
    return out


def process_tracks(
    report_frames: List[Dict[str, Any]],
    start: int,
    num_frames: int,
    max_interp_gap: int = 5,
    smooth_alpha: float = 0.35,
    jump_threshold_m: float = 0.08,
    mesh_error_threshold_m: float = 0.05,
    min_valid_depth_ratio: float = 0.5,
    drop_unstable_recovered: bool = False,
    unstable_recovered_jump_threshold_m: float = 0.08,
    unstable_recovered_max_area_ratio: Optional[float] = None,
    enable_lr_identity_gate: bool = False,
    drop_large_jumps: bool = False,
    large_jump_threshold_m: float = 0.12,
    max_bbox_center_jump_px: float = 200.0,
    lr_swap_margin_m: float = 0.05,
    fill_edge_gaps_enabled: bool = False,
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    tracks = build_side_tracks(report_frames, start=start, num_frames=num_frames)
    tracks = apply_identity_and_jump_gate(
        tracks,
        enable_lr_identity_gate=enable_lr_identity_gate,
        drop_large_jumps=drop_large_jumps,
        large_jump_threshold_m=large_jump_threshold_m,
        max_bbox_center_jump_px=max_bbox_center_jump_px,
        swap_margin_m=lr_swap_margin_m,
    )
    processed = {}
    for side, track in tracks.items():
        if drop_unstable_recovered:
            track = filter_unstable_recovered_candidates(
                track,
                jump_threshold_m=unstable_recovered_jump_threshold_m,
                max_area_ratio=unstable_recovered_max_area_ratio,
            )
        filled = interpolate_short_gaps(track, max_gap=max_interp_gap)
        if fill_edge_gaps_enabled:
            filled = fill_edge_gaps(filled, max_gap=max_interp_gap)
        processed[side] = smooth_track_translations(
            filled,
            alpha=smooth_alpha,
            jump_threshold_m=jump_threshold_m,
            mesh_error_threshold_m=mesh_error_threshold_m,
            min_valid_depth_ratio=min_valid_depth_ratio,
        )
    return processed


def track_summary(tracks: Dict[str, List[Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for side, track in tracks.items():
        counts = {
            "detected": 0,
            "recovered": 0,
            "interpolated": 0,
            "missing": 0,
            "anomaly": 0,
        }
        jumps = []
        for obs in track:
            if obs is None:
                counts["missing"] += 1
                continue
            status = obs.get("status", "detected")
            if status in counts:
                counts[status] += 1
            if obs.get("anomaly_flags"):
                non_interp_flags = [f for f in obs["anomaly_flags"] if f != "interpolated"]
                if non_interp_flags:
                    counts["anomaly"] += 1
            jump = obs.get("smoothed_translation_jump_m")
            if jump is not None:
                jumps.append(float(jump))
        side_summary: Dict[str, Any] = dict(counts)
        side_summary["present"] = len(track) - counts["missing"]
        side_summary["total"] = len(track)
        side_summary["present_ratio"] = side_summary["present"] / max(1, len(track))
        if jumps:
            side_summary["smoothed_jump_median_m"] = float(np.median(jumps))
            side_summary["smoothed_jump_p90_m"] = float(np.percentile(jumps, 90))
            side_summary["smoothed_jump_max_m"] = float(np.max(jumps))
        summary[side] = side_summary
    return summary
