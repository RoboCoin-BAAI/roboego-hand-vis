from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HIGH_PRIORITY_ACTIONS = {"grasp", "pick", "place", "touch", "operate", "press", "scan", "release"}
MEDIUM_PRIORITY_ACTIONS = {"hold", "point", "reach"}
CONTEXT_ACTIONS = {"hold", "carry", "support"}
SEMANTIC_LABEL_ACTION_PRIORITY = {
    "pick": 0,
    "grasp": 1,
    "place": 2,
    "put into": 3,
    "take out": 4,
    "scan": 5,
    "operate": 6,
    "open": 7,
    "close": 8,
    "press": 9,
    "touch": 10,
    "reach": 11,
    "point": 12,
    "push": 13,
    "pull": 14,
}


@dataclass(frozen=True)
class HandWindowConfig:
    require_both_hands: bool = True
    hand_window_mode: str = "two"
    context_expand_s: int = 1
    merge_gap_s: int = 2
    merge_after_expand: bool = False
    min_window_s: int = 1
    min_priority: str | None = None
    min_core_duration_s: int = 0
    fps: int = 30


def build_hand_estimation_windows(
    stage1_actions_payload: dict[str, Any],
    processing_plan_payload: dict[str, Any],
    *,
    config: HandWindowConfig = HandWindowConfig(),
) -> dict[str, Any]:
    frames = _frames_by_t(stage1_actions_payload)
    raw_windows: list[dict[str, Any]] = []
    for segment in processing_plan_payload.get("segments", []):
        grouped_candidates: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        for t, frame in frames.items():
            if not int(segment["start"]) <= t <= int(segment["end"]):
                continue
            group = _candidate_group(frame, config)
            if group is None:
                continue
            grouped_candidates.setdefault(group, []).append(frame)
        for (window_type, hands_required), candidates in grouped_candidates.items():
            runs = _merge_seconds([int(frame["t"]) for frame in candidates], max_gap=config.merge_gap_s)
            for core_start, core_end in runs:
                start = max(int(segment["start"]), core_start - config.context_expand_s)
                end = min(int(segment["end"]), core_end + config.context_expand_s)
                if end - start + 1 < config.min_window_s:
                    continue
                window_frames = [frames[t] for t in range(core_start, core_end + 1) if t in frames]
                raw_windows.append(
                    _window_from_range(
                        len(raw_windows),
                        segment,
                        start,
                        end,
                        core_start,
                        core_end,
                        window_frames,
                        fps=config.fps,
                        window_type=window_type,
                        hands_required=list(hands_required),
                    )
                )
    raw_windows.sort(key=lambda item: (int(item["start"]), int(item["end"]), item["window_type"], item["hands_required"]))
    if config.merge_after_expand:
        raw_windows = _merge_expanded_windows(raw_windows, fps=config.fps)
    raw_windows = _filter_windows(raw_windows, config=config)
    for idx, window in enumerate(raw_windows):
        window["window_id"] = f"handwin_{idx:04d}"
    return {
        "stage": "hand_estimation_windows",
        "source_stage1_actions": stage1_actions_payload.get("source_video", ""),
        "config": {
            "require_both_hands": config.require_both_hands,
            "hand_window_mode": config.hand_window_mode,
            "context_expand_s": config.context_expand_s,
            "merge_gap_s": config.merge_gap_s,
            "merge_after_expand": config.merge_after_expand,
            "min_window_s": config.min_window_s,
            "min_priority": config.min_priority,
            "min_core_duration_s": config.min_core_duration_s,
            "fps": config.fps,
        },
        "window_count": len(raw_windows),
        "windows": raw_windows,
    }


def write_hand_estimation_windows(
    *,
    stage1_actions_json: Path,
    processing_plan_json: Path,
    output_json: Path,
    config: HandWindowConfig = HandWindowConfig(),
) -> dict[str, Any]:
    stage1_payload = _load_stage1_actions_payload(Path(stage1_actions_json))
    plan_payload = json.loads(Path(processing_plan_json).read_text(encoding="utf-8"))
    payload = build_hand_estimation_windows(stage1_payload, plan_payload, config=config)
    payload["source_stage1_actions"] = str(stage1_actions_json)
    payload["source_processing_plan"] = str(processing_plan_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _load_stage1_actions_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        frames: list[dict[str, Any]] = []
        source_files: list[str] = []
        for chunk_path in sorted(path.glob("*.json")):
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            frames.extend(frame for frame in payload.get("frames", []) if isinstance(frame, dict))
            source_files.append(str(chunk_path))
        return {
            "stage": "stage1_action_chunks_merged",
            "source_files": source_files,
            "frame_count": len(frames),
            "frames": sorted(frames, key=lambda frame: float(frame.get("t", 0))),
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _frames_by_t(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        if isinstance(frame, dict) and "t" in frame:
            item = dict(frame)
            item["t"] = int(float(item["t"]))
            out[item["t"]] = item
    return out


def _is_candidate_frame(frame: dict[str, Any], config: HandWindowConfig) -> bool:
    return _candidate_group(frame, config) is not None


def _candidate_group(frame: dict[str, Any], config: HandWindowConfig) -> tuple[str, tuple[str, ...]] | None:
    hands = frame.get("hands_visible") or {}
    left = bool(hands.get("left"))
    right = bool(hands.get("right"))
    meaningful = bool(frame.get("hand_object_interaction")) or bool(_action_names(frame)) or (left and right)
    if not meaningful:
        return None
    mode = config.hand_window_mode
    if mode == "two":
        return ("two_hand", ("left", "right")) if left and right else None
    if mode == "single":
        if left and not right:
            return ("single_hand", ("left",))
        if right and not left:
            return ("single_hand", ("right",))
        return None
    if mode == "single_and_two":
        if left and right:
            return ("two_hand", ("left", "right"))
        if left:
            return ("single_hand", ("left",))
        if right:
            return ("single_hand", ("right",))
        return None
    if mode == "any" or not config.require_both_hands:
        if left and right:
            return ("two_hand", ("left", "right"))
        if left:
            return ("single_hand", ("left",))
        if right:
            return ("single_hand", ("right",))
    return None


def _merge_seconds(seconds: list[int], *, max_gap: int) -> list[tuple[int, int]]:
    if not seconds:
        return []
    sorted_seconds = sorted(set(seconds))
    ranges: list[tuple[int, int]] = []
    start = end = sorted_seconds[0]
    for second in sorted_seconds[1:]:
        if second <= end + max_gap + 1:
            end = second
            continue
        ranges.append((start, end))
        start = end = second
    ranges.append((start, end))
    return ranges


def _window_from_range(
    idx: int,
    segment: dict[str, Any],
    start: int,
    end: int,
    core_start: int,
    core_end: int,
    frames: list[dict[str, Any]],
    fps: int,
    window_type: str = "two_hand",
    hands_required: list[str] | None = None,
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    for frame in frames:
        for action in _action_names(frame):
            action_counts[action] += 1
        for obj in _object_names(frame):
            object_counts[obj] += 1
    actions = sorted(action_counts, key=lambda name: (-action_counts[name], name))
    objects = sorted(object_counts, key=lambda name: (-object_counts[name], name))
    segment_start = int(segment["start"])
    frame_start = max(0, (int(start) - segment_start) * int(fps))
    frame_end = max(frame_start, (int(end) - segment_start + 1) * int(fps) - 1)
    core_frame_start = max(0, (int(core_start) - segment_start) * int(fps))
    core_frame_end = max(core_frame_start, (int(core_end) - segment_start + 1) * int(fps) - 1)
    return {
        "window_id": f"handwin_{idx:04d}",
        "window_type": window_type,
        "source_segment_id": segment.get("segment_id", ""),
        "source_segment_range": {"start": int(segment["start"]), "end": int(segment["end"])},
        "start": int(start),
        "end": int(end),
        "duration": int(end - start + 1),
        "core_range": {"start": int(core_start), "end": int(core_end)},
        "segment_frame_range": {
            "start": frame_start,
            "end": frame_end,
            "num_frames": frame_end - frame_start + 1,
        },
        "core_segment_frame_range": {
            "start": core_frame_start,
            "end": core_frame_end,
            "num_frames": core_frame_end - core_frame_start + 1,
        },
        "hands_required": hands_required or ["left", "right"],
        "recommended_stage1_hand_policy": "permissive" if window_type == "two_hand" else "conservative",
        "actions": actions,
        "objects": objects,
        "priority": _priority(actions),
        "semantic_label": _semantic_label(actions),
        "source_segment_semantic_label": segment.get("semantic_label", ""),
        "reason": _reason(actions, objects),
    }


def _merge_expanded_windows(windows: list[dict[str, Any]], *, fps: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: (int(item["start"]), int(item["end"]))):
        if not merged or not _can_merge_expanded_window(merged[-1], window):
            merged.append(dict(window))
            continue
        prev = merged[-1]
        prev["end"] = max(int(prev["end"]), int(window["end"]))
        prev["duration"] = int(prev["end"]) - int(prev["start"]) + 1
        prev["core_range"] = {
            "start": min(int(prev["core_range"]["start"]), int(window["core_range"]["start"])),
            "end": max(int(prev["core_range"]["end"]), int(window["core_range"]["end"])),
        }
        prev["segment_frame_range"] = _frame_range_from_seconds(prev, fps=fps)
        core_start = int(prev["core_range"]["start"])
        core_end = int(prev["core_range"]["end"])
        segment_start = int(prev["source_segment_range"]["start"])
        core_frame_start = max(0, (core_start - segment_start) * int(fps))
        core_frame_end = max(core_frame_start, (core_end - segment_start + 1) * int(fps) - 1)
        prev["core_segment_frame_range"] = {
            "start": core_frame_start,
            "end": core_frame_end,
            "num_frames": core_frame_end - core_frame_start + 1,
        }
        prev["actions"] = _merge_unique(prev.get("actions", []), window.get("actions", []))
        prev["objects"] = _merge_unique(prev.get("objects", []), window.get("objects", []))
        prev["priority"] = _priority(prev["actions"])
        prev["reason"] = _reason(prev["actions"], prev["objects"])
    return merged


def _can_merge_expanded_window(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("source_segment_id") == b.get("source_segment_id")
        and a.get("window_type") == b.get("window_type")
        and list(a.get("hands_required", [])) == list(b.get("hands_required", []))
        and int(b["start"]) <= int(a["end"]) + 1
    )


def _filter_windows(windows: list[dict[str, Any]], *, config: HandWindowConfig) -> list[dict[str, Any]]:
    return [window for window in windows if _passes_priority(window, config.min_priority) and _passes_core_duration(window, config.min_core_duration_s)]


def _passes_priority(window: dict[str, Any], min_priority: str | None) -> bool:
    if not min_priority:
        return True
    return _priority_rank(str(window.get("priority", "low"))) >= _priority_rank(min_priority)


def _passes_core_duration(window: dict[str, Any], min_core_duration_s: int) -> bool:
    if min_core_duration_s <= 0:
        return True
    core = window.get("core_range") or {}
    core_duration = int(core.get("end", 0)) - int(core.get("start", 0)) + 1
    return core_duration >= int(min_core_duration_s)


def _priority_rank(priority: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(priority.strip().lower(), 0)


def _frame_range_from_seconds(window: dict[str, Any], *, fps: int) -> dict[str, int]:
    segment_start = int(window["source_segment_range"]["start"])
    frame_start = max(0, (int(window["start"]) - segment_start) * int(fps))
    frame_end = max(frame_start, (int(window["end"]) - segment_start + 1) * int(fps) - 1)
    return {"start": frame_start, "end": frame_end, "num_frames": frame_end - frame_start + 1}


def _merge_unique(first: list[Any], second: list[Any]) -> list[Any]:
    out: list[Any] = []
    for value in [*first, *second]:
        if value not in out:
            out.append(value)
    return out


def _action_names(frame: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for action in frame.get("frame_actions", []) or []:
        if isinstance(action, dict):
            name = str(action.get("action", "")).strip().lower()
        else:
            name = str(action).strip().lower()
        if name:
            out.append(name)
    return out


def _object_names(frame: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for obj in frame.get("interaction_objects", []) or []:
        name = str(obj).strip()
        if name and name not in names:
            names.append(name)
    for action in frame.get("frame_actions", []) or []:
        if isinstance(action, dict):
            name = str(action.get("object_name", "")).strip()
            if name and name != "unknown" and name not in names:
                names.append(name)
    return names


def _priority(actions: list[str]) -> str:
    action_set = set(actions)
    if action_set & HIGH_PRIORITY_ACTIONS:
        return "high"
    if action_set & MEDIUM_PRIORITY_ACTIONS:
        return "medium"
    return "low"


def _semantic_label(actions: list[str]) -> str:
    names = [action for action in actions if action not in CONTEXT_ACTIONS]
    if not names:
        names = list(actions)
    if not names:
        return "hand_window"
    names = sorted(names, key=lambda name: (SEMANTIC_LABEL_ACTION_PRIORITY.get(name, 1000), name))
    return "_".join(name.replace(" ", "_") for name in names[:4])


def _reason(actions: list[str], objects: list[str]) -> str:
    action_text = ", ".join(actions[:4]) if actions else "two-hand visible"
    object_text = ", ".join(objects[:4]) if objects else "visible objects"
    return f"both hands visible with actions [{action_text}] involving [{object_text}]"
