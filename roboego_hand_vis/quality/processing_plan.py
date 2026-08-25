from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVENT_ACTIONS = {
    "pick",
    "grasp",
    "release",
    "place",
    "put into",
    "take out",
    "take from",
    "put back",
    "scan",
    "operate",
    "touch",
    "reach",
    "point",
    "press",
    "press button",
    "open",
    "close",
    "rotate",
    "pull",
    "insert",
    "pull out",
    "pour",
    "wipe",
    "sort",
    "stack",
    "load",
    "unload",
    "handover",
    "pass",
}
CONTEXT_ACTIONS = {"hold", "carry", "support"}
SUMMARY_ACTION_PRIORITY = {
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

EVENT_RE = re.compile(
    r"\b(pick|picks|place|places|put|scan|scans|operate|operates|use|uses|touch|touches|reach|reaches|point|points|press|open|close|rotate|pull|insert|pour|wipe|sort|stack|load|unload)\b",
    re.I,
)
WALK_RE = re.compile(r"\b(walk|walks|walking|moves? through|moving through|corridor|aisle)\b", re.I)
HOLD_RE = re.compile(r"\b(hold|holds|holding|held|carry|carrying)\b", re.I)
IDLE_RE = re.compile(r"\b(stand|stands|standing|sit|sits|sitting|wait|waiting|idle|looks? at|checking)\b", re.I)
BAD_VISUAL_RE = re.compile(r"\b(ground|floor|wall|blocked|covered|obstructed|black screen)\b", re.I)


@dataclass(frozen=True)
class ProcessingPlanConfig:
    min_hold_interruption_s: int = 8
    min_walk_interruption_s: int = 5
    min_idle_interruption_s: int = 8
    context_expand_s: int = 3
    min_process_segment_s: int = 30
    merge_process_gap_s: int = 5


def build_processing_plan(
    stage1_payload: dict[str, Any],
    *,
    actions_payload: dict[str, Any] | None = None,
    config: ProcessingPlanConfig = ProcessingPlanConfig(),
) -> dict[str, Any]:
    frames = _merged_frames(stage1_payload, actions_payload)
    classified = [_classify_frame(frame) for frame in frames]
    raw_units = _merge_classified_frames(classified)
    semantic_units = _promote_interruptions(raw_units, config)
    process_segments = _build_process_segments(semantic_units, frames, config)
    drop_segments = _build_drop_segments(semantic_units, process_segments)
    return {
        "source_video": stage1_payload.get("source_video", ""),
        "stage": "stage2_processing_plan",
        "frame_count": len(frames),
        "config": {
            "min_hold_interruption_s": config.min_hold_interruption_s,
            "min_walk_interruption_s": config.min_walk_interruption_s,
            "min_idle_interruption_s": config.min_idle_interruption_s,
            "context_expand_s": config.context_expand_s,
            "min_process_segment_s": config.min_process_segment_s,
            "merge_process_gap_s": config.merge_process_gap_s,
        },
        "semantic_units": semantic_units,
        "segments": process_segments,
        "drop_segments": drop_segments,
    }


def write_processing_plan(
    *,
    stage1_json: Path,
    output_json: Path,
    actions_json: Path | None = None,
    config: ProcessingPlanConfig = ProcessingPlanConfig(),
) -> dict[str, Any]:
    stage1_payload = json.loads(Path(stage1_json).read_text(encoding="utf-8"))
    actions_payload = None
    if actions_json is not None and Path(actions_json).exists():
        actions_payload = _load_actions_payload(Path(actions_json))
    plan = build_processing_plan(stage1_payload, actions_payload=actions_payload, config=config)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def _load_actions_payload(path: Path) -> dict[str, Any]:
    if path.is_dir():
        frames: list[dict[str, Any]] = []
        source_files: list[str] = []
        for chunk_path in sorted(path.glob("*.json")):
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            frames.extend(frame for frame in payload.get("frames", []) if isinstance(frame, dict))
            source_files.append(str(chunk_path))
        return {
            "stage": "stage1_action_resolved_chunks_merged",
            "source_files": source_files,
            "frame_count": len(frames),
            "frames": sorted(frames, key=lambda frame: float(frame.get("t", 0))),
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_segment_summaries(
    plan_payload: dict[str, Any],
    stage1_payload: dict[str, Any],
    *,
    actions_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames = _merged_frames(stage1_payload, actions_payload)
    frame_by_t = {int(float(frame.get("t", 0))): frame for frame in frames}
    summaries = [_summarize_segment(segment, frame_by_t) for segment in plan_payload.get("segments", [])]
    return {
        "source_plan": plan_payload.get("source_plan", ""),
        "stage": "stage2_segment_summary",
        "segment_count": len(summaries),
        "segments": summaries,
    }


def write_segment_summaries(
    *,
    plan_json: Path,
    stage1_json: Path,
    output_json: Path,
    actions_json: Path | None = None,
    selected_segments_dir: Path | None = None,
) -> dict[str, Any]:
    plan_payload = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    stage1_payload = json.loads(Path(stage1_json).read_text(encoding="utf-8"))
    actions_payload = None
    if actions_json is not None and Path(actions_json).exists():
        actions_payload = json.loads(Path(actions_json).read_text(encoding="utf-8"))
    payload = build_segment_summaries(plan_payload, stage1_payload, actions_payload=actions_payload)
    payload["source_plan"] = str(plan_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if selected_segments_dir is not None:
        _write_per_segment_summaries(payload["segments"], Path(selected_segments_dir))
    return payload


def _merged_frames(stage1_payload: dict[str, Any], actions_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    frames = [dict(frame) for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)]
    action_by_t: dict[int, list[dict[str, Any]]] = {}
    if actions_payload:
        for frame in actions_payload.get("frames", []):
            if not isinstance(frame, dict) or "t" not in frame:
                continue
            action_by_t[int(float(frame["t"]))] = [dict(action) for action in frame.get("frame_actions", [])]
    for frame in frames:
        if "t" in frame and int(float(frame["t"])) in action_by_t:
            frame["frame_actions"] = action_by_t[int(float(frame["t"]))]
        else:
            frame.setdefault("frame_actions", [])
    return sorted(frames, key=lambda frame: float(frame.get("t", 0)))


def _summarize_segment(segment: dict[str, Any], frame_by_t: dict[int, dict[str, Any]]) -> dict[str, Any]:
    start = int(segment["start"])
    end = int(segment["end"])
    frames = [frame_by_t[t] for t in range(start, end + 1) if t in frame_by_t]
    action_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    activity_samples: list[str] = []
    privacy_types: set[str] = set()
    privacy_count = 0
    uncertainty_values: list[float] = []
    for frame in frames:
        if len(activity_samples) < 8:
            activity = str(frame.get("activity", "")).strip()
            if activity and activity not in activity_samples:
                activity_samples.append(activity)
        if frame.get("privacy"):
            privacy_count += 1
        for privacy_type in frame.get("privacy_types", []):
            privacy_types.add(str(privacy_type))
        uncertainty_values.append(float(frame.get("uncertainty") or 0.0))
        for action in frame.get("frame_actions", []) or []:
            if not isinstance(action, dict):
                continue
            name = str(action.get("action", "")).strip()
            if name:
                action_counts[name] += 1
            object_name = str(action.get("object_name", "")).strip()
            if object_name and object_name != "unknown":
                object_counts[object_name] += 1
        for object_name in _object_names(frame):
            object_counts[object_name] += 1
    main_actions = _top_names(action_counts, exclude=CONTEXT_ACTIONS)
    if not main_actions:
        main_actions = _top_names(action_counts)
    main_objects = _top_names(object_counts)
    return {
        "segment_id": segment.get("segment_id", ""),
        "start": start,
        "end": end,
        "duration": int(segment.get("duration", end - start + 1)),
        "semantic_label": segment.get("semantic_label", ""),
        "summary": _summary_sentence(activity_samples, main_actions, main_objects),
        "main_actions": main_actions,
        "main_objects": main_objects,
        "privacy": privacy_count > 0,
        "privacy_frame_count": privacy_count,
        "privacy_types": sorted(privacy_types),
        "mean_uncertainty": round(sum(uncertainty_values) / len(uncertainty_values), 4) if uncertainty_values else None,
        "sample_activities": activity_samples,
    }


def _top_names(counts: Counter[str], *, exclude: set[str] | None = None, limit: int = 8) -> list[str]:
    excluded = exclude or set()
    names = [name for name in counts if name not in excluded]
    names.sort(key=lambda name: (-counts[name], SUMMARY_ACTION_PRIORITY.get(name, 1000), name))
    return names[:limit]


def _summary_sentence(activity_samples: list[str], main_actions: list[str], main_objects: list[str]) -> str:
    if activity_samples:
        first = activity_samples[0]
        last = activity_samples[-1]
        if first == last:
            return first
        action_text = ", ".join(main_actions[:5]) if main_actions else "interacts"
        object_text = ", ".join(main_objects[:5]) if main_objects else "visible objects"
        return f"{first}; later, wearer performs {action_text} involving {object_text}."
    action_text = ", ".join(main_actions[:5]) if main_actions else "activity"
    object_text = ", ".join(main_objects[:5]) if main_objects else "objects"
    return f"wearer performs {action_text} involving {object_text}"


def _write_per_segment_summaries(summaries: list[dict[str, Any]], selected_segments_dir: Path) -> None:
    if not selected_segments_dir.exists():
        return
    dirs = [path for path in selected_segments_dir.iterdir() if path.is_dir()]
    for summary in summaries:
        segment_id = str(summary.get("segment_id", ""))
        matches = [path for path in dirs if path.name.startswith(f"{segment_id}_")]
        if not matches:
            continue
        (matches[0] / "segment_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _classify_frame(frame: dict[str, Any]) -> dict[str, Any]:
    activity = str(frame.get("activity", ""))
    scene = str(frame.get("scene", ""))
    actions = _action_names(frame)
    has_event = any(action in EVENT_ACTIONS for action in actions) or EVENT_RE.search(activity) is not None
    has_context_action = any(action in CONTEXT_ACTIONS for action in actions) or HOLD_RE.search(activity) is not None
    text = f"{activity} {scene}"
    if frame.get("image_quality") in {"blocked", "dark", "overexposed"} or BAD_VISUAL_RE.search(text):
        return _classified_frame(frame, "interruption_candidate", "bad_visual")
    if has_event:
        return _classified_frame(frame, "event_core", "manipulation_event")
    if WALK_RE.search(activity):
        return _classified_frame(frame, "interruption_candidate", "walking_only")
    if bool(frame.get("hand_visible")) and has_context_action:
        return _classified_frame(frame, "interruption_candidate", "long_hold_only")
    if not bool(frame.get("hand_visible")) and not bool(frame.get("hand_object_interaction")):
        return _classified_frame(frame, "interruption_candidate", "no_hand_idle")
    if IDLE_RE.search(activity):
        return _classified_frame(frame, "interruption_candidate", "idle_only")
    return _classified_frame(frame, "context", "context_only")


def _classified_frame(frame: dict[str, Any], unit_type: str, label: str) -> dict[str, Any]:
    return {
        "t": int(float(frame.get("t", 0))),
        "unit_type": unit_type,
        "label": label,
        "frame": frame,
    }


def _action_names(frame: dict[str, Any]) -> list[str]:
    return [str(action.get("action", "")).lower() for action in frame.get("frame_actions", []) if isinstance(action, dict)]


def _merge_classified_frames(classified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for item in classified:
        t = item["t"]
        if not units or t != units[-1]["end"] + 1 or item["unit_type"] != units[-1]["unit_type"] or item["label"] != units[-1]["label"]:
            units.append(
                {
                    "start": t,
                    "end": t,
                    "duration": 1,
                    "unit_type": item["unit_type"],
                    "label": item["label"],
                    "route_hint": "process" if item["unit_type"] == "event_core" else "context",
                    "sample_activities": [str(item["frame"].get("activity", ""))],
                    "actions": _action_names(item["frame"]),
                    "objects": _object_names(item["frame"]),
                    "privacy": bool(item["frame"].get("privacy")),
                }
            )
            continue
        unit = units[-1]
        unit["end"] = t
        unit["duration"] = unit["end"] - unit["start"] + 1
        if len(unit["sample_activities"]) < 3:
            unit["sample_activities"].append(str(item["frame"].get("activity", "")))
        unit["actions"] = sorted(set(unit["actions"]) | set(_action_names(item["frame"])))
        unit["objects"] = sorted(set(unit["objects"]) | set(_object_names(item["frame"])))
        unit["privacy"] = bool(unit["privacy"] or item["frame"].get("privacy"))
    return units


def _promote_interruptions(units: list[dict[str, Any]], config: ProcessingPlanConfig) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for unit in units:
        unit = dict(unit)
        if unit["unit_type"] == "interruption_candidate":
            threshold = _interruption_threshold(unit["label"], config)
            if unit["duration"] >= threshold:
                unit["unit_type"] = "interruption"
                unit["route_hint"] = "breakpoint"
            else:
                unit["unit_type"] = "context"
                unit["route_hint"] = "context"
        promoted.append(unit)
    return promoted


def _interruption_threshold(label: str, config: ProcessingPlanConfig) -> int:
    if label == "walking_only":
        return config.min_walk_interruption_s
    if label == "long_hold_only":
        return config.min_hold_interruption_s
    return config.min_idle_interruption_s


def _build_process_segments(
    semantic_units: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    config: ProcessingPlanConfig,
) -> list[dict[str, Any]]:
    if not frames:
        return []
    first_t = int(float(frames[0].get("t", 0)))
    last_t = int(float(frames[-1].get("t", first_t)))
    seeds: list[dict[str, Any]] = []
    for unit in semantic_units:
        if unit["unit_type"] != "event_core":
            continue
        start = max(first_t, int(unit["start"]) - config.context_expand_s)
        end = min(last_t, int(unit["end"]) + config.context_expand_s)
        seeds.append({"start": start, "end": end, "event_start": unit["start"], "event_end": unit["end"]})
    if not seeds:
        return []
    merged = _merge_ranges(seeds, max_gap=config.merge_process_gap_s)
    long_enough = [item for item in merged if item["end"] - item["start"] + 1 >= config.min_process_segment_s]
    return [_segment_from_range(idx, item, semantic_units) for idx, item in enumerate(long_enough)]


def _merge_ranges(ranges: list[dict[str, int]], *, max_gap: int) -> list[dict[str, int]]:
    merged: list[dict[str, int]] = []
    for item in sorted(ranges, key=lambda value: value["start"]):
        if not merged or item["start"] > merged[-1]["end"] + max_gap + 1:
            merged.append(dict(item))
            continue
        merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        merged[-1]["event_start"] = min(merged[-1]["event_start"], item["event_start"])
        merged[-1]["event_end"] = max(merged[-1]["event_end"], item["event_end"])
    return merged


def _segment_from_range(idx: int, item: dict[str, int], semantic_units: list[dict[str, Any]]) -> dict[str, Any]:
    units = [unit for unit in semantic_units if unit["end"] >= item["start"] and unit["start"] <= item["end"]]
    actions = sorted({action for unit in units for action in unit.get("actions", []) if action not in CONTEXT_ACTIONS})
    objects = sorted({obj for unit in units for obj in unit.get("objects", [])})
    return {
        "segment_id": f"seg_{idx:04d}",
        "start": item["start"],
        "end": item["end"],
        "duration": item["end"] - item["start"] + 1,
        "route": "process",
        "semantic_label": _semantic_label(actions),
        "needs_rectify": True,
        "needs_depth": True,
        "needs_hand_estimation": True,
        "needs_object_estimation": bool(objects),
        "actions": actions,
        "objects": objects,
        "source_event_range": {"start": item["event_start"], "end": item["event_end"]},
        "absorbed_interruptions": [
            {"start": unit["start"], "end": unit["end"], "label": unit["label"]}
            for unit in units
            if unit["unit_type"] == "interruption"
        ],
    }


def _build_drop_segments(semantic_units: list[dict[str, Any]], process_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drops: list[dict[str, Any]] = []
    for unit in semantic_units:
        if unit["unit_type"] != "interruption":
            continue
        for start, end in _subtract_process_ranges(unit["start"], unit["end"], process_segments):
            drops.append(_drop_segment(len(drops), start, end, unit["label"]))
    return drops


def _subtract_process_ranges(start: int, end: int, process_segments: list[dict[str, Any]]) -> list[tuple[int, int]]:
    remaining = [(int(start), int(end))]
    for segment in process_segments:
        seg_start = int(segment["start"])
        seg_end = int(segment["end"])
        next_remaining: list[tuple[int, int]] = []
        for cur_start, cur_end in remaining:
            if seg_end < cur_start or seg_start > cur_end:
                next_remaining.append((cur_start, cur_end))
                continue
            if cur_start < seg_start:
                next_remaining.append((cur_start, seg_start - 1))
            if seg_end < cur_end:
                next_remaining.append((seg_end + 1, cur_end))
        remaining = next_remaining
    return [(cur_start, cur_end) for cur_start, cur_end in remaining if cur_start <= cur_end]


def _drop_segment(idx: int, start: int, end: int, label: str) -> dict[str, Any]:
    return {
        "segment_id": f"drop_{idx:04d}",
        "start": start,
        "end": end,
        "duration": end - start + 1,
        "route": "drop",
        "semantic_label": label,
        "reason_tags": ["interruption", label, "no_event_core"],
        "needs_rectify": False,
        "needs_depth": False,
        "needs_hand_estimation": False,
        "needs_object_estimation": False,
    }


def _object_names(frame: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for attr in frame.get("interaction_object_attributes", []):
        if isinstance(attr, dict) and attr.get("name"):
            names.append(str(attr["name"]))
    for name in frame.get("interaction_objects", []):
        if str(name) not in names:
            names.append(str(name))
    return names


def _semantic_label(actions: list[str]) -> str:
    if not actions:
        return "manipulation_event"
    return "_".join(actions[:4])
