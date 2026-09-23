from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from roboego_hand_vis.objects.semantic_qwen import DEFAULT_QWEN_BASE_URL, resolve_qwen_base_url


PHASE_ACTIONS = {
    "approach",
    "align",
    "grasp",
    "secure",
    "lift",
    "carry",
    "position",
    "lower",
    "release",
    "press_motion",
    "push_motion",
    "pull_motion",
    "rotate_motion",
    "tilt_motion",
    "pour_motion",
    "fold_motion",
    "slide_motion",
    "insert_motion",
    "shake_motion",
    "strike_motion",
    "throw_motion",
    "scrub_motion",
    "navigate_motion",
    "retreat",
    "idle",
}

ACTION_TO_PHASE = {
    "reach": "approach",
    "point": "align",
    "align": "align",
    "touch": "align",
    "pick": "grasp",
    "grasp": "grasp",
    "pinch": "grasp",
    "clip": "grasp",
    "hold": "secure",
    "support": "secure",
    "lift": "lift",
    "carry": "carry",
    "move": "carry",
    "pass": "carry",
    "handover": "carry",
    "place": "release",
    "put into": "release",
    "put back": "release",
    "release": "release",
    "lower": "lower",
    "press": "press_motion",
    "press button": "press_motion",
    "scan": "press_motion",
    "push": "push_motion",
    "push together": "push_motion",
    "pull": "pull_motion",
    "pull out": "pull_motion",
    "pull apart": "pull_motion",
    "take out": "pull_motion",
    "take from": "pull_motion",
    "rotate": "rotate_motion",
    "open": "rotate_motion",
    "close": "rotate_motion",
    "operate": "rotate_motion",
    "flip": "rotate_motion",
    "uncap": "rotate_motion",
    "pour": "pour_motion",
    "fold": "fold_motion",
    "unfold": "fold_motion",
    "insert": "insert_motion",
    "stir": "shake_motion",
    "cut": "strike_motion",
    "wipe": "scrub_motion",
    "measure": "align",
    "sort": "position",
    "stack": "position",
    "load": "position",
    "unload": "position",
    "gesture": "idle",
}

GENERIC_OBJECT_WORDS = {
    "a",
    "an",
    "the",
    "object",
    "item",
    "thing",
    "something",
    "product",
    "goods",
}


def standardize_stage1_actions(
    payload: dict[str, Any],
    *,
    frames_dir: Path | None = None,
    use_vl_object_resolver: bool = False,
    vl_object_resolver: Any | None = None,
    vl_model: str = "qwen-vl-plus",
    api_key_env: str = "DASHSCOPE_API_KEY",
    timeout_s: float = 120.0,
    max_vl_frames: int = 30,
) -> dict[str, Any]:
    out = dict(payload)
    out["stage"] = "robolabel_stage1_actions"
    frames = [_standardize_frame(frame) for frame in payload.get("frames", []) if isinstance(frame, dict)]
    out["frames"] = _assign_fallback_object_tracks(frames)
    out["robolabel_standard"] = {
        "phase_actions": sorted(PHASE_ACTIONS),
        "object_policy": "free_text_with_qwen_vl_adjacent_identity_check",
    }
    if use_vl_object_resolver and frames_dir is not None:
        review_payload = build_vl_object_resolver_payload(out, frames_dir=frames_dir, max_frames=max_vl_frames)
        if review_payload["frames"]:
            resolver = vl_object_resolver or call_qwen_vl_object_resolver
            resolved = resolver(
                review_payload=review_payload,
                model=vl_model,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
            )
            out = merge_vl_object_resolver_results(out, resolved)
            out["vl_object_resolver_status"] = resolved.get("status", "unknown")
            out["vl_object_resolver_model"] = vl_model
    return out


def build_vl_object_resolver_payload(
    payload: dict[str, Any],
    *,
    frames_dir: Path,
    max_frames: int = 30,
) -> dict[str, Any]:
    frames = []
    for frame in sorted(payload.get("frames", []), key=lambda item: float(item.get("t", 0))):
        if not isinstance(frame, dict) or not frame.get("frame_actions"):
            continue
        image_path = _frame_image_path(frames_dir, frame.get("t"))
        if image_path is None:
            continue
        frames.append(
            {
                "t": frame.get("t"),
                "image_path": str(image_path),
                "activity": frame.get("activity", ""),
                "frame_actions": [
                    {
                        "action_index": idx,
                        "phase_action": action.get("phase_action", action.get("action", "")),
                        "object_name": action.get("object_name", ""),
                        "object_description": action.get("object_description", ""),
                        "hand_side": action.get("hand_side", "unknown"),
                    }
                    for idx, action in enumerate(frame.get("frame_actions", []))
                    if isinstance(action, dict)
                ],
            }
        )
        if len(frames) >= max_frames:
            break
    return {
        "task": "resolve_adjacent_second_object_identity",
        "rules": [
            "Compare each timestamp with the previous and next timestamp when visible.",
            "Decide whether object references in adjacent seconds are the same physical object.",
            "Keep object names as free text, but make them concise and visually grounded.",
            "Do not force objects into a fixed vocabulary.",
        ],
        "frames": frames,
    }


def call_qwen_vl_object_resolver(
    *,
    review_payload: dict[str, Any],
    model: str = "qwen-vl-plus",
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    body = {
        "model": model,
        "messages": build_vl_object_resolver_messages(review_payload),
        "temperature": 0,
    }
    req = urllib.request.Request(
        resolve_qwen_base_url(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        response_payload = json.loads(resp.read().decode("utf-8"))
    content = response_payload["choices"][0]["message"]["content"]
    parsed = parse_vl_object_resolver_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    parsed["raw_response"] = content
    return parsed


def build_vl_object_resolver_messages(review_payload: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You normalize egocentric hand-object annotations. Return strict JSON only with schema: "
                "{\"frames\":[{\"t\":number,\"frame_actions\":[{\"action_index\":number,"
                "\"canonical_object_name\":string,\"object_description\":string,"
                "\"object_track_id\":string,\"same_object_as_previous_second\":boolean,"
                "\"confidence\":number}]}]}. "
                "Object names are free text. Use adjacent seconds and images to decide physical identity.\n\n"
                + json.dumps(_payload_without_image_paths(review_payload), ensure_ascii=False, indent=2)
            ),
        }
    ]
    for frame in review_payload.get("frames", []):
        path = Path(frame["image_path"])
        content.append({"type": "text", "text": f"timestamp t={frame.get('t')}"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    return [{"role": "user", "content": content}]


def parse_vl_object_resolver_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return {"status": "parse_failed", "raw_response": text}
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        return {"status": "parse_failed", "raw_response": text}
    return {"status": "ok", "frames": data["frames"]}


def merge_vl_object_resolver_results(payload: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    if resolved.get("status") != "ok":
        return payload
    out = dict(payload)
    frames = []
    resolved_by_t = {float(frame.get("t", -1)): frame for frame in resolved.get("frames", []) if isinstance(frame, dict)}
    for frame in payload.get("frames", []):
        frame_out = dict(frame)
        actions = [dict(action) for action in frame_out.get("frame_actions", []) if isinstance(action, dict)]
        resolved_frame = resolved_by_t.get(float(frame.get("t", -1)))
        if resolved_frame is not None:
            for action_update in resolved_frame.get("frame_actions", []):
                if not isinstance(action_update, dict):
                    continue
                idx = int(action_update.get("action_index", -1))
                if idx < 0 or idx >= len(actions):
                    continue
                _apply_vl_object_update(actions[idx], action_update)
        frame_out["frame_actions"] = actions
        frames.append(frame_out)
    out["frames"] = frames
    return out


def _standardize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    out = dict(frame)
    attrs = frame.get("interaction_object_attributes", [])
    out["frame_actions"] = [
        _standardize_action(action, attrs=attrs) for action in frame.get("frame_actions", []) if isinstance(action, dict)
    ]
    if out["frame_actions"]:
        out["coordination_mode"] = _coordination_mode(out["frame_actions"])
    return out


def _standardize_action(action: dict[str, Any], *, attrs: list[Any]) -> dict[str, Any]:
    original_action = str(action.get("action", "")).strip().lower()
    phase_action = ACTION_TO_PHASE.get(original_action, "idle")
    object_name = _clean_object_name(action.get("object_name", "unknown"))
    out = dict(action)
    out["original_action"] = original_action
    out["action"] = phase_action
    out["phase_action"] = phase_action
    out["skill"] = phase_action
    out["subject"] = _subject_from_hand_side(out.get("hand_side", "unknown"))
    out["object_name"] = object_name
    out["manipulated_object"] = object_name
    out["object_description"] = _object_description(object_name, attrs)
    out["robolabel_source"] = "robolabel_standardizer"
    return out


def _assign_fallback_object_tracks(frames: list[dict[str, Any]], *, max_gap_s: float = 1.5) -> list[dict[str, Any]]:
    last_by_side: dict[str, tuple[float, str, str]] = {}
    next_track_id = 1
    for frame in frames:
        t = float(frame.get("t", 0))
        for action in frame.get("frame_actions", []):
            side = str(action.get("hand_side", "unknown"))
            key = _object_identity_key(str(action.get("object_name", "")))
            last = last_by_side.get(side)
            same_previous = bool(last and t - last[0] <= max_gap_s and _same_object_key(last[1], key))
            if same_previous:
                track_id = last[2]
            else:
                track_id = f"obj_{next_track_id:04d}"
                next_track_id += 1
            action["object_track_id"] = track_id
            action["same_object_as_previous_second"] = same_previous
            last_by_side[side] = (t, key, track_id)
    return frames


def _apply_vl_object_update(action: dict[str, Any], update: dict[str, Any]) -> None:
    name = _clean_object_name(update.get("canonical_object_name", action.get("object_name", "")))
    description = str(update.get("object_description") or action.get("object_description") or name).strip()
    action["object_name"] = name
    action["manipulated_object"] = name
    action["object_description"] = description
    action["object_track_id"] = str(update.get("object_track_id") or action.get("object_track_id", ""))
    action["same_object_as_previous_second"] = bool(
        update.get("same_object_as_previous_second", action.get("same_object_as_previous_second", False))
    )
    action["object_resolver_confidence"] = _confidence(update.get("confidence", 0.7))
    action["object_resolver_source"] = "qwen_vl_plus"


def _coordination_mode(actions: list[dict[str, Any]]) -> str:
    if len(actions) <= 1:
        return "single_hand"
    phase_actions = {action.get("phase_action") for action in actions}
    objects = {action.get("object_name") for action in actions}
    if len(phase_actions) == 1 and len(objects) == 1:
        return "both_same_skill_same_object"
    if len(phase_actions) == 1:
        return "both_same_skill_different_objects"
    return "parallel_different_skills"


def _subject_from_hand_side(hand_side: Any) -> str:
    return {
        "left": "left_effector",
        "right": "right_effector",
        "both": "both_effectors",
    }.get(str(hand_side), "unknown")


def _object_description(object_name: str, attrs: list[Any]) -> str:
    parts = [object_name]
    attr = _matching_attr(object_name, attrs)
    if attr is not None:
        for key in ["color", "shape", "material", "visual_description"]:
            value = str(attr.get(key, "")).strip()
            if value:
                parts.append(f"{key}={value}")
    return "; ".join(parts)


def _matching_attr(object_name: str, attrs: list[Any]) -> dict[str, Any] | None:
    object_key = _object_identity_key(object_name)
    for attr in attrs:
        if isinstance(attr, dict) and _same_object_key(_object_identity_key(str(attr.get("name", ""))), object_key):
            return attr
    return None


def _same_object_key(first: str, second: str) -> bool:
    if not first or not second:
        return False
    first_tokens = set(first.split())
    second_tokens = set(second.split())
    if first == second:
        return True
    return bool(first_tokens & second_tokens)


def _object_identity_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
    return " ".join(token for token in tokens if token not in GENERIC_OBJECT_WORDS)


def _clean_object_name(value: Any) -> str:
    name = str(value or "unknown").strip()
    return name if name else "unknown"


def _frame_image_path(frames_dir: Path, timestamp: Any) -> Path | None:
    try:
        rounded = int(round(float(timestamp)))
    except (TypeError, ValueError):
        return None
    path = frames_dir / f"frame_t{rounded:06d}.jpg"
    return path if path.exists() else None


def _payload_without_image_paths(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["frames"] = [{key: value for key, value in frame.items() if key != "image_path"} for frame in payload.get("frames", [])]
    return out


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, confidence))


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped
