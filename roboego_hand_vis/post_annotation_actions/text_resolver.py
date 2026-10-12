from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from roboego_hand_vis.objects.semantic_qwen import DEFAULT_QWEN_BASE_URL, resolve_qwen_base_url
from roboego_hand_vis.post_annotation_actions.http_client import request_json
from roboego_hand_vis.post_annotation_actions.stage1_actions import ACTION_PATTERNS


ALLOWED_ACTIONS = [action for action, _ in ACTION_PATTERNS]
ALLOWED_HAND_SIDES = {"left", "right", "both", "unknown"}


def find_uncertain_action_frames(payload: dict[str, Any], *, confidence_threshold: float = 0.7) -> list[dict[str, Any]]:
    review_frames = []
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        actions = frame.get("frame_actions", [])
        if any(_action_needs_review(action, confidence_threshold=confidence_threshold) for action in actions):
            review_frames.append(frame)
    return review_frames


def build_text_resolver_payload(frames: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": "resolve_stage1_hand_action_object_bindings",
        "allowed_actions": ALLOWED_ACTIONS,
        "rules": [
            "Use only object names from objects.name.",
            "Do not invent object names.",
            "Keep only hand-related actions.",
            "Use unknown when the binding is ambiguous.",
        ],
        "frames": [_compact_review_frame(frame) for frame in frames],
    }


def build_text_resolver_prompt(review_payload: dict[str, Any]) -> str:
    return (
        "Resolve Stage 1 egocentric hand action-object bindings from text only.\n"
        "Do not invent object names. Use only object names from each frame's objects list.\n"
        "Keep only hand-related actions. If the binding is ambiguous, use object_name=\"unknown\".\n"
        "Return strict JSON with this schema: "
        "{\"frames\":[{\"t\":number,\"frame_actions\":[{\"action\":string,\"object_name\":string,\"hand_side\":\"left|right|both|unknown\",\"confidence\":number}]}]}.\n\n"
        + json.dumps(review_payload, ensure_ascii=False, indent=2)
    )


def build_text_resolver_messages(review_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You resolve hand action-object bindings from text. Return strict JSON only.",
        },
        {"role": "user", "content": build_text_resolver_prompt(review_payload)},
    ]


def call_qwen_text_resolver(
    *,
    review_payload: dict[str, Any],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    base_url = resolve_qwen_base_url(base_url)
    body = {
        "model": model,
        "messages": build_text_resolver_messages(review_payload),
        "temperature": 0,
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    payload = request_json(req, timeout_s=timeout_s, stage="text_resolver", model=model)
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_text_resolver_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    parsed["raw_response"] = content
    return parsed


def write_text_resolver_review_input(*, input_json: Path, output_json: Path) -> dict[str, Any]:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    review_frames = find_uncertain_action_frames(payload)
    review_payload = build_text_resolver_payload(review_frames)
    review_payload["source_action_json"] = str(input_json)
    review_payload["frame_count"] = len(review_payload["frames"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(review_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return review_payload


def parse_text_resolver_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return {"status": "parse_failed", "raw_response": text}
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
        return {"status": "parse_failed", "raw_response": text}
    try:
        frames = [_validate_resolved_frame(frame) for frame in data["frames"]]
    except ValueError as exc:
        return {"status": "validation_failed", "error": str(exc), "raw_response": text}
    return {"status": "ok", "frames": frames}


def merge_text_resolver_results(payload: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    resolved_by_t = {float(frame["t"]): frame for frame in resolved.get("frames", [])}
    merged = dict(payload)
    merged_frames = []
    for frame in payload.get("frames", []):
        if not isinstance(frame, dict):
            continue
        frame_out = dict(frame)
        resolved_frame = resolved_by_t.get(float(frame.get("t", -1)))
        if resolved_frame is not None:
            frame_out["frame_actions"] = resolved_frame["frame_actions"]
            frame_out["text_resolver_applied"] = True
        merged_frames.append(frame_out)
    merged["frames"] = merged_frames
    merged["text_resolver_frame_count"] = len(resolved_by_t)
    return merged


def _action_needs_review(action: Any, *, confidence_threshold: float) -> bool:
    if not isinstance(action, dict):
        return False
    if action.get("source") == "unbound":
        return True
    if action.get("object_name") == "unknown":
        return True
    try:
        return float(action.get("confidence", 1.0)) < confidence_threshold
    except (TypeError, ValueError):
        return True


def _compact_review_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": frame.get("t"),
        "activity": frame.get("activity", ""),
        "hands_visible": frame.get("hands_visible", {}),
        "objects": _compact_objects(frame),
        "current_frame_actions": [dict(action) for action in frame.get("frame_actions", [])],
    }


def _compact_objects(frame: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    for attr in frame.get("interaction_object_attributes", []):
        if isinstance(attr, dict) and attr.get("name"):
            objects.append({"name": attr.get("name"), "hand_side": attr.get("hand_side", "unknown")})
    if objects:
        return objects
    names = []
    for field in ["attention_objects", "task_objects"]:
        for value in frame.get(field, []):
            name = str(value)
            if name and name not in names:
                names.append(name)
    return [{"name": name, "hand_side": "unknown"} for name in names]


def _validate_resolved_frame(frame: Any) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise ValueError("resolved frame must be an object")
    if "t" not in frame or not isinstance(frame.get("frame_actions"), list):
        raise ValueError("resolved frame requires t and frame_actions")
    actions = [_validate_resolved_action(action) for action in frame["frame_actions"]]
    return {"t": frame["t"], "frame_actions": actions}


def _validate_resolved_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("resolved action must be an object")
    out = {
        "action": str(action.get("action", "")),
        "object_name": str(action.get("object_name", "unknown")),
        "hand_side": str(action.get("hand_side", "unknown")),
        "source": "qwen_text_resolver",
        "confidence": _confidence(action.get("confidence", 0.7)),
    }
    if out["action"] not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid action: {out['action']}")
    if out["hand_side"] not in ALLOWED_HAND_SIDES:
        raise ValueError(f"invalid hand_side: {out['hand_side']}")
    return out


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
