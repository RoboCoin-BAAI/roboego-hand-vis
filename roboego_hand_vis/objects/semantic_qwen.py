from __future__ import annotations

import base64
import json
import math
import os
import urllib.request
from pathlib import Path
from typing import Any

from .segmentation import ObjectObservation


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
REQUIRED_FIELDS = ["category", "color", "shape", "material", "visual_description", "confidence"]
TARGET_FIELDS = ["manipulated_object", "object_type", "avoid_parts", "visual_attributes", "confidence"]
CANIDATE_FIELDS = ["is_whole_object", "matches_target", "is_part_or_pattern", "score", "reason"]


def _validated_semantic_object(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict) or any(field not in data for field in REQUIRED_FIELDS):
        return None
    for field in REQUIRED_FIELDS:
        if field == "confidence":
            continue
        if not isinstance(data[field], str):
            return None
    confidence_value = data["confidence"]
    if isinstance(confidence_value, bool):
        return None
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(confidence):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    out = {field: data[field] for field in REQUIRED_FIELDS}
    out["confidence"] = confidence
    out["status"] = "ok"
    return out


def _scan_for_semantic_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        parsed = _validated_semantic_object(data)
        if parsed is not None:
            return parsed
    return None


def parse_semantic_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        parsed = _scan_for_semantic_object(text)
        return parsed or {"status": "parse_failed", "raw_response": text}
    parsed = _validated_semantic_object(data)
    return parsed or {"status": "parse_failed", "raw_response": text}


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    text = _strip_markdown_json_fence(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None
    return data if isinstance(data, dict) else None


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _finite_unit_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        return None
    return out


def parse_qwen_target_response(text: str) -> dict[str, Any]:
    data = _json_object_from_text(text)
    if data is None or any(field not in data for field in TARGET_FIELDS):
        return {"status": "parse_failed", "raw_response": text}
    if not all(isinstance(data[field], str) for field in ["manipulated_object", "object_type"]):
        return {"status": "parse_failed", "raw_response": text}
    avoid_parts = _string_list_field(data["avoid_parts"])
    visual_attributes = _string_list_field(data["visual_attributes"])
    if avoid_parts is None or visual_attributes is None:
        return {"status": "parse_failed", "raw_response": text}
    confidence = _finite_unit_float(data["confidence"])
    if confidence is None:
        return {"status": "parse_failed", "raw_response": text}
    return {
        "status": "ok",
        "manipulated_object": data["manipulated_object"],
        "object_type": data["object_type"],
        "avoid_parts": avoid_parts,
        "visual_attributes": visual_attributes,
        "confidence": confidence,
    }


def _string_list_field(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def parse_qwen_candidate_response(text: str) -> dict[str, Any]:
    data = _json_object_from_text(text)
    if data is None or any(field not in data for field in CANIDATE_FIELDS):
        return {"status": "parse_failed", "raw_response": text}
    for field in ["is_whole_object", "matches_target", "is_part_or_pattern"]:
        if not isinstance(data[field], bool):
            return {"status": "parse_failed", "raw_response": text}
    score = _finite_unit_float(data["score"])
    if score is None or not isinstance(data["reason"], str):
        return {"status": "parse_failed", "raw_response": text}
    return {
        "status": "ok",
        "is_whole_object": data["is_whole_object"],
        "matches_target": data["matches_target"],
        "is_part_or_pattern": data["is_part_or_pattern"],
        "score": score,
        "reason": data["reason"],
    }


def select_key_frames(observations: list[ObjectObservation], max_frames: int) -> list[int]:
    if max_frames <= 0:
        return []
    valid = [obs for obs in observations if obs.bbox_xywh is not None and obs.crop_path is not None]
    if not valid:
        return []
    if max_frames == 1:
        return [valid[0].frame_idx]
    selected = {valid[0].frame_idx}
    largest = max(valid, key=lambda obs: obs.mask_area_px)
    selected.add(largest.frame_idx)
    if max_frames > len(selected):
        step = max(1, (len(valid) - 1) // max(1, max_frames - len(selected)))
        for obs in valid[::step]:
            if obs.frame_idx in selected:
                continue
            selected.add(obs.frame_idx)
            if len(selected) >= max_frames:
                break
    if len(selected) < max_frames:
        for obs in valid:
            if obs.frame_idx in selected:
                continue
            selected.add(obs.frame_idx)
            if len(selected) >= max_frames:
                break
    return sorted(selected)


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_qwen_messages(crop_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Identify the manipulated object in these crops. Return strict JSON with keys: "
                "category, color, shape, material, visual_description, confidence. "
                "Use concise English values and confidence from 0 to 1."
            ),
        }
    ]
    for path in crop_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(path)}})
    return [{"role": "user", "content": content}]


def build_qwen_object_target_messages(frame_path: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Observe this egocentric hand-manipulation frame. Identify the main whole object "
                "the hand is touching or operating. Distinguish the whole object from any local pattern, "
                "printed graphic, logo, face, text, or part on it. Return strict JSON with keys: "
                "manipulated_object, object_type, avoid_parts, visual_attributes, confidence. "
                "Use concise English values. confidence must be 0 to 1."
            ),
        },
        {"type": "image_url", "image_url": {"url": _image_data_url(frame_path)}},
    ]
    return [{"role": "user", "content": content}]


def build_qwen_object_candidate_messages(crop_path: Path, target: dict[str, Any]) -> list[dict[str, Any]]:
    avoid_parts = ", ".join(target.get("avoid_parts", [])) or "none"
    visual_attributes = ", ".join(target.get("visual_attributes", [])) or "unknown"
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Score whether this crop is the whole manipulated target object, not merely a local part "
                "or printed pattern. Target: "
                f"{target.get('manipulated_object', '')}. Object type: {target.get('object_type', '')}. "
                f"Visual attributes: {visual_attributes}. Avoid selecting parts/patterns: {avoid_parts}. "
                "Return strict JSON with keys: is_whole_object, matches_target, is_part_or_pattern, score, reason. "
                "score must be 0 to 1."
            ),
        },
        {"type": "image_url", "image_url": {"url": _image_data_url(crop_path)}},
    ]
    return [{"role": "user", "content": content}]


def _call_qwen_messages(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key_env: str,
    base_url: str,
    timeout_s: float,
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    body = {"model": model, "messages": messages, "temperature": 0}
    req = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def call_qwen_semantics(
    *,
    crop_paths: list[Path],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    content = _call_qwen_messages(
        messages=build_qwen_messages(crop_paths),
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    parsed = parse_semantic_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    return parsed


def call_qwen_object_target(
    *,
    frame_path: Path,
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    content = _call_qwen_messages(
        messages=build_qwen_object_target_messages(frame_path),
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    parsed = parse_qwen_target_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    return parsed


def call_qwen_object_candidate(
    *,
    crop_path: Path,
    target: dict[str, Any],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    content = _call_qwen_messages(
        messages=build_qwen_object_candidate_messages(crop_path, target),
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    parsed = parse_qwen_candidate_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    return parsed
