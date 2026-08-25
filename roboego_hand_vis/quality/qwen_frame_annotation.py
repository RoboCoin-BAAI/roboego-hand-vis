from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from roboego_hand_vis.objects.semantic_qwen import DEFAULT_QWEN_BASE_URL


REQUIRED_FRAME_FIELDS = [
    "t",
    "scene",
    "activity",
    "hand_visible",
    "hands_visible",
    "wearer_region",
    "hand_object_interaction",
    "interaction_objects",
    "interaction_object_attributes",
    "attention_objects",
    "task_context",
    "task_objects",
    "privacy",
    "privacy_types",
    "image_quality",
    "uncertainty",
    "notes",
]

IMAGE_QUALITIES = {"clear", "motion_blur", "blocked", "dark", "overexposed", "unknown"}


def read_prompt_markdown(prompt_path: Path) -> tuple[str, str]:
    text = Path(prompt_path).read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)\n```", text, flags=re.S)
    if len(blocks) < 2:
        raise ValueError(f"{prompt_path}: expected at least two ```text blocks")
    return blocks[0], blocks[1]


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_qwen_frame_annotation_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    frame_paths: list[Path],
    timestamps_s: list[float],
) -> list[dict[str, Any]]:
    if len(frame_paths) != len(timestamps_s):
        raise ValueError("frame_paths and timestamps_s must have the same length")
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for frame_path, timestamp_s in zip(frame_paths, timestamps_s):
        timestamp = int(timestamp_s) if float(timestamp_s).is_integer() else round(float(timestamp_s), 3)
        content.append({"type": "text", "text": f"Frame timestamp t={timestamp} seconds"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(frame_path)}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def parse_qwen_frame_annotation_response(text: str) -> dict[str, Any]:
    data = _json_object_from_text(text)
    if data is None or "frames" not in data or not isinstance(data["frames"], list):
        return {"status": "parse_failed", "raw_response": text}
    normalize_frame_annotations(data["frames"])
    migrate_object_attribute_hand_side(data["frames"])
    try:
        validate_frame_annotations(data["frames"])
    except ValueError as exc:
        return {"status": "validation_failed", "error": str(exc), "raw_response": text}
    return {"status": "ok", "frames": data["frames"]}


def normalize_frame_annotations(frames: list[Any]) -> None:
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        if "hands_visible" not in frame:
            frame["hands_visible"] = _hands_visible_from_legacy_side(frame.get("hand_side"))
        elif isinstance(frame["hands_visible"], dict):
            frame["hands_visible"] = {
                "left": frame["hands_visible"].get("left") is True,
                "right": frame["hands_visible"].get("right") is True,
            }
        _normalize_hand_visible_from_hands_visible(frame)
        _normalize_wearer_region(frame)
        _normalize_no_hand_interaction_objects(frame)
        _normalize_pointing_without_target(frame)
        _normalize_false_interaction_with_objects(frame)
        frame.pop("hand_side", None)
        frame.pop("hand_state", None)
        for attr in frame.get("interaction_object_attributes", []):
            if not isinstance(attr, dict):
                continue
            attr.pop("visible_state", None)


def _append_note(frame: dict[str, Any], extra: str) -> None:
    notes = str(frame.get("notes", "")).strip()
    frame["notes"] = f"{notes}; {extra}" if notes else extra


def _normalize_pointing_without_target(frame: dict[str, Any]) -> None:
    if not bool(frame.get("hand_object_interaction")):
        return
    if frame.get("interaction_objects") or frame.get("interaction_object_attributes"):
        return
    activity = str(frame.get("activity", "")).lower()
    if not re.search(r"\b(point|points|pointing|pointed|gesture|gestures|gesturing|gestured)\b", activity):
        return
    frame["hand_object_interaction"] = False
    _append_note(frame, "normalized_pointing_without_target")


def _normalize_hand_visible_from_hands_visible(frame: dict[str, Any]) -> None:
    hands_visible = frame.get("hands_visible")
    if not isinstance(hands_visible, dict):
        return
    any_hand_visible = hands_visible.get("left") is True or hands_visible.get("right") is True
    if any_hand_visible and frame.get("hand_visible") is not True:
        frame["hand_visible"] = True
        _append_note(frame, "normalized_hand_visible_from_hands_visible")
        return
    if any_hand_visible or frame.get("hand_visible") is not True:
        return
    if frame.get("interaction_objects") or frame.get("interaction_object_attributes") or bool(frame.get("hand_object_interaction")):
        return
    frame["hand_visible"] = False
    _append_note(frame, "normalized_hand_visible_from_hands_visible")


def _normalize_wearer_region(frame: dict[str, Any]) -> None:
    if "wearer_region" not in frame:
        frame["wearer_region"] = {
            "visible": False,
            "bbox": None,
            "body_parts": [],
            "confidence": 0.0,
            "notes": "normalized_missing_wearer_region",
        }
        return
    value = frame["wearer_region"]
    if not isinstance(value, dict):
        return
    if value.get("visible") is False and value.get("bbox") == []:
        value["bbox"] = None
    if "body_parts" not in value or value["body_parts"] is None:
        value["body_parts"] = []
    if "notes" not in value or value["notes"] is None:
        value["notes"] = ""


def _normalize_no_hand_interaction_objects(frame: dict[str, Any]) -> None:
    if bool(frame.get("hand_visible")):
        return
    interaction_objects = frame.get("interaction_objects")
    interaction_attributes = frame.get("interaction_object_attributes")
    if not interaction_objects and not interaction_attributes:
        return
    object_names = [str(name) for name in interaction_objects or [] if str(name).strip()]
    for attr in interaction_attributes or []:
        if isinstance(attr, dict) and str(attr.get("name", "")).strip():
            object_names.append(str(attr["name"]))
    for field in ["attention_objects", "task_objects"]:
        values = frame.get(field)
        if not isinstance(values, list):
            values = []
        for name in object_names:
            if name not in values:
                values.append(name)
        frame[field] = values
    frame["hand_object_interaction"] = False
    frame["interaction_objects"] = []
    frame["interaction_object_attributes"] = []
    _append_note(frame, "normalized_no_hand_interaction_objects")


def _normalize_false_interaction_with_objects(frame: dict[str, Any]) -> None:
    if bool(frame.get("hand_object_interaction")):
        return
    if not bool(frame.get("hand_visible")):
        return
    interaction_objects = frame.get("interaction_objects")
    interaction_attributes = frame.get("interaction_object_attributes")
    if not interaction_objects or not interaction_attributes:
        return
    if not isinstance(interaction_objects, list) or not isinstance(interaction_attributes, list):
        return
    frame["hand_object_interaction"] = True
    _append_note(frame, "normalized_interaction_flag_from_objects")


def _hands_visible_from_legacy_side(side: Any) -> dict[str, bool]:
    if side == "left":
        return {"left": True, "right": False}
    if side == "right":
        return {"left": False, "right": True}
    if side == "both":
        return {"left": True, "right": True}
    return {"left": False, "right": False}


def migrate_object_attribute_hand_side(frames: list[Any]) -> None:
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for attr in frame.get("interaction_object_attributes", []):
            if not isinstance(attr, dict) or "hand_side" in attr:
                continue
            attr["hand_side"] = infer_object_hand_side(frame, attr)


def infer_object_hand_side(frame: dict[str, Any], attr: dict[str, Any]) -> str:
    text = " ".join(
        str(value)
        for value in [
            attr.get("visual_description", ""),
            frame.get("activity", ""),
            frame.get("notes", ""),
        ]
    ).lower()
    left_patterns = ["left hand", "left-hand", "wearer's left", "left wrist"]
    right_patterns = ["right hand", "right-hand", "wearer's right", "right wrist"]
    has_left = any(pattern in text for pattern in left_patterns)
    has_right = any(pattern in text for pattern in right_patterns)
    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    frame_side = frame.get("hand_side")
    attrs = frame.get("interaction_object_attributes", [])
    if frame_side in {"left", "right"}:
        return str(frame_side)
    if frame_side == "both" and isinstance(attrs, list) and len(attrs) == 1:
        return "both"
    hands_visible = frame.get("hands_visible")
    if isinstance(hands_visible, dict):
        left_visible = hands_visible.get("left") is True
        right_visible = hands_visible.get("right") is True
        if left_visible and not right_visible:
            return "left"
        if right_visible and not left_visible:
            return "right"
    return "unknown"


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    stripped = _strip_markdown_json_fence(text).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        repaired = _repair_common_qwen_json(stripped)
        if repaired != stripped:
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return data
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(stripped, index)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None
    return data if isinstance(data, dict) else None


def _repair_common_qwen_json(text: str) -> str:
    text = _strip_json_line_comments(text)
    text = _strip_qwen_assignment_noise_lines(text)
    return re.sub(
        r'("hands_visible"\s*:\s*\{)(.*?)(\n\s*\})',
        _repair_hands_visible_json_block,
        text,
        flags=re.S,
    )


def _strip_json_line_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        cut_at = None
        for idx in range(len(line) - 1):
            char = line[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string and char == "/" and line[idx + 1] == "/":
                cut_at = idx
                break
        if cut_at is not None:
            line = line[:cut_at].rstrip()
        lines.append(line)
    return "\n".join(lines)


def _strip_qwen_assignment_noise_lines(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not re.match(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*["\']?[^,\n]*["\']?\s*,?\s*$', line)
    )


def _repair_hands_visible_json_block(match: re.Match[str]) -> str:
    prefix, body, suffix = match.groups()
    body = re.sub(r',?\s*"confidence"\s*:\s*[-+]?\d+(?:\.\d+)?\s*,?', "", body)
    body = re.sub(r'("left"\s*:\s*(?:true|false))\s+("right"\s*:)', r"\1,\n        \2", body)
    body = re.sub(r'("right"\s*:\s*(?:true|false))\s+("left"\s*:)', r"\1,\n        \2", body)
    body = re.sub(r",\s*$", "", body)
    return f"{prefix}{body}{suffix}"


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def validate_frame_annotations(frames: list[Any]) -> None:
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame[{idx}] must be an object")
        missing = [field for field in REQUIRED_FRAME_FIELDS if field not in frame]
        if missing:
            raise ValueError(f"frame[{idx}] missing fields: {missing}")
        _validate_bool(frame, idx, "hand_visible")
        _validate_bool(frame, idx, "hand_object_interaction")
        _validate_bool(frame, idx, "task_context")
        _validate_bool(frame, idx, "privacy")
        _validate_hands_visible(frame, idx)
        _validate_wearer_region(frame, idx)
        _validate_unit_float(frame.get("uncertainty"), f"frame[{idx}].uncertainty")
        if frame["image_quality"] not in IMAGE_QUALITIES:
            raise ValueError(f"frame[{idx}].image_quality invalid: {frame['image_quality']}")
        for field in ["interaction_objects", "interaction_object_attributes", "attention_objects", "task_objects", "privacy_types"]:
            if not isinstance(frame[field], list):
                raise ValueError(f"frame[{idx}].{field} must be a list")
        if not bool(frame["hand_visible"]):
            if frame["hands_visible"]["left"] or frame["hands_visible"]["right"]:
                raise ValueError(f"frame[{idx}] hand_visible=false requires no visible hands")
            if bool(frame["hand_object_interaction"]) or frame["interaction_objects"] or frame["interaction_object_attributes"]:
                raise ValueError(f"frame[{idx}] hand_visible=false cannot have interaction objects")
        if bool(frame["hand_object_interaction"]):
            if not frame["interaction_objects"] or not frame["interaction_object_attributes"]:
                raise ValueError(f"frame[{idx}] hand_object_interaction=true requires interaction_objects and interaction_object_attributes")
        elif frame["interaction_objects"] or frame["interaction_object_attributes"]:
            raise ValueError(f"frame[{idx}] interaction_objects require hand_object_interaction=true")
        for attr_idx, attr in enumerate(frame["interaction_object_attributes"]):
            _validate_object_attributes(attr, f"frame[{idx}].interaction_object_attributes[{attr_idx}]")


def _validate_object_attributes(attr: Any, prefix: str) -> None:
    required = ["name", "hand_side", "color", "shape", "material", "visual_description", "confidence"]
    if not isinstance(attr, dict):
        raise ValueError(f"{prefix} must be an object")
    missing = [field for field in required if field not in attr]
    if missing:
        raise ValueError(f"{prefix} missing fields: {missing}")
    for field in ["name", "hand_side", "color", "shape", "material", "visual_description"]:
        if not isinstance(attr[field], str):
            raise ValueError(f"{prefix}.{field} must be a string")
    if attr["hand_side"] not in {"left", "right", "both", "unknown"}:
        raise ValueError(f"{prefix}.hand_side invalid: {attr['hand_side']}")
    _validate_unit_float(attr["confidence"], f"{prefix}.confidence")


def _validate_hands_visible(frame: dict[str, Any], idx: int) -> None:
    value = frame["hands_visible"]
    if not isinstance(value, dict):
        raise ValueError(f"frame[{idx}].hands_visible must be an object")
    for side in ["left", "right"]:
        if side not in value or not isinstance(value[side], bool):
            raise ValueError(f"frame[{idx}].hands_visible.{side} must be boolean")
    if bool(frame["hand_visible"]) != (value["left"] or value["right"]):
        raise ValueError(f"frame[{idx}].hand_visible must match hands_visible")


def _validate_wearer_region(frame: dict[str, Any], idx: int) -> None:
    value = frame["wearer_region"]
    prefix = f"frame[{idx}].wearer_region"
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} must be an object")
    required = ["visible", "bbox", "body_parts", "confidence", "notes"]
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{prefix} missing fields: {missing}")
    if not isinstance(value["visible"], bool):
        raise ValueError(f"{prefix}.visible must be boolean")
    if not isinstance(value["body_parts"], list) or not all(isinstance(part, str) for part in value["body_parts"]):
        raise ValueError(f"{prefix}.body_parts must be a list of strings")
    if not isinstance(value["notes"], str):
        raise ValueError(f"{prefix}.notes must be a string")
    _validate_unit_float(value["confidence"], f"{prefix}.confidence")
    if value["visible"]:
        _validate_normalized_bbox(value["bbox"], f"{prefix}.bbox")
    elif value["bbox"] is not None:
        raise ValueError(f"{prefix}.bbox must be null when visible=false")


def _validate_normalized_bbox(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field_name} must be [x1, y1, x2, y2]")
    if any(isinstance(item, bool) for item in value):
        raise ValueError(f"{field_name} must contain numbers")
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must contain numbers") from exc
    if not all(math.isfinite(item) for item in [x1, y1, x2, y2]):
        raise ValueError(f"{field_name} must contain finite numbers")
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{field_name} must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")


def _validate_bool(frame: dict[str, Any], idx: int, field: str) -> None:
    if not isinstance(frame[field], bool):
        raise ValueError(f"frame[{idx}].{field} must be boolean")


def _validate_unit_float(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number from 0 to 1")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a number from 0 to 1") from exc
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be a number from 0 to 1")


def call_qwen_frame_annotations(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 120.0,
    max_retries: int = 0,
    retry_sleep_s: float = 1.0,
) -> dict[str, Any]:
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
    max_attempts = max(1, int(max_retries) + 1)
    last_error: dict[str, Any] | None = None
    for attempt_idx in range(max_attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            raw_response = exc.read().decode("utf-8", errors="replace")
            last_error = {
                "status": "http_error",
                "http_status": exc.code,
                "http_reason": exc.reason,
                "provider": "qwen",
                "model": model,
                "raw_response": raw_response,
                "retry_count": attempt_idx,
                "attempt_count": attempt_idx + 1,
            }
            if attempt_idx >= max_attempts - 1 or not is_retriable_qwen_error(last_error):
                return last_error
            if retry_sleep_s > 0:
                time.sleep(float(retry_sleep_s))
        except urllib.error.URLError as exc:
            last_error = {
                "status": "url_error",
                "provider": "qwen",
                "model": model,
                "raw_response": str(exc.reason),
                "retry_count": attempt_idx,
                "attempt_count": attempt_idx + 1,
            }
            if attempt_idx >= max_attempts - 1:
                return last_error
            if retry_sleep_s > 0:
                time.sleep(float(retry_sleep_s))
    else:
        return last_error or {"status": "unknown_error", "provider": "qwen", "model": model}
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_qwen_frame_annotation_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    parsed["raw_response"] = content
    parsed["usage"] = normalize_qwen_usage(payload.get("usage"))
    parsed["request_id"] = str(payload.get("request_id") or payload.get("id") or "")
    parsed["retry_count"] = 0
    parsed["attempt_count"] = 1
    if last_error is not None:
        parsed["retry_count"] = int(last_error.get("attempt_count", 1))
        parsed["attempt_count"] = int(last_error.get("attempt_count", 1)) + 1
    return parsed


def normalize_qwen_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    normalized: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return normalized


def merge_qwen_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        usage = normalize_qwen_usage(item)
        for key, value in usage.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def is_retriable_qwen_error(parsed: dict[str, Any]) -> bool:
    status = str(parsed.get("status", ""))
    raw_response = str(parsed.get("raw_response", "")).lower()
    http_status = int(parsed.get("http_status", 0) or 0)
    if status == "url_error":
        return True
    if http_status in {429, 500, 502, 503, 504}:
        return True
    return "download multimodal file timed out" in raw_response or "timed out" in raw_response or "timeout" in raw_response


def is_data_inspection_failed(parsed: dict[str, Any]) -> bool:
    raw_response = str(parsed.get("raw_response", ""))
    return parsed.get("status") == "http_error" and "data_inspection_failed" in raw_response


def build_data_inspection_placeholder_frames(
    timestamps_s: list[float],
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    request_id = _data_inspection_request_id(parsed)
    for timestamp_s in timestamps_s:
        timestamp: int | float = int(timestamp_s) if float(timestamp_s).is_integer() else round(float(timestamp_s), 3)
        frames.append(
            {
                "t": timestamp,
                "scene": "content blocked by provider safety inspection",
                "activity": "unknown due to provider data inspection failure",
                "hand_visible": False,
                "hands_visible": {"left": False, "right": False},
                "wearer_region": {
                    "visible": False,
                    "bbox": None,
                    "body_parts": [],
                    "confidence": 0.0,
                    "notes": "",
                },
                "hand_object_interaction": False,
                "interaction_objects": [],
                "interaction_object_attributes": [],
                "attention_objects": [],
                "task_context": True,
                "task_objects": [],
                "privacy": True,
                "privacy_types": ["face", "computer_screen", "checkout_screen"],
                "image_quality": "unknown",
                "uncertainty": 1.0,
                "notes": f"data_inspection_failed; request_id={request_id}" if request_id else "data_inspection_failed",
            }
        )
    validate_frame_annotations(frames)
    return frames


def _data_inspection_request_id(parsed: dict[str, Any]) -> str:
    raw_response = str(parsed.get("raw_response", ""))
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return ""
    return str(payload.get("request_id") or payload.get("id") or "")


def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    start_s: float,
    duration_s: float | None,
    fps: float,
    max_width: int,
) -> tuple[list[Path], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / video_fps if video_fps > 0 else 0.0
    end_s = video_duration if duration_s is None else min(video_duration, start_s + duration_s)
    step = 1.0 / max(fps, 1e-6)
    times = list(np.arange(start_s, end_s, step))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    timestamps: list[float] = []
    for time_s in times:
        frame_idx = int(round(time_s * video_fps)) if video_fps > 0 else 0
        frame_idx = max(0, min(frame_count - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize_max_width(frame, max_width=max_width)
        timestamp = float(time_s)
        path = output_dir / f"frame_t{int(round(timestamp)):06d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
        timestamps.append(timestamp)
    cap.release()
    return paths, timestamps


def extract_video_frames_from_windows(
    *,
    video_path: Path,
    output_dir: Path,
    windows: list[dict[str, Any]],
    fps: float,
    max_width: int,
) -> tuple[list[Path], list[float]]:
    """Extract frames from valid windows while keeping absolute video timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / video_fps if video_fps > 0 else 0.0
    step = 1.0 / max(fps, 1e-6)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    timestamps: list[float] = []
    seen_timestamps: set[float] = set()
    for window in sorted(windows, key=lambda item: float(item["start"])):
        start_s = max(0.0, float(window["start"]))
        end_s = min(video_duration, float(window["end"]))
        if end_s <= start_s:
            continue
        for time_s in np.arange(start_s, end_s, step):
            timestamp = round(float(time_s), 6)
            if timestamp in seen_timestamps:
                continue
            seen_timestamps.add(timestamp)
            frame_idx = int(round(timestamp * video_fps)) if video_fps > 0 else 0
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_max_width(frame, max_width=max_width)
            path = output_dir / f"frame_t{int(round(timestamp)):06d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(path)
            timestamps.append(float(timestamp))
    cap.release()
    return paths, timestamps


def _resize_max_width(frame: np.ndarray, *, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def write_qwen_frame_annotation_output(
    *,
    output_json: Path,
    parsed_frames: list[dict[str, Any]],
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source_video": str(video_path),
        "stage": "qwen_ego_frame_annotation",
        "sampling_fps": sampling_fps,
        "prompt": str(prompt_path),
        "model": model,
        "usage": normalize_qwen_usage(usage),
        "frames": parsed_frames,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_batch_annotation(
    *,
    batch_output_dir: Path,
    batch_idx: int,
    frames: list[dict[str, Any]],
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
    usage: dict[str, Any] | None = None,
    retry_count: int = 0,
    attempt_count: int = 1,
) -> Path:
    validate_frame_annotations(frames)
    sorted_frames = sorted(frames, key=lambda frame: float(frame["t"]))
    t_start, t_end = _frame_time_bounds(sorted_frames)
    payload = {
        "source_video": str(video_path),
        "stage": "qwen_ego_frame_annotation_batch",
        "batch_idx": int(batch_idx),
        "time_range": {"start": t_start, "end": t_end},
        "sampling_fps": sampling_fps,
        "prompt": str(prompt_path),
        "model": model,
        "usage": normalize_qwen_usage(usage),
        "retry_count": int(retry_count),
        "attempt_count": int(attempt_count),
        "frames": sorted_frames,
    }
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = batch_output_dir / f"batch_{batch_idx:04d}_t{int(t_start):06d}_{int(t_end):06d}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_chunk_annotations(
    *,
    batch_output_dir: Path,
    chunk_output_dir: Path,
    batches_per_chunk: int,
) -> list[Path]:
    batches_per_chunk = max(1, int(batches_per_chunk))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for batch_path in sorted(Path(batch_output_dir).glob("batch_*.json")):
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        batch_idx = int(payload["batch_idx"])
        chunk_idx = batch_idx // batches_per_chunk
        grouped.setdefault(chunk_idx, []).append(payload)

    outputs: list[Path] = []
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    for chunk_idx in sorted(grouped):
        batch_payloads = sorted(grouped[chunk_idx], key=lambda item: int(item["batch_idx"]))
        batch_indices = [int(item["batch_idx"]) for item in batch_payloads]
        frames = [frame for item in batch_payloads for frame in item["frames"]]
        usage = merge_qwen_usage([item.get("usage", {}) for item in batch_payloads])
        frames = sorted(frames, key=lambda frame: float(frame["t"]))
        t_start, t_end = _frame_time_bounds(frames)
        chunk_payload = {
            "stage": "qwen_ego_frame_annotation_chunk",
            "chunk_idx": chunk_idx,
            "batch_indices": batch_indices,
            "time_range": {"start": t_start, "end": t_end},
            "batch_count": len(batch_indices),
            "frame_count": len(frames),
            "usage": usage,
            "frames": frames,
        }
        output_path = (
            chunk_output_dir
            / f"chunk_{chunk_idx:04d}_batches_{batch_indices[0]:04d}_{batch_indices[-1]:04d}_t{int(t_start):06d}_{int(t_end):06d}.json"
        )
        for stale_path in chunk_output_dir.glob(f"chunk_{chunk_idx:04d}_*.json"):
            if stale_path != output_path:
                stale_path.unlink()
        output_path.write_text(json.dumps(chunk_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs.append(output_path)
    return outputs


def _frame_time_bounds(frames: list[dict[str, Any]]) -> tuple[float, float]:
    if not frames:
        return 0.0, 0.0
    times = [float(frame["t"]) for frame in frames]
    return min(times), max(times)


def write_checkpoint(
    *,
    checkpoint_path: Path,
    frames: list[dict[str, Any]],
    completed_batches: list[int],
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source_video": str(video_path),
        "stage": "qwen_ego_frame_annotation_checkpoint",
        "sampling_fps": sampling_fps,
        "prompt": str(prompt_path),
        "model": model,
        "usage": normalize_qwen_usage(usage),
        "completed_batches": sorted({int(idx) for idx in completed_batches}),
        "frames": sorted(frames, key=lambda frame: float(frame["t"])),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(checkpoint_path)
    return payload


def load_checkpoint_frames(checkpoint_path: Path) -> list[dict[str, Any]]:
    if not Path(checkpoint_path).exists():
        return []
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    normalize_frame_annotations(frames)
    migrate_object_attribute_hand_side(frames)
    validate_frame_annotations(frames)
    return frames


def completed_batch_indices(checkpoint_path: Path) -> set[int]:
    if not Path(checkpoint_path).exists():
        return set()
    payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    return {int(idx) for idx in payload.get("completed_batches", [])}


def recover_checkpoint_from_raw(
    *,
    raw_output_dir: Path,
    checkpoint_path: Path,
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    completed_batches: list[int] = []
    for raw_path in sorted(Path(raw_output_dir).glob("batch_*.raw.txt")):
        match = re.search(r"batch_(\d+)\.raw\.txt$", raw_path.name)
        if match is None:
            continue
        parsed = parse_qwen_frame_annotation_response(raw_path.read_text(encoding="utf-8"))
        if parsed.get("status") != "ok":
            continue
        completed_batches.append(int(match.group(1)))
        frames.extend(parsed["frames"])
    return write_checkpoint(
        checkpoint_path=checkpoint_path,
        frames=_deduplicate_frames(frames),
        completed_batches=completed_batches,
        video_path=video_path,
        prompt_path=prompt_path,
        model=model,
        sampling_fps=sampling_fps,
    )


def recover_batches_from_raw(
    *,
    raw_output_dir: Path,
    batch_output_dir: Path,
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
) -> list[Path]:
    outputs: list[Path] = []
    for raw_path in sorted(Path(raw_output_dir).glob("batch_*.raw.txt")):
        match = re.search(r"batch_(\d+)\.raw\.txt$", raw_path.name)
        if match is None:
            continue
        batch_idx = int(match.group(1))
        parsed = parse_qwen_frame_annotation_response(raw_path.read_text(encoding="utf-8"))
        if parsed.get("status") != "ok":
            continue
        outputs.append(
            write_batch_annotation(
                batch_output_dir=batch_output_dir,
                batch_idx=batch_idx,
                frames=parsed["frames"],
                video_path=video_path,
                prompt_path=prompt_path,
                model=model,
                sampling_fps=sampling_fps,
            )
        )
    return outputs


def _deduplicate_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_t: dict[float, dict[str, Any]] = {}
    for frame in frames:
        by_t[float(frame["t"])] = frame
    return [by_t[key] for key in sorted(by_t)]
