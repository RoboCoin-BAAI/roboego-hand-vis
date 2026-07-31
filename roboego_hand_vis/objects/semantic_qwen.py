from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from .segmentation import ObjectObservation


DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
REQUIRED_FIELDS = ["category", "color", "shape", "material", "visual_description", "confidence"]


def parse_semantic_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return {"status": "parse_failed", "raw_response": text}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"status": "parse_failed", "raw_response": text}
    out = {field: data.get(field) for field in REQUIRED_FIELDS}
    out["confidence"] = float(out["confidence"] or 0.0)
    out["status"] = "ok"
    return out


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


def call_qwen_semantics(
    *,
    crop_paths: list[Path],
    model: str,
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    body = {
        "model": model,
        "messages": build_qwen_messages(crop_paths),
        "temperature": 0,
    }
    req = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    parsed = parse_semantic_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    return parsed
