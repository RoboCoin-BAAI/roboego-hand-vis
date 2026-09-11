"""Nodes for privacy-preserving video processing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .video_blur import run_video_privacy_blur_node  # noqa: F401


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _required_path(config: dict[str, Any], key: str, namespace: str) -> Path:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return Path(value)


def _append_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def run_egoblur_temporal_face_blur_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "egoblur_temporal_face_blur", "status": "skipped", "reason": "disabled"}

    output_json = _required_path(config, "output_json", "egoblur_temporal_face_blur")
    python = _optional_str(config.get("python")) or "python"
    script = _optional_str(config.get("script")) or "scripts/privacy/run_egoblur_temporal_face_blur.py"
    args = [
        python,
        script,
        "--input_video",
        str(_required_path(config, "input_video", "egoblur_temporal_face_blur")),
        "--output_video",
        str(_required_path(config, "output_video", "egoblur_temporal_face_blur")),
        "--output_json",
        str(output_json),
        "--egoblur_root",
        str(_required_path(config, "egoblur_root", "egoblur_temporal_face_blur")),
        "--face_model_path",
        str(_required_path(config, "face_model_path", "egoblur_temporal_face_blur")),
    ]
    _append_arg(args, "--camera_name", config.get("camera_name"))
    _append_arg(args, "--detect_every_n_frames", config.get("detect_every_n_frames"))
    _append_arg(args, "--face_model_score_threshold", config.get("face_model_score_threshold"))
    _append_arg(args, "--nms_iou_threshold", config.get("nms_iou_threshold"))
    _append_arg(args, "--scale_factor_detections", config.get("scale_factor_detections"))
    _append_arg(args, "--max_gap_frames", config.get("max_gap_frames"))
    _append_arg(args, "--smooth_window", config.get("smooth_window"))
    _append_arg(args, "--crf", config.get("crf"))
    _append_arg(args, "--preset", config.get("preset"))

    subprocess.run(args, check=True)
    if output_json.is_file():
        return json.loads(output_json.read_text(encoding="utf-8"))
    return {"stage": "egoblur_temporal_face_blur", "status": "complete", "output_json": str(output_json)}
