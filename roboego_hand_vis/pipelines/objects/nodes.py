"""Nodes for object segmentation pipelines."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _append_optional_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value


def _configured_hand_window_dirs(config: dict[str, Any]) -> list[str]:
    dirs: list[str] = []
    hand_window_dir = _optional_str(config.get("hand_window_dir"))
    if hand_window_dir:
        dirs.append(hand_window_dir)
    for value in config.get("hand_window_dirs") or []:
        value_str = _optional_str(value)
        if value_str:
            dirs.append(value_str)
    return dirs


def run_sam3_hand_object_segmentation_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "sam3_hand_object_segmentation", "status": "skipped", "reason": "disabled"}

    hand_window_dirs = _configured_hand_window_dirs(config)
    if not hand_window_dirs:
        raise ValueError("sam3_hand_object_segmentation.hand_window_dir is required when enabled")

    stage1_actions = _required_value(config, "stage1_actions", "sam3_hand_object_segmentation")
    python = _optional_str(config.get("python")) or "python"
    script = _optional_str(config.get("script")) or "scripts/object/run_sam3_hand_clip_objects.py"

    processed: list[str] = []
    for hand_window_dir in hand_window_dirs:
        args = [
            python,
            script,
            "--hand_window_dir",
            hand_window_dir,
            "--stage1_actions",
            stage1_actions,
        ]
        _append_optional_arg(args, "--windows_json", config.get("windows_json"))
        _append_optional_arg(args, "--output_dir", config.get("output_dir"))
        _append_optional_arg(args, "--object_name", config.get("object_name"))
        _append_optional_arg(args, "--object_shape", config.get("object_shape"))
        _append_optional_arg(args, "--object_color", config.get("object_color"))
        _append_optional_arg(args, "--object_id", config.get("object_id"))
        _append_optional_arg(args, "--sam3_script", config.get("sam3_script"))
        _append_optional_arg(args, "--python", config.get("sam3_python"))
        _append_optional_arg(args, "--fps", config.get("fps"))
        _append_optional_arg(args, "--prompt_frame_idx", config.get("prompt_frame_idx"))
        _append_optional_arg(args, "--checkpoint", config.get("checkpoint"))
        if bool(config.get("auto_from_stage1", True)):
            args.append("--auto_from_stage1")
        if bool(config.get("no_offload_video_to_cpu", False)):
            args.append("--no_offload_video_to_cpu")
        if bool(config.get("dry_run", False)):
            args.append("--dry_run")

        subprocess.run(args, check=True)
        processed.append(str(Path(hand_window_dir)))

    return {
        "stage": "sam3_hand_object_segmentation",
        "status": "complete",
        "processed_count": len(processed),
        "hand_window_dirs": processed,
        "stage1_actions": stage1_actions,
    }
