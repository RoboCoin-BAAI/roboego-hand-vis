"""Nodes for selected segment export and dataset preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from roboego_hand_vis.datasets.manifest import ResolveConfig, resolve_dataset_manifest, write_manifest
from roboego_hand_vis.datasets.hand_window_segments import export_hand_window_segments
from roboego_hand_vis.quality.video_segments import (
    DEFAULT_CAMERA_STREAMS,
    export_selected_video_segments,
)


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _append_optional_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def _optional_path(value: Any) -> Path | None:
    value_str = _optional_str(value)
    return Path(value_str) if value_str else None


def run_resolve_dataset_manifest_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "resolve_dataset_manifest", "status": "skipped", "reason": "disabled"}

    dataset = _required_value(config, "dataset", "resolve_dataset_manifest")
    root = Path(_required_value(config, "root", "resolve_dataset_manifest"))
    output_dir = Path(_required_value(config, "output_dir", "resolve_dataset_manifest"))
    resolve_config = ResolveConfig(
        dataset=dataset,
        root=root,
        output_dir=output_dir,
        sequence_name=_optional_str(config.get("sequence_name")) or "default",
        left_video=_optional_path(config.get("left_video")),
        right_video=_optional_path(config.get("right_video")),
        intrinsics_json=_optional_path(config.get("intrinsics_json")),
        tracking_path=_optional_path(config.get("tracking_path")),
        timestamps_left=_optional_path(config.get("timestamps_left")),
        timestamps_right=_optional_path(config.get("timestamps_right")),
        assume_rectified=bool(config.get("assume_rectified", True)),
        assume_undistorted=bool(config.get("assume_undistorted", True)),
    )
    manifest = resolve_dataset_manifest(resolve_config)
    manifest_path, summary_path = write_manifest(manifest, output_dir)
    return {
        "stage": "resolve_dataset_manifest",
        "status": "complete",
        "dataset": dataset,
        "root": str(root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
    }


def run_export_selected_video_segments_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "export_selected_video_segments", "status": "skipped", "reason": "disabled"}

    session = _required_value(config, "session", "export_selected_video_segments")
    processing_plan = _required_value(config, "processing_plan", "export_selected_video_segments")
    output_dir = _required_value(config, "output_dir", "export_selected_video_segments")
    camera_streams = tuple(config.get("camera_streams") or DEFAULT_CAMERA_STREAMS)
    summary = export_selected_video_segments(
        session_dir=Path(session),
        processing_plan_json=Path(processing_plan),
        output_dir=Path(output_dir),
        camera_streams=camera_streams,
        fps=config.get("fps"),
        codec=_optional_str(config.get("codec")) or "h264",
        overwrite=bool(config.get("overwrite", False)),
    )
    return {"stage": "export_selected_video_segments", "status": "complete", **summary}


def run_prepare_selected_segments_node(
    export_selected_video_segments_report: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "prepare_selected_segments", "status": "skipped", "reason": "disabled"}
    export_selected_video_segments_report = export_selected_video_segments_report or {}
    if export_selected_video_segments_report.get("status") != "complete":
        return {
            "stage": "prepare_selected_segments",
            "status": "skipped",
            "reason": "export_selected_video_segments_not_complete",
        }

    session = _optional_str(config.get("session")) or _optional_str(export_selected_video_segments_report.get("source_session"))
    if not session:
        raise ValueError("prepare_selected_segments.session is required when enabled")
    segments_root = _optional_str(config.get("segments_root")) or _optional_str(export_selected_video_segments_report.get("output_dir"))
    if not segments_root:
        raise ValueError("prepare_selected_segments.segments_root is required when enabled")

    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/datasets/prepare_selected_segments.py",
        "--session",
        session,
        "--segments_root",
        segments_root,
    ]
    _append_optional_arg(args, "--roboegopipe_root", config.get("roboegopipe_root"))
    _append_optional_arg(args, "--dataset_adapter", config.get("dataset_adapter"))
    _append_optional_arg(args, "--model", config.get("model"))
    _append_optional_arg(args, "--balance", config.get("balance"))
    _append_optional_arg(args, "--width", config.get("width"))
    _append_optional_arg(args, "--height", config.get("height"))
    segment_ids = config.get("segment_ids") or []
    if segment_ids:
        args.append("--segment_ids")
        args.extend([str(segment_id) for segment_id in segment_ids])
    if bool(config.get("overwrite", False)):
        args.append("--overwrite")
    if bool(config.get("dry_run", False)):
        args.append("--dry_run")

    subprocess.run(args, check=True)
    return {
        "stage": "prepare_selected_segments",
        "status": "complete",
        "session": session,
        "segments_root": segments_root,
    }


def run_export_hand_window_segments_node(
    prepare_selected_segments_report: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "export_hand_window_segments", "status": "skipped", "reason": "disabled"}
    prepare_selected_segments_report = prepare_selected_segments_report or {}
    if prepare_selected_segments_report.get("status") != "complete":
        return {
            "stage": "export_hand_window_segments",
            "status": "skipped",
            "reason": "prepare_selected_segments_not_complete",
        }

    segments_root = _optional_str(config.get("segments_root")) or _optional_str(
        prepare_selected_segments_report.get("segments_root")
    )
    if not segments_root:
        raise ValueError("export_hand_window_segments.segments_root is required when enabled")
    hand_windows_json = _required_value(config, "hand_windows_json", "export_hand_window_segments")
    summary = export_hand_window_segments(
        segments_root=Path(segments_root),
        hand_windows_json=Path(hand_windows_json),
        overwrite=bool(config.get("overwrite", False)),
        codec=_optional_str(config.get("codec")) or "h264",
    )
    return {"stage": "export_hand_window_segments", "status": "complete", **summary}


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value
