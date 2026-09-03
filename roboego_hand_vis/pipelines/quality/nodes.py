"""Nodes for quality-control pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from roboego_hand_vis.depth.numeric_quality import DepthNumericQualityConfig, run_depth_numeric_quality_check
from roboego_hand_vis.depth.quality import DepthQualityConfig, run_depth_quality_check
from roboego_hand_vis.quality.pre_semantic_gate import (
    PreSemanticGateConfig,
    run_pre_semantic_quality_gate,
)
from roboego_hand_vis.quality.raw_headband_integrity import (
    DEFAULT_CAMERA_STREAMS,
    DEFAULT_VIDEO_EXT,
    export_quality_report,
)
from roboego_hand_vis.quality.raw_headband_rules import evaluate_quality_report
from roboego_hand_vis.quality.stereo_video import (
    SegmentQualityConfig,
    SummaryQualityConfig,
    run_segment_quality_check,
    run_summary_quality_check,
)


LOGGER = logging.getLogger(__name__)


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _camera_streams_from_config(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return DEFAULT_CAMERA_STREAMS
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split("|") if part.strip())
    return tuple(str(part) for part in value)


def run_raw_headband_integrity_check(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "raw_headband_integrity_check", "status": "skipped", "reason": "disabled"}

    session_dir = _optional_str(config.get("session_dir"))
    if not session_dir:
        raise ValueError("raw_headband_quality.session_dir is required when enabled")
    output_root = _optional_str(config.get("output_root")) or session_dir
    camera_streams = _camera_streams_from_config(config.get("camera_streams"))
    report_path = export_quality_report(
        session_dir=Path(session_dir),
        output_root=Path(output_root),
        camera_streams=camera_streams,
        start=int(config.get("start", 0)),
        num_frames=config.get("num_frames"),
        timestamp_tolerance_s=config.get("timestamp_tolerance_s"),
        video_ext=_optional_str(config.get("video_ext")) or DEFAULT_VIDEO_EXT,
        overwrite=bool(config.get("overwrite", False)),
        environment_name=_optional_str(config.get("environment_name")) or "ego",
    )
    return {"stage": "raw_headband_integrity_check", "status": "complete", "report_path": str(report_path)}


def run_raw_headband_quality_gate(
    integrity_result: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    integrity_result = integrity_result or {}
    config = config or {}
    if integrity_result.get("status") != "complete":
        return {
            "stage": "raw_headband_quality_gate",
            "status": "skipped",
            "reason": "raw_headband_integrity_check_skipped",
        }
    report_path = Path(str(integrity_result["report_path"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = evaluate_quality_report(report, config.get("rules") or {})
    result["report_path"] = str(report_path)
    report["raw_headband_quality_gate"] = result
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    message = (
        f"Raw headband quality gate: status={result['status']} passed={result['passed']} "
        f"failed={len(result['failed_checks'])} warnings={len(result['warnings'])} "
        f"failed_checks={result['failed_checks']} warnings_list={result['warnings']}"
    )
    LOGGER.info(message)
    print(_format_raw_quality_gate_message(result))
    return result


def _format_raw_quality_gate_message(result: dict[str, Any]) -> str:
    if result.get("passed") is False:
        lines = [
            "",
            "========== RAW QUALITY GATE FAILED ==========",
            f"passed=False status={result.get('status', 'failed')}",
            f"failed_checks={result.get('failed_checks', [])}",
            f"warnings={result.get('warnings', [])}",
        ]
        for check in result.get("checks", []):
            if not isinstance(check, dict) or check.get("passed") is not False:
                continue
            detail = f"- {check.get('name')}"
            if check.get("stream"):
                detail += f":{check.get('stream')}"
            if check.get("message"):
                detail += f" | {check.get('message')}"
            if "value" in check:
                detail += f" | value={check.get('value')}"
            if "threshold" in check:
                detail += f" | threshold={check.get('threshold')}"
            lines.append(detail)
        if result.get("report_path"):
            lines.append(f"report={result['report_path']}")
        lines.append("=============================================")
        return "\n".join(lines)
    return (
        f"Raw headband quality gate: status={result['status']} passed={result['passed']} "
        f"failed={len(result['failed_checks'])} warnings={len(result['warnings'])} "
        f"failed_checks={result['failed_checks']} warnings_list={result['warnings']}"
    )


def run_pre_semantic_quality_gate_node(
    raw_gate_result: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "video_simple_quality_gate", "status": "skipped", "reason": "disabled"}, []
    if raw_gate_result and raw_gate_result.get("passed") is False:
        return {
            "stage": "video_simple_quality_gate",
            "status": "skipped",
            "reason": "raw_headband_quality_gate_not_passed",
        }, []
    video = _optional_str(config.get("video"))
    if not video:
        raise ValueError("pre_semantic_quality.video is required when enabled")
    video_path = Path(video)
    output_json = Path(_optional_str(config.get("output_json")) or video_path.parent.parent / "annotation" / f"video_simple_quality_{video_path.stem}.json")
    gate_config = PreSemanticGateConfig(
        sample_every_s=float(config.get("sample_every_s", 1.0)),
        dark_threshold=int(config.get("dark_threshold", 10)),
        bright_threshold=int(config.get("bright_threshold", 245)),
        max_black_ratio=float(config.get("max_black_ratio", 0.98)),
        max_overexposed_ratio=float(config.get("max_overexposed_ratio", 0.98)),
        min_texture_std=float(config.get("min_texture_std", 3.0)),
        freeze_absdiff_threshold=float(config.get("freeze_absdiff_threshold", 1.5)),
        low_motion_absdiff_threshold=float(config.get("low_motion_absdiff_threshold", 4.0)),
        min_freeze_duration_s=float(config.get("min_freeze_duration_s", 5.0)),
        min_bad_visual_duration_s=float(config.get("min_bad_visual_duration_s", 3.0)),
    )
    report = run_pre_semantic_quality_gate(video_path=video_path, output_json=output_json, config=gate_config)
    windows = []
    for interval in report.get("intervals", []):
        if interval.get("route_hint") not in {"pass_to_qwen", "pass_to_qwen_with_aux"}:
            continue
        windows.append(
            {
                "source_video": str(video_path),
                "start": float(interval["start"]),
                "end": float(interval["end"]),
                "label": interval.get("label", ""),
                "route_hint": interval.get("route_hint", ""),
                "confidence": interval.get("confidence"),
            }
        )
    report_out = dict(report)
    report_out["status"] = "complete"
    report_out["output_json"] = str(output_json)
    return report_out, windows


def run_stereo_summary_quality_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "stereo_summary_quality", "status": "skipped", "reason": "disabled"}
    manifest = Path(_required_value(config, "manifest", "stereo_summary_quality"))
    output_dir = _quality_output_dir(config, manifest, "stereo_summary")
    report = run_summary_quality_check(
        manifest_path=manifest,
        output_dir=output_dir,
        config=SummaryQualityConfig(
            sample_count=int(config.get("sample_count", 12)),
            dark_threshold=int(config.get("dark_threshold", 10)),
            bright_threshold=int(config.get("bright_threshold", 245)),
            orb_features=int(config.get("orb_features", 1500)),
            max_brightness_gap=float(config.get("max_brightness_gap", 35.0)),
            min_sharpness=float(config.get("min_sharpness", 50.0)),
            max_dark_ratio=float(config.get("max_dark_ratio", 0.25)),
            max_bright_ratio=float(config.get("max_bright_ratio", 0.25)),
            min_orb_matches=int(config.get("min_orb_matches", 30)),
            max_median_abs_dy=float(config.get("max_median_abs_dy", 2.0)),
            max_p90_abs_dy=float(config.get("max_p90_abs_dy", 5.0)),
        ),
    )
    return {
        "stage": "stereo_summary_quality",
        "status": "complete",
        "passed": bool(report.get("summary", {}).get("passed", False)),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "stereo_quality_report.json"),
        "summary": report.get("summary", {}),
        "outputs": report.get("outputs", {}),
    }


def run_stereo_segment_quality_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "stereo_segment_quality", "status": "skipped", "reason": "disabled"}
    manifest = Path(_required_value(config, "manifest", "stereo_segment_quality"))
    output_dir = _quality_output_dir(config, manifest, "stereo_segments")
    report = run_segment_quality_check(
        manifest_path=manifest,
        output_dir=output_dir,
        config=SegmentQualityConfig(
            sample_every_s=float(config.get("sample_every_s", 1.0)),
            orb_every_s=float(config.get("orb_every_s", 5.0)),
            dark_threshold=int(config.get("dark_threshold", 10)),
            bright_threshold=int(config.get("bright_threshold", 245)),
            orb_features=int(config.get("orb_features", 1500)),
            max_bad_contact_frames=int(config.get("max_bad_contact_frames", 40)),
            max_brightness_gap=float(config.get("max_brightness_gap", 35.0)),
            min_sharpness=float(config.get("min_sharpness", 50.0)),
            max_dark_ratio=float(config.get("max_dark_ratio", 0.25)),
            max_bright_ratio=float(config.get("max_bright_ratio", 0.25)),
            min_orb_matches=int(config.get("min_orb_matches", 30)),
            max_median_abs_dy=float(config.get("max_median_abs_dy", 2.0)),
            max_p90_abs_dy=float(config.get("max_p90_abs_dy", 5.0)),
        ),
    )
    return {
        "stage": "stereo_segment_quality",
        "status": "complete",
        "passed": bool(report.get("passed", False)),
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "bad_segments.json"),
        "bad_frame_count": int(report.get("bad_frame_count", 0)),
        "bad_segment_count": int(report.get("bad_segment_count", 0)),
        "outputs": report.get("outputs", {}),
    }


def run_depth_quality_node(
    stereo_segment_quality_result: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "depth_quality", "status": "skipped", "reason": "disabled"}
    manifest = Path(_required_value(config, "manifest", "depth_quality"))
    depth_dir = Path(_required_value(config, "depth_dir", "depth_quality"))
    output_dir = _quality_output_dir(config, manifest, "depth")
    stereo_report = _optional_str(config.get("stereo_quality_report"))
    if not stereo_report and stereo_segment_quality_result and stereo_segment_quality_result.get("status") == "complete":
        stereo_report = _optional_str(stereo_segment_quality_result.get("report_path"))
    report = run_depth_quality_check(
        manifest_path=manifest,
        depth_dir=depth_dir,
        output_dir=output_dir,
        stereo_quality_report=Path(stereo_report) if stereo_report else None,
        config=DepthQualityConfig(
            start=int(config.get("start", 0)),
            num_frames=int(config.get("num_frames", 300)),
            min_depth_m=float(config.get("min_depth_m", 0.15)),
            max_depth_m=float(config.get("max_depth_m", 4.0)),
            point_stride=int(config.get("point_stride", 4)),
            max_preview_points=int(config.get("max_preview_points", 90000)),
            overlay_alpha=float(config.get("overlay_alpha", 0.45)),
            warp_sample_stride=int(config.get("warp_sample_stride", 2)),
            edge_percentile=float(config.get("edge_percentile", 85.0)),
        ),
    )
    return {
        "stage": "depth_quality",
        "status": "complete",
        "manifest": str(manifest),
        "depth_dir": str(depth_dir),
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "depth_quality_report.json"),
        "summary": report.get("summary", {}),
        "outputs": report.get("outputs", {}),
    }


def run_depth_quality_node_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return run_depth_quality_node(None, config)


def run_depth_numeric_quality_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "depth_numeric_quality", "status": "skipped", "reason": "disabled"}
    depth_dir = Path(_required_value(config, "depth_dir", "depth_numeric_quality"))
    output_dir = _optional_str(config.get("output_dir"))
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = depth_dir.parent.parent / "quality" / "depth_numeric"
    report = run_depth_numeric_quality_check(
        depth_dir=depth_dir,
        output_dir=output_path,
        config=DepthNumericQualityConfig(
            start=int(config.get("start", 0)),
            num_frames=int(config.get("num_frames", 300)),
            min_depth_m=float(config.get("min_depth_m", 0.15)),
            max_depth_m=float(config.get("max_depth_m", 4.0)),
            min_valid_depth_ratio=float(config.get("min_valid_depth_ratio", 0.5)),
            min_depth_std_m=float(config.get("min_depth_std_m", 0.02)),
            min_depth_iqr_m=float(config.get("min_depth_iqr_m", 0.03)),
            max_near_ratio=float(config.get("max_near_ratio", 0.8)),
            max_far_ratio=float(config.get("max_far_ratio", 0.8)),
            edge_band_m=float(config.get("edge_band_m", 0.05)),
            max_median_delta_m=float(config.get("max_median_delta_m", 0.5)),
            min_valid_mask_iou=float(config.get("min_valid_mask_iou", 0.5)),
        ),
    )
    return {
        "stage": "depth_numeric_quality",
        "status": "complete",
        "depth_dir": str(depth_dir),
        "output_dir": str(output_path),
        "report_path": str(output_path / "depth_numeric_quality_report.json"),
        "summary": report.get("summary", {}),
        "outputs": report.get("outputs", {}),
    }


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value


def _quality_output_dir(config: dict[str, Any], manifest: Path, quality_name: str) -> Path:
    output_dir = _optional_str(config.get("output_dir"))
    if output_dir:
        return Path(output_dir)
    data_root = manifest.parent.parent if manifest.parent.name == "dataset_manifest" else manifest.parent
    return data_root / "quality" / quality_name
