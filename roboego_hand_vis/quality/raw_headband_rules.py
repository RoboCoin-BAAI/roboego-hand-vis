"""Rule evaluation for raw headband quality reports."""

from __future__ import annotations

from typing import Any


DEFAULT_RULES: dict[str, dict[str, Any]] = {
    "modality_presence": {
        "enabled": True,
        "level": "fail",
    },
    "camera_readability": {
        "enabled": True,
        "level": "fail",
    },
    "timestamp_readability": {
        "enabled": True,
        "level": "fail",
    },
    "timestamp_alignment": {
        "enabled": True,
        "level": "fail",
        "max_over_tolerance_rate": 0.0,
    },
    "timestamp_duplicates": {
        "enabled": True,
        "level": "warning",
        "max_duplicate_rate": 0.0,
    },
    "frame_count_consistency": {
        "enabled": True,
        "level": "fail",
        "max_delta_frames": 10,
    },
    "imu_missing": {
        "enabled": True,
        "level": "fail",
    },
    "imu_time_coverage": {
        "enabled": True,
        "level": "fail",
        "max_missing_gap_frames": 10,
    },
    "imu_duplicates": {
        "enabled": True,
        "level": "fail",
        "max_duplicate_rate": 0.0,
    },
    "imu_all_zero": {
        "enabled": True,
        "level": "fail",
    },
    "imu_frozen": {
        "enabled": True,
        "level": "fail",
    },
    "imu_jump": {
        "enabled": True,
        "level": "fail",
    },
    "video_fps_delta": {
        "enabled": True,
        "level": "warning",
        "max_delta": 1.0,
    },
}


def _rule_config(rules: dict[str, Any], name: str) -> dict[str, Any]:
    config = dict(DEFAULT_RULES[name])
    config.update(rules.get(name) or {})
    return config


def _add_check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    config: dict[str, Any],
    passed: bool,
    message: str,
    value: Any = None,
    threshold: Any = None,
    stream: str | None = None,
) -> None:
    if passed:
        return
    item = {
        "name": name,
        "level": str(config.get("level", "warning")),
        "passed": False,
        "value": value,
        "threshold": threshold,
        "message": message,
    }
    if stream is not None:
        item["stream"] = stream
    checks.append(item)


def _check_id(check: dict[str, Any]) -> str:
    stream = check.get("stream")
    if stream:
        return f"{check['name']}:{stream}"
    return str(check["name"])


def evaluate_quality_report(report: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate a raw headband quality report using configurable rules."""
    rules = rules or {}
    checks: list[dict[str, Any]] = []

    modality_rule = _rule_config(rules, "modality_presence")
    if modality_rule.get("enabled", True):
        for name, summary in sorted((report.get("modality_presence") or {}).items()):
            required = bool(summary.get("required", True))
            exists = bool(summary.get("exists", False))
            _add_check(
                checks,
                name="modality_presence",
                config=modality_rule,
                passed=(not required) or exists,
                stream=name,
                value=exists,
                threshold=True,
                message=f"Required modality {name} exists={exists}",
            )

    readability_rule = _rule_config(rules, "camera_readability")
    if readability_rule.get("enabled", True):
        for stream, summary in sorted((report.get("camera_readability") or {}).items()):
            readable = bool(summary.get("readable", False))
            _add_check(
                checks,
                name="camera_readability",
                config=readability_rule,
                passed=readable,
                stream=stream,
                value=readable,
                threshold=True,
                message=str(summary.get("error") or f"{stream} video is not readable"),
            )

    timestamp_readability_rule = _rule_config(rules, "timestamp_readability")
    if timestamp_readability_rule.get("enabled", True):
        for stream, summary in sorted((report.get("timestamp_readability") or {}).items()):
            readable = bool(summary.get("readable", False))
            _add_check(
                checks,
                name="timestamp_readability",
                config=timestamp_readability_rule,
                passed=readable,
                stream=stream,
                value=readable,
                threshold=True,
                message=str(summary.get("error") or f"{stream} timestamp is not readable"),
            )

    alignment_rule = _rule_config(rules, "timestamp_alignment")
    if alignment_rule.get("enabled", True):
        threshold = float(alignment_rule.get("max_over_tolerance_rate", 0.0))
        for stream, summary in sorted((report.get("timestamp_alignment") or {}).items()):
            count = int(summary.get("over_tolerance_count", 0))
            if "over_tolerance_rate" in summary:
                value = float(summary["over_tolerance_rate"])
            else:
                frame_count = int(summary.get("frame_count", 0))
                value = float(count / frame_count) if frame_count > 0 else float(count > 0)
            before_count = len(checks)
            _add_check(
                checks,
                name="timestamp_alignment",
                config=alignment_rule,
                passed=value <= threshold,
                stream=stream,
                value=value,
                threshold=threshold,
                message=f"{stream} timestamp over-tolerance rate is {value:.6f}",
            )
            if len(checks) > before_count:
                checks[-1]["count"] = count

    duplicate_rule = _rule_config(rules, "timestamp_duplicates")
    if duplicate_rule.get("enabled", True):
        threshold = float(duplicate_rule.get("max_duplicate_rate", 0.0))
        for stream, summary in sorted((report.get("timestamp_duplicates") or {}).items()):
            value = float(summary.get("duplicate_rate", 0.0))
            _add_check(
                checks,
                name="timestamp_duplicates",
                config=duplicate_rule,
                passed=value <= threshold,
                stream=stream,
                value=value,
                threshold=threshold,
                message=f"{stream} timestamp duplicate rate is {value:.6f}",
            )

    frame_count_rule = _rule_config(rules, "frame_count_consistency")
    frame_count_consistency = report.get("frame_count_consistency") or {}
    if frame_count_rule.get("enabled", True) and frame_count_consistency:
        threshold = int(frame_count_rule.get("max_delta_frames", 10))
        value = max(
            int(frame_count_consistency.get("max_video_frame_count_delta", 0)),
            int(frame_count_consistency.get("max_timestamp_frame_count_delta", 0)),
            int(frame_count_consistency.get("max_video_timestamp_count_delta", 0)),
        )
        _add_check(
            checks,
            name="frame_count_consistency",
            config=frame_count_rule,
            passed=value <= threshold,
            value=value,
            threshold=threshold,
            message=f"Max frame count delta across videos/timestamps is {value}",
        )

    imu = report.get("imu")
    imu_missing_rule = _rule_config(rules, "imu_missing")
    if imu_missing_rule.get("enabled", True):
        _add_check(
            checks,
            name="imu_missing",
            config=imu_missing_rule,
            passed=imu is not None,
            message="imu/imu0.csv is missing",
        )

    imu_coverage_rule = _rule_config(rules, "imu_time_coverage")
    imu_time_coverage = report.get("imu_time_coverage") or {}
    if imu_time_coverage.get("computed") and imu_coverage_rule.get("enabled", True):
        threshold = float(imu_coverage_rule.get("max_missing_gap_frames", 10))
        value = float(imu_time_coverage.get("max_missing_gap_frames", 0.0))
        _add_check(
            checks,
            name="imu_time_coverage",
            config=imu_coverage_rule,
            passed=value <= threshold,
            value=value,
            threshold=threshold,
            message=f"IMU time coverage missing gap is {value:.3f} frames",
        )

    imu_duplicate_rule = _rule_config(rules, "imu_duplicates")
    if imu is not None and imu_duplicate_rule.get("enabled", True):
        threshold = float(imu_duplicate_rule.get("max_duplicate_rate", 0.0))
        value = float(imu.get("duplicate_rate", 0.0))
        _add_check(
            checks,
            name="imu_duplicates",
            config=imu_duplicate_rule,
            passed=value <= threshold,
            value=value,
            threshold=threshold,
            message=f"IMU timestamp duplicate rate is {value:.6f}",
        )

    if imu is not None:
        signal_quality = imu.get("signal_quality") or {}
        imu_all_zero_rule = _rule_config(rules, "imu_all_zero")
        if imu_all_zero_rule.get("enabled", True):
            columns = list(signal_quality.get("all_zero_columns") or [])
            _add_check(
                checks,
                name="imu_all_zero",
                config=imu_all_zero_rule,
                passed=len(columns) == 0,
                value=columns,
                threshold=[],
                message=f"IMU all-zero signal columns: {columns}",
            )

        imu_frozen_rule = _rule_config(rules, "imu_frozen")
        if imu_frozen_rule.get("enabled", True):
            columns = list(signal_quality.get("frozen_columns") or [])
            _add_check(
                checks,
                name="imu_frozen",
                config=imu_frozen_rule,
                passed=len(columns) == 0,
                value=columns,
                threshold=[],
                message=f"IMU frozen signal columns: {columns}",
            )

        imu_jump_rule = _rule_config(rules, "imu_jump")
        if imu_jump_rule.get("enabled", True):
            columns = list(signal_quality.get("jump_columns") or [])
            _add_check(
                checks,
                name="imu_jump",
                config=imu_jump_rule,
                passed=len(columns) == 0,
                value=columns,
                threshold=[],
                message=f"IMU jump signal columns: {columns}",
            )

    fps_rule = _rule_config(rules, "video_fps_delta")
    expected_fps = report.get("fps")
    if expected_fps is not None and fps_rule.get("enabled", True):
        threshold = float(fps_rule.get("max_delta", 1.0))
        expected_fps_float = float(expected_fps)
        for stream, info in sorted((report.get("video_infos") or {}).items()):
            fps = info.get("fps")
            if fps is None:
                continue
            value = abs(float(fps) - expected_fps_float)
            _add_check(
                checks,
                name="video_fps_delta",
                config=fps_rule,
                passed=value <= threshold,
                stream=stream,
                value=value,
                threshold=threshold,
                message=f"{stream} video fps differs from meta fps by {value:.6f}",
            )

    failed_checks = [_check_id(check) for check in checks if check["level"] == "fail"]
    warnings = [_check_id(check) for check in checks if check["level"] == "warning"]
    passed = len(failed_checks) == 0
    return {
        "stage": "raw_headband_quality_gate",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "source_session": report.get("source_session"),
    }
