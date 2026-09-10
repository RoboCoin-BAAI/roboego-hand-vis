#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.pipeline_state import discover_sessions, read_pipeline_state


DEFAULT_OUTPUT_DIR = Path("reports/dataset_audit")
DEFAULT_FINAL_STAGE = "final_hand_trajectory_polish"
STAGE_COLUMNS = [
    "raw_quality",
    "raw_quality_gate",
    "video_simple_quality",
    "video_simple_quality_gate",
    "qwen_annotation",
    "post_annotation_actions",
    "processing_plan",
    "hand_windows",
    "materialize_auto",
    "prepare_stereo",
    "stereo_summary_quality",
    "stereo_quality",
    "fast_foundation_depth",
    "depth_numeric_quality",
    "archive_depth_cache",
    "hamer",
    "hand_filter",
    "joints21_pose_gate",
    "depth_anchored_joints21",
    "final_hand_result",
    "final_hand_trajectory_polish",
    "hand_parquet_export",
]


def build_audit_report(*, root: Path, output_dir: Path, final_stage: str = DEFAULT_FINAL_STAGE) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = _discover_audit_sessions(root)
    all_rows = [_session_row(session, final_stage=final_stage) for session in sessions]
    final_rows = [row for row in all_rows if row.get(f"stage_{final_stage}") == "success"]
    final_sessions = [Path(row["session"]) for row in final_rows]

    window_rows = []
    semantic_rows = []
    reason_rows = []
    runtime_rows = []
    for row in all_rows:
        session = Path(str(row["session"]))
        semantic_rows.extend(_semantic_rows(session, row, final_only=False))
        reason_rows.extend(_quality_reason_rows(session, row))
        runtime_rows.extend(_runtime_rows(session, row))
    for row in final_rows:
        session = Path(str(row["session"]))
        semantic_rows.extend(_semantic_rows(session, row, final_only=True))
        window_rows.extend(_window_rows(session, row))

    all_sessions_df = pd.DataFrame(all_rows)
    final_sessions_df = pd.DataFrame(final_rows)
    final_windows_df = pd.DataFrame(window_rows)
    semantic_df = pd.DataFrame(semantic_rows)
    reasons_df = pd.DataFrame(reason_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    _write_parquet(all_sessions_df, output_dir / "all_sessions.parquet")
    _write_parquet(final_sessions_df, output_dir / "final_sessions.parquet")
    _write_parquet(final_windows_df, output_dir / "final_windows.parquet")
    _write_parquet(semantic_df, output_dir / "semantic_distribution.parquet")
    _write_parquet(reasons_df, output_dir / "quality_reasons.parquet")
    _write_parquet(runtime_df, output_dir / "runtime.parquet")

    final_list = "\n".join(str(path) for path in final_sessions)
    (output_dir / "final_sessions.txt").write_text(final_list + ("\n" if final_list else ""), encoding="utf-8")

    summary = _summary(
        root=root,
        final_stage=final_stage,
        all_rows=all_rows,
        final_rows=final_rows,
        final_windows=window_rows,
        semantic_rows=semantic_rows,
        reason_rows=reason_rows,
        runtime_rows=runtime_rows,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_parquet(pd.DataFrame(summary.get("processed_stage_runtime", [])), output_dir / "processed_stage_runtime.parquet")
    summary_zh = _summary_zh(summary)
    (output_dir / "summary_zh.json").write_text(json.dumps(summary_zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_parquet(pd.DataFrame(_metric_overview_rows(summary_zh)), output_dir / "中文指标总览.parquet")
    _render_html(summary, output_dir / "index.html")
    return summary


def _session_row(session: Path, *, final_stage: str) -> dict[str, Any]:
    state = read_pipeline_state(session)
    stages = state.get("stages", {}) if isinstance(state.get("stages"), dict) else {}
    meta = _read_json(session / "meta.json")
    raw_report = _read_json(session / "annotation/quality_report.json")
    raw_gate = raw_report.get("raw_headband_quality_gate") if isinstance(raw_report.get("raw_headband_quality_gate"), dict) else {}
    stereo = _read_json(session / "quality/stereo_summary/stereo_quality_report.json")
    stereo_summary = stereo.get("summary") if isinstance(stereo.get("summary"), dict) else {}
    stage1 = _read_json(session / "annotation/stage1_actions_left_v1_resolved.json")
    windows = _read_json(session / "annotation/hand_estimation_windows_left_v1.json")

    frame_count = _number(meta.get("synced_frames") or meta.get("total_written_frames"))
    duration_s = _duration_s(meta, raw_report, stereo_summary)
    row: dict[str, Any] = {
        "session": str(Path(session).resolve()),
        "session_name": session.name,
        "headband": _headband(session),
        "scene_label": _scene_label(session.name),
        "duration_s": duration_s,
        "duration_h": duration_s / 3600.0 if duration_s else 0.0,
        "fps": _number(meta.get("fps") or raw_report.get("fps")) or 30.0,
        "frame_count": frame_count,
        "raw_quality_passed": _bool_or_none(raw_gate.get("passed")),
        "raw_failed_checks": "|".join(str(x) for x in raw_gate.get("failed_checks", []) if x),
        "stage1_frame_count": _number(stage1.get("frame_count")),
        "stage1_action_frame_count": _number(stage1.get("action_frame_count")),
        "stage1_action_ratio": _ratio(_number(stage1.get("action_frame_count")), _number(stage1.get("frame_count"))),
        "hand_window_count": _number(windows.get("window_count") or len(windows.get("windows", [])) if isinstance(windows.get("windows"), list) else 0),
        "hand_window_duration_s": _window_duration_sum(windows),
        "stereo_passed": _bool_or_none(stereo_summary.get("passed")),
        "stereo_strict_passed": _bool_or_none(stereo_summary.get("strict_passed")),
        "usable_for_depth": _bool_or_none(stereo_summary.get("usable_for_depth")),
        "depth_risk_level": str(stereo_summary.get("depth_risk_level") or ""),
        "depth_risk_reasons": "|".join(str(x) for x in stereo_summary.get("depth_risk_reasons", []) if x),
        "median_abs_dy_avg": _number(stereo_summary.get("median_abs_dy_avg")),
        "p90_abs_dy_max": _number(stereo_summary.get("p90_abs_dy_max")),
        "orb_matches_min": _number(stereo_summary.get("orb_matches_min")),
        "final_complete": stages.get(final_stage, {}).get("status") == "success" if isinstance(stages.get(final_stage), dict) else False,
    }
    row.update(_modality_presence(session, raw_report))
    for stage in STAGE_COLUMNS:
        record = stages.get(stage, {}) if isinstance(stages.get(stage), dict) else {}
        row[f"stage_{stage}"] = str(record.get("status") or "missing")
        row[f"duration_{stage}_s"] = _number(record.get("duration_s"))
    return row


def _discover_audit_sessions(root: Path) -> list[Path]:
    sessions = set(discover_sessions(root))
    if _looks_like_session_dir(root):
        sessions.add(root.resolve())
    for pattern in (
        "cp_*/session_*/divide/session_*",
        "cs_*/*/session_*/divide/session_*",
        "session_*/divide/session_*",
        "*/session_*/divide/session_*",
    ):
        for session in root.glob(pattern):
            if session.is_dir():
                sessions.add(session.resolve())
    return sorted(session.resolve() for session in sessions if _looks_like_session_dir(session))


def _looks_like_session_dir(path: Path) -> bool:
    return (
        (path / "meta.json").is_file()
        or (
            (path / "pipeline_state.json").is_file()
            and ((path / "annotation").is_dir() or (path / "videos").is_dir())
        )
    )


def _semantic_rows(session: Path, session_row: dict[str, Any], *, final_only: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage1 = _read_json(session / "annotation/stage1_actions_left_v1_resolved.json")
    for frame in stage1.get("frames", []) if isinstance(stage1.get("frames"), list) else []:
        if not isinstance(frame, dict):
            continue
        scene = str(frame.get("scene") or session_row.get("scene_label") or "")
        for action in frame.get("frame_actions", []) if isinstance(frame.get("frame_actions"), list) else []:
            if not isinstance(action, dict):
                continue
            rows.append(
                {
                    "scope": "final" if final_only else "all",
                    "session": session_row["session"],
                    "headband": session_row["headband"],
                    "scene": scene,
                    "action": str(action.get("action") or ""),
                    "object": _normalize_object_name(action.get("object_name") or action.get("object") or ""),
                    "count": 1,
                }
            )
        for obj in frame.get("interaction_objects", []) if isinstance(frame.get("interaction_objects"), list) else []:
            rows.append(
                {
                    "scope": "final" if final_only else "all",
                    "session": session_row["session"],
                    "headband": session_row["headband"],
                    "scene": scene,
                    "action": "",
                    "object": _normalize_object_name(obj),
                    "count": 1,
                }
            )
    return rows


def _normalize_object_name(value: Any) -> str:
    name = str(value or "").strip()
    aliases = {
        "products": "product",
    }
    return aliases.get(name.lower(), name)


def _window_rows(session: Path, session_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    windows = _read_json(session / "annotation/hand_estimation_windows_left_v1.json")
    window_by_id = {
        str(window.get("window_id") or ""): window
        for window in windows.get("windows", [])
        if isinstance(window, dict) and window.get("window_id")
    } if isinstance(windows.get("windows"), list) else {}
    for window_dir in _materialized_final_window_dirs(session):
        window_id = _window_id_from_dir(window_dir)
        window = window_by_id.get(window_id, {})
        if not isinstance(window, dict):
            window = {}
        num_frames = (
            _range_num_frames(window.get("segment_frame_range"))
            or _seconds_to_frames(window.get("duration"), session_row.get("fps"))
            or _frame_count_from_window_name(window_dir.name, session_row.get("fps"))
        )
        hand_root = window_dir
        polish = _read_json(hand_root / "hand/final_hand_result/trajectory_polish_report.json")
        final_summary = _read_json(hand_root / "hand/final_hand_result/final_hand_summary.json")
        frame_count = int(_number(final_summary.get("frame_count")) or num_frames or _number(session_row.get("frame_count")) or 0)
        hand_metrics = _hand_metrics(polish, frame_count)
        rows.append(
            {
                "session": session_row["session"],
                "unit": str(hand_root.resolve()),
                "unit_type": "hand_window",
                "headband": session_row["headband"],
                "scene_label": session_row["scene_label"],
                "window_id": window_id,
                "semantic_label": str(window.get("semantic_label") or _semantic_label_from_window_name(window_dir.name)),
                "duration_s": _number(window.get("duration")) or (num_frames / float(session_row.get("fps") or 30.0) if num_frames else 0.0),
                "num_frames": num_frames,
                "actions": "|".join(str(x) for x in window.get("actions", []) if x),
                "objects": "|".join(str(x) for x in window.get("objects", []) if x),
                "source_start_frame": _range_value(window.get("segment_frame_range"), "start"),
                "source_end_frame": _range_value(window.get("segment_frame_range"), "end"),
                **hand_metrics,
                "hand_continuity_bin": _hand_continuity_bin(hand_metrics.get("any_hand_valid_ratio")),
            }
        )
    if not rows:
        polish = _read_json(session / "hand/final_hand_result/trajectory_polish_report.json")
        final_summary = _read_json(session / "hand/final_hand_result/final_hand_summary.json")
        frame_count = int(_number(final_summary.get("frame_count")) or _number(session_row.get("frame_count")) or 0)
        hand_metrics = _hand_metrics(polish, frame_count)
    if not rows and frame_count:
        rows.append(
            {
                "session": session_row["session"],
                "unit": session_row["session"],
                "unit_type": "whole_session",
                "headband": session_row["headband"],
                "scene_label": session_row["scene_label"],
                "window_id": "session",
                "semantic_label": session_row["scene_label"],
                "duration_s": session_row.get("duration_s", 0.0),
                "num_frames": frame_count,
                "actions": "",
                "objects": "",
                "source_start_frame": 0,
                "source_end_frame": frame_count - 1,
                **hand_metrics,
                "hand_continuity_bin": _hand_continuity_bin(hand_metrics.get("any_hand_valid_ratio")),
            }
        )
    return rows


def _materialized_final_window_dirs(session: Path) -> list[Path]:
    root = session / "hand_windows"
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "hand/final_hand_result/trajectory_polish_report.json").is_file()
    )


def _window_id_from_dir(path: Path) -> str:
    match = re.match(r"(handwin_\d+)", path.name)
    return match.group(1) if match else path.name


def _semantic_label_from_window_name(name: str) -> str:
    match = re.match(r"handwin_\d+_t\d+_\d+_(.+)$", name)
    return match.group(1) if match else ""


def _frame_count_from_window_name(name: str, fps: Any) -> int:
    match = re.match(r"handwin_\d+_t(\d+)_(\d+)", name)
    if not match:
        return 0
    start_s = int(match.group(1))
    end_s = int(match.group(2))
    rate = _number(fps) or 30.0
    return int((end_s - start_s + 1) * rate)


def _find_window_dir(session: Path, window_id: str) -> Path | None:
    root = session / "hand_windows"
    if not root.is_dir() or not window_id:
        return None
    exact = root / window_id
    if exact.is_dir():
        return exact
    matches = sorted(path for path in root.glob(f"{window_id}*") if path.is_dir())
    return matches[0] if matches else None


def _quality_reason_rows(session: Path, session_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check in str(session_row.get("raw_failed_checks") or "").split("|"):
        if check:
            rows.append(_reason(session_row, "raw_quality", check, "raw:" + check))
    stereo = _read_json(session / "quality/stereo_summary/stereo_quality_report.json")
    summary = stereo.get("summary") if isinstance(stereo.get("summary"), dict) else {}
    checks = summary.get("checks") if isinstance(summary.get("checks"), dict) else {}
    for name, passed in checks.items():
        if passed is False:
            rows.append(_reason(session_row, "stereo_summary_quality", name, "stereo:" + str(name)))
    for reason in summary.get("depth_risk_reasons", []) if isinstance(summary.get("depth_risk_reasons"), list) else []:
        rows.append(_reason(session_row, "stereo_summary_quality", reason, "depth_risk:" + str(reason)))
    stages = read_pipeline_state(session).get("stages", {})
    for stage, record in stages.items() if isinstance(stages, dict) else []:
        if not isinstance(record, dict) or record.get("status") not in {"fail", "blocked"}:
            continue
        rows.append(_reason(session_row, str(stage), record.get("error") or record.get("status"), f"{stage}:{record.get('status')}"))
    return rows


def _runtime_rows(session: Path, session_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    stages = read_pipeline_state(session).get("stages", {})
    for stage, record in stages.items() if isinstance(stages, dict) else []:
        if not isinstance(record, dict):
            continue
        duration = _number(record.get("duration_s"))
        rows.append(
            {
                "session": session_row["session"],
                "headband": session_row["headband"],
                "stage": str(stage),
                "status": str(record.get("status") or ""),
                "duration_s": duration,
                "duration_per_data_minute_s": duration / (session_row["duration_s"] / 60.0) if duration and session_row["duration_s"] else None,
            }
        )
    return rows


def _summary(
    *,
    root: Path,
    final_stage: str,
    all_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    final_windows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_duration_h = sum(float(row.get("duration_h") or 0.0) for row in all_rows)
    final_duration_h = sum(float(row.get("duration_h") or 0.0) for row in final_rows)
    all_status = {stage: dict(Counter(row.get(f"stage_{stage}", "missing") for row in all_rows)) for stage in STAGE_COLUMNS}
    final_status = {stage: dict(Counter(row.get(f"stage_{stage}", "missing") for row in final_rows)) for stage in STAGE_COLUMNS}
    reason_counts = Counter(str(row["reason"]) for row in reason_rows)
    raw_reason_counts = Counter(
        str(row["reason"])
        for row in reason_rows
        if row.get("stage") == "raw_quality" and str(row.get("reason") or "").startswith("raw:")
    )
    runtime_by_stage: dict[str, list[float]] = defaultdict(list)
    final_session_set = {str(row.get("session")) for row in final_rows}
    final_runtime_by_stage: dict[str, list[float]] = defaultdict(list)
    for row in runtime_rows:
        duration = row.get("duration_s")
        if duration is not None and not (isinstance(duration, float) and math.isnan(duration)):
            runtime_by_stage[str(row["stage"])].append(float(duration))
            if str(row.get("session")) in final_session_set:
                final_runtime_by_stage[str(row["stage"])].append(float(duration))
    final_hand_window_count = sum(1 for row in final_windows if row.get("unit_type") == "hand_window")
    final_whole_session_unit_count = sum(1 for row in final_windows if row.get("unit_type") == "whole_session")
    return {
        "root": str(root),
        "final_stage": final_stage,
        "all_session_count": len(all_rows),
        "final_session_count": len(final_rows),
        "all_total_duration_h": all_duration_h,
        "final_total_duration_h": final_duration_h,
        "final_window_count": len(final_windows),
        "final_hand_window_count": final_hand_window_count,
        "final_whole_session_unit_count": final_whole_session_unit_count,
        "final_window_duration_h": sum(float(row.get("duration_s") or 0.0) for row in final_windows) / 3600.0,
        "all_headbands": _headband_summary(all_rows),
        "final_headbands": _headband_summary(final_rows),
        "stage_counts_all": all_status,
        "stage_counts_final": final_status,
        "duration_distribution_all_s": _distribution([float(row.get("duration_s") or 0.0) for row in all_rows]),
        "duration_distribution_final_s": _distribution([float(row.get("duration_s") or 0.0) for row in final_rows]),
        "duration_distribution_final_units_s": _distribution([float(row.get("duration_s") or 0.0) for row in final_windows]),
        "final_unit_duration_bins": _duration_bins([float(row.get("duration_s") or 0.0) for row in final_windows]),
        "top_quality_reasons": reason_counts.most_common(30),
        "top_raw_quality_reasons": raw_reason_counts.most_common(30),
        "top_scenes_all": Counter(str(row.get("scene_label") or "") for row in all_rows).most_common(100),
        "top_scenes_final": Counter(str(row.get("scene_label") or "") for row in final_rows).most_common(100),
        "top_actions_all": Counter(
            str(row.get("action") or "") for row in semantic_rows if row.get("scope") == "all" and row.get("action")
        ).most_common(100),
        "top_actions_final": Counter(
            str(row.get("action") or "") for row in semantic_rows if row.get("scope") == "final" and row.get("action")
        ).most_common(100),
        "top_objects_all": Counter(
            str(row.get("object") or "") for row in semantic_rows if row.get("scope") == "all" and row.get("object")
        ).most_common(100),
        "top_objects_final": Counter(
            str(row.get("object") or "") for row in semantic_rows if row.get("scope") == "final" and row.get("object")
        ).most_common(100),
        "hand_continuity_bins": _hand_continuity_bins(final_windows),
        "slowest_stages": [
            {"stage": stage, "total_duration_s": sum(values), "avg_duration_s": sum(values) / len(values), "count": len(values)}
            for stage, values in sorted(runtime_by_stage.items(), key=lambda item: sum(item[1]), reverse=True)[:20]
        ],
        "processed_stage_runtime": _processed_stage_runtime(final_status, final_runtime_by_stage, final_count=len(final_rows)),
        "interpretation": _interpretation(all_rows, final_rows, final_windows, reason_counts, runtime_by_stage),
    }


def _processed_stage_runtime(
    final_status: dict[str, dict[str, int]],
    final_runtime_by_stage: dict[str, list[float]],
    *,
    final_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for stage in STAGE_COLUMNS:
        status_counts = final_status.get(stage, {})
        success = int(status_counts.get("success") or 0)
        blocked = int(status_counts.get("blocked") or 0)
        fail = int(status_counts.get("fail") or 0)
        processing = int(status_counts.get("processing") or 0)
        missing = int(status_counts.get("missing") or 0)
        if success != final_count or blocked or fail or processing or missing:
            continue
        values = sorted(final_runtime_by_stage.get(stage, []))
        if not values:
            continue
        total = sum(values)
        rows.append(
            {
                "stage": stage,
                "success": success,
                "duration_count": len(values),
                "total_duration_s": total,
                "total_duration_h": total / 3600.0,
                "avg_duration_s": total / len(values) if values else 0.0,
                "p50_duration_s": _percentile(values, 50) if values else 0.0,
                "p90_duration_s": _percentile(values, 90) if values else 0.0,
            }
        )
    return rows


def _interpretation(
    all_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    final_windows: list[dict[str, Any]],
    reason_counts: Counter,
    runtime_by_stage: dict[str, list[float]],
) -> list[str]:
    notes = []
    if all_rows:
        notes.append(f"全库进入统计 {len(all_rows)} 条，已处理完成 {len(final_rows)} 条。")
    if reason_counts:
        notes.append(f"最常见质量/阻塞原因是 {reason_counts.most_common(1)[0][0]}。")
    if final_windows:
        notes.append(f"已处理分析单元手部连续性按 10% 分档统计，共 {len(final_windows)} 个单元。")
    if runtime_by_stage:
        slowest = max(runtime_by_stage.items(), key=lambda item: sum(item[1]))
        notes.append(f"累计耗时最长流程是 {slowest[0]}。")
    return notes


def _summary_zh(summary: dict[str, Any]) -> dict[str, Any]:
    all_count = int(summary.get("all_session_count") or 0)
    final_count = int(summary.get("final_session_count") or 0)
    final_units = int(summary.get("final_window_count") or 0)
    final_hand_windows = int(summary.get("final_hand_window_count") or 0)
    final_whole_sessions = int(summary.get("final_whole_session_unit_count") or 0)
    continuity_bins = summary.get("hand_continuity_bins", {}) if isinstance(summary.get("hand_continuity_bins"), dict) else {}
    stage_all = summary.get("stage_counts_all", {}) if isinstance(summary.get("stage_counts_all"), dict) else {}
    raw_success = _count_status(stage_all, "raw_quality", "success")
    raw_processing = _count_status(stage_all, "raw_quality", "processing")
    raw_blocked = _count_status(stage_all, "raw_quality", "blocked")
    raw_missing = _count_status(stage_all, "raw_quality", "missing")
    final_total_duration_h = float(summary.get("final_total_duration_h") or 0.0)
    final_unit_duration_h = float(summary.get("final_window_duration_h") or 0.0)
    final_unit_loss_h = max(final_total_duration_h - final_unit_duration_h, 0.0)
    final_unit_duration_bins = summary.get("final_unit_duration_bins", {}) if isinstance(summary.get("final_unit_duration_bins"), dict) else {}

    core = {
        "全量原始数据条数": _metric(all_count, "条", "扫描到的全部 session 数量。", "表示当前数据资产总入口规模。"),
        "全量切分片段总时长": _metric(round(float(summary.get("all_total_duration_h") or 0.0), 3), "小时", "全部 divide 子任务切片的时长总和。", "这是当前进入处理系统的片段级数据规模。"),
    }
    final_units_group = {
        "已处理完成数据条数": _metric(final_count, "条", "final_hand_trajectory_polish 成功的数据数量。", "这是当前可以进入手部 parquet/下游分析的数据范围。"),
        "已处理完成数据总时长": _metric(round(final_total_duration_h, 3), "小时", "已处理完成数据对应的 divide 子任务切片时长。", "用于衡量当前已完成处理的数据规模。"),
        "已处理完成率": _metric(_pct(_safe_ratio(final_count, all_count)), "%", "已处理完成数据条数 / 全量切分片段条数。", "表示当前处理进度，不等价于数据损失率。"),
        "已处理分析单元总数": _metric(final_units, "个", "已处理完成数据展开后的分析单元数量。", "hand_windows 存在时按窗口计数；没有 hand_windows 时整个 session 作为一个完整单元。"),
        "已处理动作窗口数量": _metric(final_hand_windows, "个", "已处理完成数据中实际存在的 hand/action windows 数量。", "只统计 hand_windows 下切出的动作窗口。"),
        "已处理完整session单元数量": _metric(final_whole_sessions, "个", "没有 hand_windows、但自身就是完整数据的已处理完成 session 数量。", "这类数据不应被误叫成动作窗口。"),
        "已处理分析单元总时长": _metric(round(final_unit_duration_h, 3), "小时", "已处理分析单元的时长总和。", "包含 hand_windows 动作片段，也包含完整 session 单元。"),
        "已处理分析单元未纳入时长": _metric(round(final_unit_loss_h, 3), "小时", "已处理完成数据切片时长中未进入分析单元的部分。", "主要来自 window 化后被排除的非动作、过渡或非目标片段。"),
        "已处理分析单元时长损失率": _metric(_pct(_safe_ratio(final_unit_loss_h, final_total_duration_h)), "%", "未纳入分析单元时长 / 已处理完成数据总时长。", "用于衡量从已处理 session 到可用分析单元的时间覆盖损失。"),
    }
    duration_group = {
        label: _metric(count, "个", "已处理分析单元按单条时长分段统计的数量。", "用于观察训练/分析样本长度分布。")
        for label, count in final_unit_duration_bins.items()
    }
    hand_quality_group = {
        **{
            f"手部连续性{label}单元数": _metric(count, "个", "按 any_hand_valid_ratio 每 10% 分档统计的分析单元数。", "any_hand_valid_ratio 表示左右手中至少一只手有效的帧比例。")
            for label, count in continuity_bins.items()
        }
    }
    raw_quality_group = {
        "原始质检通过数据条数": _metric(raw_success, "条", "raw_quality 成功的数据数量。", "反映基础视频、时间戳、IMU/音频等采集质量。"),
        "原始质检阻塞数据条数": _metric(raw_blocked, "条", "raw_quality blocked 的数据数量。", "这些数据未通过原始采集质量检查，需要看失败项。"),
        "原始质检缺失状态数据条数": _metric(raw_missing, "条", "pipeline_state 中 raw_quality 缺失的数据数量。", "这通常表示还没跑或状态文件不完整。"),
        "原始质检通过率": _metric(_pct(_safe_ratio(raw_success, all_count)), "%", "raw_quality 成功数量 / 全量数据条数。", "低时需要看硬件同步、IMU、视频完整性。"),
        "原始质检处理中数据条数": _metric(raw_processing, "条", "raw_quality 仍为 processing 的数据数量。", "这些数据不能归为失败，需要等流程完成后再判断。"),
        "原始质检最多问题": _metric(str((summary.get("top_raw_quality_reasons") or [["", 0]])[0][0]), "问题", "raw_quality 失败项中出现最多的问题。", f"出现 {(summary.get('top_raw_quality_reasons') or [['', 0]])[0][1]} 次。"),
    }
    return {
        "报告名称": "RoboEgo 数据生产全链路审计报告",
        "统计范围": {
            "数据根目录": summary.get("root"),
            "全量范围": "扫描到的全部 session",
            "已处理完成范围": f"{summary.get('final_stage')} == success 的 session",
        },
        "核心指标": core,
        "全量数据质检统计": raw_quality_group,
        "全量原始质检主要问题Top30": summary.get("top_raw_quality_reasons", []),
        "已处理可用分析单元统计": final_units_group,
        "已处理分析单元单条时长分布": duration_group,
        "已处理数据结构与模态说明": _processed_data_structure_notes(),
        **_analysis_notes(summary, raw_quality_group, final_units_group, continuity_bins),
        "手部轨迹质量统计": hand_quality_group,
        "头环分布_全量": summary.get("all_headbands", {}),
        "头环分布_最终完成": summary.get("final_headbands", {}),
        "场景分布_全量Top30": summary.get("top_scenes_all", []),
        "场景分布_最终完成Top30": summary.get("top_scenes_final", []),
        "动作分布_全量Top30": summary.get("top_actions_all", []),
        "动作分布_最终完成Top30": summary.get("top_actions_final", []),
        "物品分布_全量Top30": summary.get("top_objects_all", []),
        "物品分布_最终完成Top30": summary.get("top_objects_final", []),
        "手部连续性分档": continuity_bins,
        "主要问题Top30": summary.get("top_quality_reasons", []),
        "耗时流程Top20": summary.get("slowest_stages", []),
    }


def _metric(value: Any, unit: str, description: str, interpretation: str) -> dict[str, Any]:
    return {"数值": value, "单位": unit, "指标说明": description, "结果解读": interpretation}


def _processed_data_structure_notes() -> list[dict[str, str]]:
    return [
        {
            "目录": "videos/",
            "模态": "RGB 视频",
            "用途": "存放相机视频，通常包含原始左右目 left.mp4、right.mp4，以及去畸变后左右目 bleft.mp4、bright.mp4；window 数据下还会包含 left_rectified.mp4、right_rectified.mp4。",
        },
        {
            "目录": "audio/",
            "模态": "音频",
            "用途": "存放与当前 session 对齐的音频文件，例如 audio.wav；window 级导出通常通过相对路径指向父 session 的音频。",
        },
        {
            "目录": "annotation/",
            "模态": "语义标注",
            "用途": "存放场景、动作、物体等语义标注，主要使用 stage1_actions_left_v1_resolved.json 作为对外动作/物品统计来源。",
        },
        {
            "目录": "calibrations/",
            "模态": "相机标定",
            "用途": "存放原始相机标定、去畸变/立体校正参数；对外使用时重点关注 rectify_params.yaml 和导出 meta 中的视频相机信息。",
        },
        {
            "目录": "depth/",
            "模态": "深度",
            "用途": "存放深度估计与归档结果；当前对外推荐使用 fast_foundation_archive/depth_hevc12.mkv 和 valid_mask_ffv1.mkv。",
        },
        {
            "目录": "hand/",
            "模态": "手部轨迹",
            "用途": "存放手部估计、过滤、深度锚定和轨迹平滑后的中间/最终结果，是生成手部 parquet 的主要来源。",
        },
        {
            "目录": "hand_windows/",
            "模态": "动作窗口",
            "用途": "当一个 session 被切成多个动作片段时，每个 handwin_* 子目录是一段独立分析单元，并保留其对应的视频、深度、手部结果和 meta/data 导出。",
        },
        {
            "目录": "data/",
            "模态": "手部轨迹 parquet",
            "用途": "存放已处理分析单元的手部轨迹 parquet，例如 episode_000000.parquet，面向下游训练/分析直接读取。",
        },
        {
            "目录": "meta/",
            "模态": "数据索引与元信息",
            "用途": "存放 info.json、episodes.jsonl、tasks.jsonl、stats.json、roboego_sources.json 等，说明 parquet 字段、视频/深度/音频相对路径和统计信息。",
        },
    ]


def _analysis_notes(
    summary: dict[str, Any],
    raw_quality_group: dict[str, Any],
    final_units_group: dict[str, Any],
    continuity_bins: dict[str, int],
) -> dict[str, dict[str, str]]:
    raw_rate = raw_quality_group.get("原始质检通过率", {}).get("数值", 0)
    raw_top = raw_quality_group.get("原始质检最多问题", {}).get("数值", "")
    final_loss_rate = final_units_group.get("已处理分析单元时长损失率", {}).get("数值", 0)
    top_scenes = summary.get("top_scenes_final", [])[:5]
    top_actions = summary.get("top_actions_final", [])[:8]
    top_objects = summary.get("top_objects_final", [])[:8]
    low_continuity = sum(
        int(count)
        for label, count in continuity_bins.items()
        if label in {"0%-10%", "10%-20%", "20%-30%", "30%-40%", "40%-50%"}
    )
    total_units = sum(int(count) for count in continuity_bins.values())
    return {
        "全量数据质检小结": {
            "观察": f"原始质检通过率为 {raw_rate}%，主要问题集中在 {raw_top}。",
            "解读": "这说明部分片段的 IMU 质量会影响严格质检结论，但并不等价于 RGB 视频、语义标注、深度或手部分析完全不可用。",
            "建议": "建议后续按用途拆分可用性标签，例如 vision_usable、imu_usable、depth_usable；对手部轨迹、RGB-D 和动作理解任务，RGB 视频仍具备继续利用价值。",
        },
        "已处理语义分布小结": {
            "观察": f"已处理数据的高频场景为 {top_scenes}，高频动作为 {top_actions}，高频物品为 {top_objects}。",
            "解读": "当前数据很好地覆盖了超市场景中的主任务，但高频类别会支配训练分布，长尾动作和物品相对不足。",
            "建议": "建议后续通过任务设计扩大泛化覆盖，主动补充 open、pull、grasp、scan、handover、wipe 等低频动作，以及更多物品类别和交互方式。",
        },
        "已处理窗口化小结": {
            "观察": f"已处理分析单元时长损失率为 {final_loss_rate}%。",
            "解读": "这个损失率处在合理范围内，说明窗口化保留了大部分有效时长，同时过滤了部分过渡、无手或非目标动作片段。",
            "建议": "建议把这部分理解为面向手部任务的有效筛选，而不是简单的数据丢失；后续可继续报告窗口外时长占比来说明样本纯度。",
        },
        "手部轨迹连续性小结": {
            "观察": f"低于 50% 连续性的分析单元有 {low_continuity}/{total_units} 个。",
            "解读": "这说明虽然 209 条数据已经跑通全链路，但手部轨迹的连续可用比例仍是后续下游训练前最需要关注的问题。",
            "建议": "建议优先分析手是否出画、遮挡、近距离、运动模糊、HAMER 第一视角稳定性、左右手 identity 跳变，以及深度锚定是否放大异常。",
        },
    }


def _metric_overview_rows(summary_zh: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name in ("核心指标", "全量数据质检统计", "已处理可用分析单元统计", "已处理分析单元单条时长分布", "手部轨迹质量统计"):
        group = summary_zh.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for metric_name, item in group.items():
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "指标分组": group_name,
                    "指标": metric_name,
                    "指标数值": str(item.get("数值", "")),
                    "单位": item.get("单位"),
                    "指标说明": item.get("指标说明"),
                    "结果解读": item.get("结果解读"),
                }
            )
    return rows


def _count_status(stage_counts: dict[str, Any], stage: str, status: str) -> int:
    counts = stage_counts.get(stage, {}) if isinstance(stage_counts.get(stage), dict) else {}
    return int(counts.get(status) or 0)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _pct(value: float) -> float:
    return round(float(value) * 100.0, 2)


def _headband_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("headband") or "unknown")].append(row)
    return {
        headband: {
            "session_count": len(items),
            "duration_h": sum(float(item.get("duration_h") or 0.0) for item in items),
            "final_count": sum(1 for item in items if item.get("final_complete")),
        }
        for headband, items in sorted(grouped.items())
    }


def _render_html(summary: dict[str, Any], path: Path) -> None:
    summary_zh = _summary_zh(summary)
    core = summary_zh.get("核心指标", {})
    raw_quality = summary_zh.get("全量数据质检统计", {})
    final_units = summary_zh.get("已处理可用分析单元统计", {})
    kpis = [
        ("全量 session", _metric_display(core, "全量原始数据条数"), "数据资产入口"),
        ("全量时长", _metric_display(core, "全量切分片段总时长"), "divide 切片总规模"),
        ("质检通过率", _metric_display(raw_quality, "原始质检通过率"), "raw_quality"),
        ("已处理完成", _metric_display(final_units, "已处理完成数据条数"), "可进入下游"),
        ("分析单元", _metric_display(final_units, "已处理分析单元总数"), "session/window 展开后"),
        ("时长损失率", _metric_display(final_units, "已处理分析单元时长损失率"), "窗口化后未纳入"),
    ]
    raw_success = _metric_number(raw_quality, "原始质检通过数据条数")
    raw_blocked = _metric_number(raw_quality, "原始质检阻塞数据条数")
    raw_missing = _metric_number(raw_quality, "原始质检缺失状态数据条数")
    path.write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>RoboEgo 数据生产全链路审计报告</title>
<style>
:root{{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#dbe4f0;--panel:rgba(255,255,255,.86);--blue:#2563eb;--cyan:#0891b2;--green:#16a34a;--amber:#d97706;--red:#dc2626;--violet:#7c3aed}}
*{{box-sizing:border-box}}
body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:0;line-height:1.5;color:var(--ink);background:linear-gradient(135deg,#f8fbff 0%,#eef5f7 42%,#f7f4ec 100%);}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;background:linear-gradient(90deg,rgba(37,99,235,.06) 1px,transparent 1px),linear-gradient(0deg,rgba(37,99,235,.05) 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(#000,transparent 82%)}}
.page{{max-width:1320px;margin:0 auto;padding:34px 28px 56px;position:relative}}
.hero{{padding:30px 32px;border:1px solid rgba(148,163,184,.35);border-radius:18px;background:linear-gradient(135deg,rgba(15,23,42,.96),rgba(30,64,175,.86) 58%,rgba(20,184,166,.7));color:white;box-shadow:0 24px 80px rgba(15,23,42,.22)}}
.hero h1{{margin:0 0 8px;font-size:32px;letter-spacing:0}}
.hero p{{margin:0;color:rgba(255,255,255,.78)}}
.kpi-grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:18px 0 0}}
.kpi{{border:1px solid rgba(255,255,255,.2);border-radius:14px;padding:14px 14px;background:rgba(255,255,255,.12);backdrop-filter:blur(12px)}}
.kpi .label{{font-size:12px;color:rgba(255,255,255,.72)}}
.kpi .value{{font-size:25px;font-weight:760;margin-top:4px}}
.kpi .hint{{font-size:12px;color:rgba(255,255,255,.62);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.section{{margin-top:20px;padding:22px;border:1px solid rgba(148,163,184,.34);border-radius:16px;background:var(--panel);box-shadow:0 14px 42px rgba(15,23,42,.08);backdrop-filter:blur(18px)}}
.section h2{{margin:0 0 14px;font-size:20px;letter-spacing:0}}
.grid-2{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}}
.grid-3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}}
table{{border-collapse:separate;border-spacing:0;width:100%;margin:10px 0 0;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:white}}td,th{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}tr:last-child td{{border-bottom:0}}th{{background:#f8fafc;color:#334155;font-weight:700;font-size:13px}}td{{font-size:13px}}
code{{background:#eef2ff;color:#1e3a8a;padding:2px 5px;border-radius:6px}}
.donut-wrap{{display:flex;gap:18px;align-items:center;padding:8px 0}}
.donut-chart{{width:150px;height:150px;border-radius:50%;background:conic-gradient(var(--green) 0 var(--success-deg),var(--red) var(--success-deg) var(--blocked-deg),#cbd5e1 var(--blocked-deg) 360deg);position:relative;box-shadow:inset 0 0 0 1px rgba(15,23,42,.08)}}
.donut-chart:after{{content:"";position:absolute;inset:31px;border-radius:50%;background:white;box-shadow:inset 0 0 0 1px var(--line)}}
.donut-center{{position:absolute;inset:0;display:grid;place-items:center;font-weight:800;font-size:22px;z-index:1}}
.legend{{display:grid;gap:8px;font-size:13px;color:var(--muted)}}
.legend span:before{{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;background:var(--dot)}}
.bar-chart{{display:grid;gap:9px}}
.bar-row{{display:grid;grid-template-columns:170px 1fr 54px;gap:10px;align-items:center;font-size:13px}}
.bar-track{{height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden}}
.bar-fill{{height:100%;width:var(--w);background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:999px}}
.line-chart{{width:100%;height:auto;border:1px solid var(--line);border-radius:14px;background:linear-gradient(180deg,#fff,#f8fafc);margin-top:8px}}
.line-legend{{display:flex;gap:14px;font-size:12px;color:var(--muted);margin:8px 0 0}}
.line-legend span:before{{content:"";display:inline-block;width:18px;height:3px;border-radius:99px;background:var(--line-color);margin-right:6px;vertical-align:middle}}
.treemap-chart{{width:100%;height:280px;border:1px solid var(--line);border-radius:14px;background:white;box-shadow:inset 0 0 0 1px rgba(255,255,255,.6);margin-bottom:12px}}
.treemap-cell text{{paint-order:stroke;stroke:rgba(15,23,42,.24);stroke-width:2px;stroke-linejoin:round;font-weight:760;fill:#fff}}
.treemap-cell .count{{font-weight:650;opacity:.86}}
.note table td{{font-size:14px}}
@media (max-width:980px){{.kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.grid-2,.grid-3{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:120px 1fr 44px}}.hero{{padding:24px}}}}
</style>
<div class="page">
<section class="hero">
<h1>RoboEgo 数据生产全链路审计报告</h1>
<p>数据根目录: <code>{summary['root']}</code></p>
<div class="kpi-grid">
{_kpi_cards(kpis)}
</div>
</section>
<section class="section">
<h2>核心指标</h2>
{_zh_metric_table(summary_zh.get('核心指标', {}))}
</section>
<section class="section grid-2">
<div>
<h2>全量数据质检统计</h2>
{_zh_metric_table(summary_zh.get('全量数据质检统计', {}))}
</div>
<div>
<h2>质检通过/阻塞占比</h2>
{_donut_chart(raw_success, raw_blocked, raw_missing)}
<h2>全量原始质检主要问题 Top30</h2>
{_bar_chart(summary.get('top_raw_quality_reasons', []), limit=12)}
</div>
<div class="note" style="grid-column:1 / -1">
<h2>全量原始质检主要问题 Top30</h2>
{_counter_table(summary.get('top_raw_quality_reasons', []), limit=30)}
<h2>全量数据质检小结</h2>
{_note_table(summary_zh.get('全量数据质检小结', {}))}
</div>
</section>
<section class="section">
<h2>已处理可用分析单元统计</h2>
{_zh_metric_table(summary_zh.get('已处理可用分析单元统计', {}))}
<h2>已处理窗口化小结</h2>
{_note_table(summary_zh.get('已处理窗口化小结', {}))}
</section>
<section class="section grid-2">
<div>
<h2>已处理分析单元单条时长分布</h2>
{_zh_metric_table(summary_zh.get('已处理分析单元单条时长分布', {}))}
</div>
<div>
<h2>时长分布柱状图</h2>
{_metric_bar_chart(summary_zh.get('已处理分析单元单条时长分布', {}))}
</div>
</section>
<section class="section">
<h2>已处理数据结构与模态说明</h2>
{_data_structure_table(summary_zh.get('已处理数据结构与模态说明', []))}
</section>
<section class="section grid-2">
<div>
<h2>已处理数据流程耗时统计</h2>
{_processed_stage_runtime_table(summary.get('processed_stage_runtime', []))}
</div>
<div>
<h2>流程累计耗时柱状图</h2>
{_stage_runtime_chart(summary.get('processed_stage_runtime', []), limit=12)}
<h2>流程单条耗时折线图</h2>
{_stage_line_chart(summary.get('processed_stage_runtime', []), limit=12)}
</div>
</section>
<section class="section grid-3">
<div>
<h2>已处理数据场景分布 Top100</h2>
{_treemap_chart(summary.get('top_scenes_final', []), limit=24)}
{_counter_table(summary.get('top_scenes_final', []), limit=100)}
</div>
<div>
<h2>已处理数据动作分布 Top100</h2>
{_treemap_chart(summary.get('top_actions_final', []), limit=24)}
{_counter_table(summary.get('top_actions_final', []), limit=100)}
</div>
<div>
<h2>已处理数据物品分布 Top100</h2>
{_treemap_chart(summary.get('top_objects_final', []), limit=24)}
{_counter_table(summary.get('top_objects_final', []), limit=100)}
</div>
<div class="note" style="grid-column:1 / -1">
<h2>已处理语义分布小结</h2>
{_note_table(summary_zh.get('已处理语义分布小结', {}))}
</div>
</section>
<section class="section grid-2">
<div>
<h2>手部轨迹质量统计</h2>
{_zh_metric_table(summary_zh.get('手部轨迹质量统计', {}))}
</div>
<div>
<h2>手部连续性柱状图</h2>
{_metric_bar_chart(summary_zh.get('手部轨迹质量统计', {}))}
</div>
<div class="note" style="grid-column:1 / -1">
<h2>手部轨迹连续性小结</h2>
{_note_table(summary_zh.get('手部轨迹连续性小结', {}))}
</div>
</section>
</div>
""",
        encoding="utf-8",
    )


def _metric_display(group: Any, name: str) -> str:
    if not isinstance(group, dict):
        return "-"
    item = group.get(name)
    if not isinstance(item, dict):
        return "-"
    value = item.get("数值", "-")
    unit = item.get("单位", "")
    return f"{value}{unit}" if unit else str(value)


def _metric_number(group: Any, name: str) -> float:
    if not isinstance(group, dict):
        return 0.0
    item = group.get(name)
    if not isinstance(item, dict):
        return 0.0
    value = _number(item.get("数值"))
    return float(value or 0.0)


def _kpi_cards(items: list[tuple[str, str, str]]) -> str:
    return "".join(
        '<div class="kpi">'
        f'<div class="label">{_escape(label)}</div>'
        f'<div class="value">{_escape(value)}</div>'
        f'<div class="hint">{_escape(hint)}</div>'
        "</div>"
        for label, value, hint in items
    )


def _donut_chart(success: float, blocked: float, missing: float) -> str:
    total = max(success + blocked + missing, 1.0)
    success_deg = success / total * 360.0
    blocked_deg = (success + blocked) / total * 360.0
    success_pct = success / total * 100.0
    return (
        '<div class="donut-wrap">'
        f'<div class="donut-chart" style="--success-deg:{success_deg:.2f}deg;--blocked-deg:{blocked_deg:.2f}deg">'
        f'<div class="donut-center">{success_pct:.1f}%</div></div>'
        '<div class="legend">'
        f'<span style="--dot:var(--green)">通过 {int(success)}</span>'
        f'<span style="--dot:var(--red)">阻塞 {int(blocked)}</span>'
        f'<span style="--dot:#cbd5e1">缺失 {int(missing)}</span>'
        "</div></div>"
    )


def _bar_chart(items: list[list[Any]] | list[tuple[Any, Any]], *, limit: int = 12) -> str:
    visible = list(items[:limit])
    if not visible:
        return '<div class="bar-chart"></div>'
    max_value = max(float(_number(value) or 0.0) for _, value in visible) or 1.0
    rows = []
    for name, value in visible:
        number = float(_number(value) or 0.0)
        width = max(number / max_value * 100.0, 1.5)
        rows.append(
            '<div class="bar-row">'
            f'<div title="{_escape(str(name))}">{_escape(str(name))}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="--w:{width:.2f}%"></div></div>'
            f"<div>{number:g}</div>"
            "</div>"
        )
    return '<div class="bar-chart">' + "".join(rows) + "</div>"


def _metric_bar_chart(group: Any) -> str:
    if not isinstance(group, dict):
        return '<div class="bar-chart"></div>'
    items = [
        (name, item.get("数值"))
        for name, item in group.items()
        if isinstance(item, dict) and _number(item.get("数值")) is not None
    ]
    return _bar_chart(items, limit=len(items))


def _stage_runtime_chart(items: list[dict[str, Any]], *, limit: int = 12) -> str:
    pairs = [(item.get("stage") or "", item.get("total_duration_s") or 0.0) for item in items[:limit]]
    return _bar_chart(pairs, limit=limit)


def _stage_line_chart(items: list[dict[str, Any]], *, limit: int = 12) -> str:
    visible = list(items[:limit])
    if not visible:
        return '<svg class="line-chart" viewBox="0 0 720 240" role="img"></svg>'
    width = 720
    height = 240
    left = 46
    right = 18
    top = 24
    bottom = 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    avg_values = [float(_number(item.get("avg_duration_s")) or 0.0) for item in visible]
    p90_values = [float(_number(item.get("p90_duration_s")) or 0.0) for item in visible]
    max_value = max(avg_values + p90_values) or 1.0

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (plot_w * index / max(len(visible) - 1, 1))
        y = top + plot_h - (plot_h * value / max_value)
        return x, y

    def point_string(values: list[float]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, value) for i, value in enumerate(values)))

    grid = []
    for i in range(5):
        y = top + plot_h * i / 4
        label = max_value * (4 - i) / 4
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e2e8f0"/>'
            f'<text x="8" y="{y + 4:.1f}" font-size="10" fill="#64748b">{label:.0f}s</text>'
        )
    dots = []
    for series, values, color in (("平均", avg_values, "#2563eb"), ("P90", p90_values, "#f43f5e")):
        for index, value in enumerate(values):
            x, y = point(index, value)
            stage = _escape(str(visible[index].get("stage") or ""))
            dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                f"<title>{series} {stage}: {value:.1f}s</title></circle>"
            )
    labels = []
    for index, item in enumerate(visible):
        x, _ = point(index, 0)
        stage = str(item.get("stage") or "")
        label = stage[:10] + ("..." if len(stage) > 10 else "")
        labels.append(
            f'<text x="{x:.1f}" y="{height - 22}" font-size="10" fill="#64748b" '
            f'text-anchor="end" transform="rotate(-35 {x:.1f} {height - 22})">{_escape(label)}</text>'
        )
    return (
        '<svg class="line-chart" viewBox="0 0 720 240" role="img">'
        "<title>流程单条耗时折线图</title>"
        + "".join(grid)
        + f'<polyline points="{point_string(avg_values)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        + f'<polyline points="{point_string(p90_values)}" fill="none" stroke="#f43f5e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        + "".join(dots)
        + "".join(labels)
        + "</svg>"
        '<div class="line-legend"><span style="--line-color:#2563eb">平均秒/条</span><span style="--line-color:#f43f5e">P90秒</span></div>'
    )


def _treemap_chart(
    items: list[list[Any]] | list[tuple[Any, Any]],
    *,
    limit: int = 24,
) -> str:
    visible = list(items[:limit])

    chart_w = 720.0
    chart_h = 500.0

    if not visible:
        return (
            f'<svg class="treemap-chart" '
            f'viewBox="0 0 {chart_w:g} {chart_h:g}" '
            f'role="img"></svg>'
        )

    values = [
        max(float(_number(value) or 0.0), 0.0)
        for _, value in visible
    ]

    total = sum(values) or 1.0

    palette = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ]

    rects = _slice_treemap(
        values,
        0.0,
        0.0,
        chart_w,
        chart_h,
    )

    cells = []

    for index, ((name, value), (x, y, w, h)) in enumerate(
        zip(visible, rects)
    ):
        number = float(_number(value) or 0.0)
        area_ratio = number / total * 100.0
        color = palette[index % len(palette)]

        label = str(name)
        if len(label) > 16:
            label = label[:15] + "..."

        show_label = w >= 72 and h >= 42
        font_size = 13 if w >= 110 and h >= 62 else 11

        text = ""

        if show_label:
            text = (
                f'<text x="{x + 8:.1f}" '
                f'y="{y + 19:.1f}" '
                f'font-size="{font_size}">'
                f'{_escape(label)}</text>'
                f'<text class="count" '
                f'x="{x + 8:.1f}" '
                f'y="{y + 37:.1f}" '
                f'font-size="{max(font_size - 1, 10)}">'
                f'{number:g} / {area_ratio:.1f}%</text>'
            )

        cells.append(
            f'<g class="treemap-cell">'
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(w - 2, 0):.1f}" '
            f'height="{max(h - 2, 0):.1f}" '
            f'rx="5" fill="{color}">'
            f'<title>{_escape(str(name))}: '
            f'{number:g} ({area_ratio:.1f}%)</title>'
            f'</rect>'
            f'{text}'
            f'</g>'
        )

    return (
        f'<svg class="treemap-chart" '
        f'viewBox="0 0 {chart_w:g} {chart_h:g}" '
        f'role="img">'
        + "".join(cells)
        + "</svg>"
    )

# 
def _slice_treemap(
    values: list[float],
    x: float,
    y: float,
    w: float,
    h: float,
    depth: int = 0,
) -> list[tuple[float, float, float, float]]:
    if not values:
        return []

    if len(values) == 1:
        return [(x, y, w, h)]

    total = sum(values) or 1.0
    half = total / 2.0

    running = 0.0
    split = 1
    best_delta = abs(half - values[0])

    for index in range(1, len(values)):
        running += values[index - 1]
        delta = abs(half - running)

        if delta < best_delta:
            best_delta = delta
            split = index

    first_values = values[:split]
    second_values = values[split:]

    if not second_values:
        return [(x, y, w, h)]

    first_total = sum(first_values)

    # 前两层：保持你要的“向下排”
    if depth < 2:
        first_h = h * first_total / total

        return (
            _slice_treemap(
                first_values,
                x,
                y,
                w,
                first_h,
                depth + 1,
            )
            +
            _slice_treemap(
                second_values,
                x,
                y + first_h,
                w,
                h - first_h,
                depth + 1,
            )
        )

    # -----------------------------
    # 小矩形不要继续切成很薄的横条
    # -----------------------------
    MIN_CELL_H = 48.0

    horizontal_first_h = h * first_total / total
    horizontal_second_h = h - horizontal_first_h

    # 如果上下切会导致任意一块太矮，
    # 就强制改成左右切
    force_vertical_split = (
        horizontal_first_h < MIN_CELL_H
        or horizontal_second_h < MIN_CELL_H
    )

    if w >= h or force_vertical_split:
        first_w = w * first_total / total

        return (
            _slice_treemap(
                first_values,
                x,
                y,
                first_w,
                h,
                depth + 1,
            )
            +
            _slice_treemap(
                second_values,
                x + first_w,
                y,
                w - first_w,
                h,
                depth + 1,
            )
        )

    first_h = h * first_total / total

    return (
        _slice_treemap(
            first_values,
            x,
            y,
            w,
            first_h,
            depth + 1,
        )
        +
        _slice_treemap(
            second_values,
            x,
            y + first_h,
            w,
            h - first_h,
            depth + 1,
        )
    )

def _zh_metric_table(group: Any) -> str:
    if not isinstance(group, dict):
        return ""
    rows = []
    for name, item in group.items():
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{_escape(str(name))}</td>"
            f"<td>{_escape(str(item.get('数值', '')))}</td>"
            f"<td>{_escape(str(item.get('单位', '')))}</td>"
            f"<td>{_escape(str(item.get('指标说明', '')))}</td>"
            f"<td>{_escape(str(item.get('结果解读', '')))}</td>"
            "</tr>"
        )
    return "<table><tr><th>指标</th><th>数值</th><th>单位</th><th>指标说明</th><th>结果解读</th></tr>" + "".join(rows) + "</table>"


def _data_structure_table(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{_escape(str(item.get('目录', '')))}</td>"
        f"<td>{_escape(str(item.get('模态', '')))}</td>"
        f"<td>{_escape(str(item.get('用途', '')))}</td>"
        "</tr>"
        for item in items
        if isinstance(item, dict)
    )
    return f"<table><tr><th>目录</th><th>模态</th><th>用途</th></tr>{rows}</table>"


def _note_table(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return (
        "<table><tr><th>观察</th><th>解读</th><th>建议</th></tr>"
        "<tr>"
        f"<td>{_escape(str(item.get('观察', '')))}</td>"
        f"<td>{_escape(str(item.get('解读', '')))}</td>"
        f"<td>{_escape(str(item.get('建议', '')))}</td>"
        "</tr></table>"
    )


def _counter_table(items: list[list[Any]] | list[tuple[Any, Any]], *, limit: int = 30) -> str:
    rows = "".join(f"<tr><td>{_escape(str(k))}</td><td>{v}</td></tr>" for k, v in items[:limit])
    return f"<table><tr><th>名称</th><th>次数</th></tr>{rows}</table>"


def _processed_stage_runtime_table(items: list[dict[str, Any]]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{_escape(str(item.get('stage') or ''))}</td>"
        f"<td>{int(item.get('success') or 0)}</td>"
        f"<td>{float(item.get('total_duration_s') or 0.0):.1f}</td>"
        f"<td>{float(item.get('avg_duration_s') or 0.0):.1f}</td>"
        f"<td>{float(item.get('p50_duration_s') or 0.0):.1f}</td>"
        f"<td>{float(item.get('p90_duration_s') or 0.0):.1f}</td>"
        "</tr>"
        for item in items
    )
    return (
        "<table><tr><th>流程</th><th>成功</th>"
        "<th>累计秒</th><th>平均秒/条</th><th>P50秒</th><th>P90秒</th></tr>"
        + rows
        + "</table>"
    )


def _slow_stage_table(items: list[dict[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{_escape(str(item['stage']))}</td><td>{float(item['total_duration_s']):.1f}</td><td>{float(item['avg_duration_s']):.1f}</td><td>{item['count']}</td></tr>"
        for item in items[:20]
    )
    return f"<table><tr><th>流程</th><th>累计秒</th><th>平均秒/条</th><th>数量</th></tr>{rows}</table>"


def _hand_metrics(polish: dict[str, Any], frame_count: int) -> dict[str, Any]:
    frame_count = frame_count or _max_interval_end(polish) + 1
    metrics: dict[str, Any] = {"trajectory_frame_count": frame_count}
    ratios = []
    for side in ("ego_left", "ego_right"):
        data = polish.get(side, {}) if isinstance(polish.get(side), dict) else {}
        final_valid = _number(data.get("final_valid"))
        interpolated = _number(data.get("interpolated"))
        intervals = data.get("intervals", []) if isinstance(data.get("intervals"), list) else []
        ratio = _ratio(final_valid, frame_count)
        ratios.append(ratio or 0.0)
        metrics[f"{side}_final_valid"] = final_valid
        metrics[f"{side}_valid_ratio"] = ratio
        metrics[f"{side}_interpolated"] = interpolated
        metrics[f"{side}_interpolated_ratio"] = _ratio(interpolated, frame_count)
        metrics[f"{side}_interval_count"] = len(intervals)
        metrics[f"{side}_max_gap_frames"] = _max_gap(intervals, frame_count)
    metrics["any_hand_valid_ratio"] = max(ratios) if ratios else None
    metrics["both_hand_valid_ratio_lower_bound"] = min(ratios) if ratios else None
    return metrics


def _hand_continuity_bins(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = [_hand_continuity_label(i) for i in range(10)]
    counts = Counter(_hand_continuity_bin(row.get("any_hand_valid_ratio")) for row in rows)
    return {label: int(counts.get(label, 0)) for label in labels}


def _hand_continuity_bin(ratio: Any) -> str:
    value = _number(ratio)
    if value is None:
        return "0%-10%"
    index = min(max(int(float(value) * 10), 0), 9)
    return _hand_continuity_label(index)


def _hand_continuity_label(index: int) -> str:
    start = int(index) * 10
    return f"{start}%-{start + 10}%"


def _duration_bins(values: list[float]) -> dict[str, int]:
    labels = ["0-5秒", "5-10秒", "10-20秒", "20-30秒", "30-60秒", "1-2分钟", "2-5分钟", "5分钟以上"]
    counts = Counter(_duration_bin(value) for value in values if value is not None and not math.isnan(value))
    return {label: int(counts.get(label, 0)) for label in labels}


def _duration_bin(value: float) -> str:
    seconds = max(float(value), 0.0)
    if seconds < 5:
        return "0-5秒"
    if seconds < 10:
        return "5-10秒"
    if seconds < 20:
        return "10-20秒"
    if seconds < 30:
        return "20-30秒"
    if seconds < 60:
        return "30-60秒"
    if seconds < 120:
        return "1-2分钟"
    if seconds < 300:
        return "2-5分钟"
    return "5分钟以上"


def _max_gap(intervals: list[Any], frame_count: int) -> int:
    ranges = sorted(
        (int(item.get("start", 0)), int(item.get("end", item.get("start", 0))))
        for item in intervals
        if isinstance(item, dict)
    )
    if not ranges:
        return frame_count
    prev_end = -1
    max_gap = 0
    for start, end in ranges:
        max_gap = max(max_gap, start - prev_end - 1)
        prev_end = max(prev_end, end)
    return max(max_gap, frame_count - prev_end - 1)


def _max_interval_end(polish: dict[str, Any]) -> int:
    ends = []
    for side in ("ego_left", "ego_right"):
        data = polish.get(side, {}) if isinstance(polish.get(side), dict) else {}
        for item in data.get("intervals", []) if isinstance(data.get("intervals"), list) else []:
            if isinstance(item, dict):
                ends.append(int(item.get("end", 0)))
    return max(ends) if ends else 0


def _reason(session_row: dict[str, Any], stage: str, detail: Any, reason: str) -> dict[str, Any]:
    return {
        "session": session_row["session"],
        "headband": session_row["headband"],
        "scene_label": session_row["scene_label"],
        "stage": stage,
        "reason": reason,
        "detail": str(detail)[:2000],
    }


def _modality_presence(session: Path, raw_report: dict[str, Any]) -> dict[str, Any]:
    presence = raw_report.get("modality_presence") if isinstance(raw_report.get("modality_presence"), dict) else {}
    return {
        "has_left_video": _file_exists_or_report(session / "videos/left.mp4", presence, "videos:left"),
        "has_right_video": _file_exists_or_report(session / "videos/right.mp4", presence, "videos:right"),
        "has_bleft_video": _file_exists_or_report(session / "videos/bleft.mp4", presence, "videos:bleft"),
        "has_bright_video": _file_exists_or_report(session / "videos/bright.mp4", presence, "videos:bright"),
        "has_audio": _file_exists_or_report(session / "audio/audio.wav", presence, "audio"),
    }


def _file_exists_or_report(path: Path, presence: dict[str, Any], key: str) -> bool | None:
    if path.is_file():
        return True
    return _exists_value(presence, key)


def _exists_value(presence: dict[str, Any], key: str) -> bool | None:
    value = presence.get(key)
    if isinstance(value, dict):
        return _bool_or_none(value.get("exists"))
    return None


def _duration_s(meta: dict[str, Any], raw_report: dict[str, Any], stereo_summary: dict[str, Any]) -> float:
    for value in (
        meta.get("duration_sec"),
        meta.get("duration_s"),
        meta.get("span_sec"),
        raw_report.get("meta", {}).get("duration_sec") if isinstance(raw_report.get("meta"), dict) else None,
        stereo_summary.get("duration_s"),
    ):
        number = _number(value)
        if number and number > 0:
            return float(number)
    fps = _number(meta.get("fps")) or 30.0
    frames = _number(meta.get("synced_frames") or meta.get("total_written_frames"))
    return float(frames / fps) if frames and fps else 0.0


def _window_duration_sum(payload: dict[str, Any]) -> float:
    total = 0.0
    for window in payload.get("windows", []) if isinstance(payload.get("windows"), list) else []:
        if isinstance(window, dict):
            total += float(_number(window.get("duration")) or 0.0)
    return total


def _range_num_frames(payload: Any) -> int:
    if not isinstance(payload, dict):
        return 0
    value = _number(payload.get("num_frames"))
    if value:
        return int(value)
    start = _number(payload.get("start"))
    end = _number(payload.get("end"))
    return int(end - start + 1) if start is not None and end is not None else 0


def _range_value(payload: Any, key: str) -> int | None:
    if not isinstance(payload, dict):
        return None
    value = _number(payload.get(key))
    return int(value) if value is not None else None


def _seconds_to_frames(seconds: Any, fps: Any) -> int:
    duration = _number(seconds)
    rate = _number(fps) or 30.0
    return int(round(duration * rate)) if duration and rate else 0


def _headband(session: Path) -> str:
    for part in session.parts:
        if part.startswith(("cp_", "cs_")):
            return part
    return "unknown"


def _scene_label(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "session":
        return _clean_scene_label("_".join(parts[3:-1]) or name)
    match = re.search(r"session_\d+_\d+_(.+)_\d+$", name)
    return _clean_scene_label(match.group(1) if match else name)


def _clean_scene_label(label: str) -> str:
    value = str(label or "").strip("_")
    while True:
        cleaned = re.sub(r"_[0-9a-fA-F]{8}$", "", value)
        cleaned = re.sub(r"_\d{14}$", "", cleaned)
        cleaned = re.sub(r"_\d+(?:\.\d+)?-\d+(?:\.\d+)?$", "", cleaned)
        cleaned = cleaned.strip("_")
        if cleaned == value:
            return cleaned or str(label or "")
        value = cleaned


def _distribution(values: list[float]) -> dict[str, float]:
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return {}
    return {
        "min": values[0],
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": values[-1],
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = (len(values) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - idx) + values[hi] * (idx - lo))


def _ratio(numerator: Any, denominator: Any) -> float | None:
    num = _number(numerator)
    den = _number(denominator)
    if num is None or not den:
        return None
    return float(num) / float(den)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if str(value).lower() in {"true", "1", "yes"}:
        return True
    if str(value).lower() in {"false", "0", "no"}:
        return False
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        df = pd.DataFrame({"empty": []})
    df.to_parquet(path, index=False)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all-session and final-session RoboEgo dataset audit reports.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root, cp_* root, divide dir, or a single session.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-stage", default=DEFAULT_FINAL_STAGE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_audit_report(root=args.root, output_dir=args.output_dir, final_stage=args.final_stage)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote audit: {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
