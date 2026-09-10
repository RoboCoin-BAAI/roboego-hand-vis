import json
from pathlib import Path

import pandas as pd

from scripts.report_dataset_audit import build_audit_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_session(root: Path, cp: str, name: str, *, final_success: bool) -> Path:
    session = root / cp / "session_20260826_100000" / "divide" / name
    _write_json(session / "meta.json", {"duration_sec": 24.0, "fps": 30, "synced_frames": 720})
    _write_json(
        session / "pipeline_state.json",
        {
            "stages": {
                "raw_quality": {"status": "success", "duration_s": 1.0},
                "qwen_annotation": {"status": "success", "duration_s": 80.0},
                "hand_windows": {"status": "success", "duration_s": 0.5},
                "stereo_summary_quality": {"status": "success", "duration_s": 5.0},
                "fast_foundation_depth": {"status": "success", "duration_s": 300.0},
                "hamer": {"status": "success", "duration_s": 120.0},
                "archive_depth_cache": {"status": "fail", "duration_s": 2.0},
                "final_hand_trajectory_polish": {
                    "status": "success" if final_success else "blocked",
                    "duration_s": 3.0,
                },
            }
        },
    )
    _write_json(
        session / "annotation/quality_report.json",
        {
            "raw_headband_quality_gate": {"passed": True, "failed_checks": []},
            "modality_presence": {
                "videos:left": {"exists": True},
                "videos:right": {"exists": True},
                "audio": {"exists": True},
            },
        },
    )
    _write_json(
        session / "annotation/stage1_actions_left_v1_resolved.json",
        {
            "frame_count": 24,
            "action_frame_count": 20,
            "frames": [
                {
                    "scene": "shelf",
                    "frame_actions": [{"action": "reach", "object_name": "products"}],
                    "interaction_objects": ["product"],
                    "privacy": False,
                }
            ],
        },
    )
    _write_json(
        session / "annotation/hand_estimation_windows_left_v1.json",
        {
            "window_count": 1,
            "windows": [
                {
                    "window_id": "handwin_0000",
                    "duration": 24,
                    "semantic_label": "reach",
                    "actions": ["reach", "hold"],
                    "objects": ["product"],
                    "segment_frame_range": {"start": 0, "end": 719, "num_frames": 720},
                }
            ],
        },
    )
    _write_json(
        session / "quality/stereo_summary/stereo_quality_report.json",
        {
            "summary": {
                "passed": False,
                "strict_passed": False,
                "usable_for_depth": True,
                "depth_risk_level": "high_risk",
                "depth_risk_reasons": ["geometry_warning"],
                "checks": {"median_abs_dy": False, "p90_abs_dy": False},
                "median_abs_dy_avg": 8.5,
                "p90_abs_dy_max": 171.0,
                "orb_matches_min": 91,
            }
        },
    )
    _write_json(
        session / "hand/final_hand_result/trajectory_polish_report.json",
        {
            "ego_left": {
                "initial_valid": 680,
                "interpolated": 20,
                "final_valid": 700,
                "intervals": [{"start": 0, "end": 699}],
            },
            "ego_right": {
                "initial_valid": 680,
                "interpolated": 30,
                "final_valid": 700,
                "intervals": [{"start": 0, "end": 699}],
            },
        },
    )
    _write_json(session / "hand/final_hand_result/final_hand_summary.json", {"frame_count": 720})
    return session


def _make_windowed_session(root: Path, cp: str, name: str) -> Path:
    session = _make_session(root, cp, name, final_success=True)
    window = session / "hand_windows/handwin_0000_t000000_000023_reach"
    _write_json(
        window / "hand/final_hand_result/trajectory_polish_report.json",
        {
            "ego_left": {
                "initial_valid": 680,
                "interpolated": 20,
                "final_valid": 700,
                "intervals": [{"start": 0, "end": 699}],
            },
            "ego_right": {
                "initial_valid": 680,
                "interpolated": 30,
                "final_valid": 700,
                "intervals": [{"start": 0, "end": 699}],
            },
        },
    )
    _write_json(window / "hand/final_hand_result/final_hand_summary.json", {"frame_count": 720})
    return session


def _make_whole_session(root: Path, cp: str, name: str) -> Path:
    session = _make_session(root, cp, name, final_success=True)
    (session / "annotation/hand_estimation_windows_left_v1.json").unlink()
    return session


def test_build_audit_report_writes_all_and_final_layers(tmp_path: Path):
    root = tmp_path / "baai_ego_task"
    kept = _make_windowed_session(root, "cp_0001", "session_20260826_100000_整理货架_20260901000000")
    _make_session(root, "cp_0002", "session_20260826_100000_收银_20260901000001", final_success=False)
    _make_whole_session(root, "cp_0003", "session_20260826_100000_完整任务_20260901000002")
    _make_session(root, "cp_0005", "session_20260826_102423_物品称重标价_20260828133320_540f5020", final_success=False)
    annotated_but_not_materialized = _make_session(
        root, "cp_0004", "session_20260826_100000_只有标注窗口_20260901000003", final_success=True
    )
    output = tmp_path / "audit"

    summary = build_audit_report(root=root, output_dir=output, final_stage="final_hand_trajectory_polish")

    assert summary["all_session_count"] == 5
    assert summary["final_session_count"] == 3
    assert summary["all_total_duration_h"] == 120.0 / 3600.0
    assert summary["final_total_duration_h"] == 72.0 / 3600.0
    assert summary["final_hand_window_count"] == 1
    assert summary["final_whole_session_unit_count"] == 2
    assert summary["all_headbands"]["cp_0001"]["session_count"] == 1
    assert (output / "index.html").is_file()
    assert (output / "summary_zh.json").is_file()
    final_session_lines = (output / "final_sessions.txt").read_text(encoding="utf-8").splitlines()
    assert str(kept.resolve()) in final_session_lines
    assert str(annotated_but_not_materialized.resolve()) in final_session_lines
    assert len(final_session_lines) == 3

    all_sessions = pd.read_parquet(output / "all_sessions.parquet")
    assert len(all_sessions) == 5
    assert "物品称重标价_20260828133320" not in set(all_sessions["scene_label"])
    assert "物品称重标价" in set(all_sessions["scene_label"])

    final_sessions = pd.read_parquet(output / "final_sessions.parquet")
    assert len(final_sessions) == 3
    assert final_sessions.loc[0, "scene_label"] == "整理货架"
    assert bool(final_sessions.loc[0, "raw_quality_passed"]) is True

    windows = pd.read_parquet(output / "final_windows.parquet")
    assert windows.loc[0, "any_hand_valid_ratio"] >= 0.95
    assert set(windows["unit_type"]) == {"hand_window", "whole_session"}

    reasons = pd.read_parquet(output / "quality_reasons.parquet")
    assert "stereo:median_abs_dy" in set(reasons["reason"])

    semantic = pd.read_parquet(output / "semantic_distribution.parquet")
    assert "products" not in set(semantic["object"])
    assert "product" in set(semantic["object"])

    metrics = pd.read_parquet(output / "中文指标总览.parquet")
    assert "指标" in metrics.columns
    assert "指标数值" in metrics.columns
    assert "全量原始数据条数" in set(metrics["指标"])

    zh = json.loads((output / "summary_zh.json").read_text(encoding="utf-8"))
    assert zh["核心指标"]["全量原始数据条数"]["数值"] == 5
    assert zh["核心指标"]["全量切分片段总时长"]["数值"] == round(120.0 / 3600.0, 3)
    assert "已处理完成数据条数" not in zh["核心指标"]
    assert zh["已处理可用分析单元统计"]["已处理完成数据条数"]["数值"] == 3
    assert zh["已处理可用分析单元统计"]["已处理完成数据总时长"]["数值"] == round(72.0 / 3600.0, 3)
    assert "最终分析单元总数" not in zh["核心指标"]
    assert zh["已处理可用分析单元统计"]["已处理分析单元总数"]["数值"] == 3
    assert "已处理分析单元单条时长P50" not in zh["已处理可用分析单元统计"]
    assert not any("单条时长" in key for key in zh["已处理可用分析单元统计"])
    assert zh["已处理分析单元单条时长分布"]["20-30秒"]["数值"] == 3
    assert zh["已处理可用分析单元统计"]["已处理动作窗口数量"]["数值"] == 1
    assert zh["已处理可用分析单元统计"]["已处理完整session单元数量"]["数值"] == 2
    assert zh["已处理可用分析单元统计"]["已处理分析单元未纳入时长"]["数值"] == 0.0
    assert zh["已处理可用分析单元统计"]["已处理分析单元时长损失率"]["数值"] == 0.0
    assert "A级手部轨迹单元数" not in zh["已处理可用分析单元统计"]
    assert "A级手部轨迹单元数" not in zh["手部轨迹质量统计"]
    assert zh["手部轨迹质量统计"]["手部连续性90%-100%单元数"]["数值"] == 3
    assert zh["全量数据质检统计"]["原始质检通过数据条数"]["数值"] == 5
    assert zh["全量数据质检统计"]["原始质检阻塞数据条数"]["数值"] == 0
    assert zh["已处理数据结构与模态说明"][0]["目录"] == "videos/"
    assert any(item["目录"] == "data/" for item in zh["已处理数据结构与模态说明"])
    assert "统计解读与建议" not in zh
    assert "RGB 视频仍具备继续利用价值" in zh["全量数据质检小结"]["建议"]
    assert "任务设计" in zh["已处理语义分布小结"]["建议"]
    assert "有效筛选" in zh["已处理窗口化小结"]["建议"]
    assert "HAMER" in zh["手部轨迹连续性小结"]["建议"]
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "已处理数据结构与模态说明" in html
    assert "全量数据质检小结" in html
    assert "已处理语义分布小结" in html
    assert "手部轨迹 parquet" in html
    assert "kpi-grid" in html
    assert "donut-chart" in html
    assert "bar-chart" in html
    assert "line-chart" in html
    assert "treemap-chart" in html
    assert "bubble-cloud" not in html
    assert "生成文件" not in html
    assert "#4e79a7" in html
    assert "#2563eb" in html
    assert "流程漏斗指标" not in zh
    assert "质量与耗时诊断" not in zh
    assert summary["processed_stage_runtime"][0]["stage"] == "raw_quality"
    assert summary["processed_stage_runtime"][0]["success"] == 3
    assert all(row["success"] == 3 for row in summary["processed_stage_runtime"])
    assert all("blocked" not in row for row in summary["processed_stage_runtime"])
    assert all("fail" not in row for row in summary["processed_stage_runtime"])
    assert all("processing" not in row for row in summary["processed_stage_runtime"])
    assert all("missing" not in row for row in summary["processed_stage_runtime"])
    assert "archive_depth_cache" not in {row["stage"] for row in summary["processed_stage_runtime"]}
