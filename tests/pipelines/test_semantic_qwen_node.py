from pathlib import Path

import pytest

import json

from roboego_hand_vis.pipelines.semantic.nodes import (
    run_build_hand_estimation_windows_node,
    run_build_processing_plan_node,
    run_hand_visibility_quality_gate_node,
    run_qwen_frame_annotation_node,
    run_stage1_action_extraction_chunks_node,
    run_stage1_action_text_resolver_chunks_node,
    run_stage1_action_text_review_chunks_node,
)
from roboego_hand_vis.pipelines.semantic.pipeline import create_pipeline


def test_qwen_frame_annotation_node_skips_when_disabled():
    result = run_qwen_frame_annotation_node([], {"enabled": False})

    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "skipped",
        "reason": "disabled",
    }


def test_qwen_frame_annotation_node_requires_video_when_enabled():
    with pytest.raises(ValueError, match="video"):
        run_qwen_frame_annotation_node([], {"enabled": True})


def test_qwen_frame_annotation_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.semantic.nodes.subprocess.run",
        fake_run,
    )

    video = tmp_path / "session" / "videos" / "left.mp4"
    output_json = tmp_path / "session" / "annotation" / "qwen_frame_annotations_left_v4.json"
    result = run_qwen_frame_annotation_node(
        [
            {
                "start": 0.0,
                "end": 5.0,
                "route_hint": "pass_to_qwen",
            }
        ],
        {
            "enabled": True,
            "video": str(video),
            "output_json": str(output_json),
            "checkpoint_json": str(output_json.with_suffix(".checkpoint.json")),
            "raw_output_dir": str(tmp_path / "raw"),
            "batch_output_dir": str(tmp_path / "batches"),
            "chunk_output_dir": str(tmp_path / "chunks"),
            "fps": 1,
            "batch_size": 10,
            "batches_per_chunk": 5,
            "model": "qwen-vl-plus",
            "resume": True,
        },
    )

    assert calls[0]["check"] is True
    assert calls[0]["args"] == [
        "python",
        "scripts/quality/run_qwen_frame_annotation.py",
        "--video",
        str(video),
        "--output_json",
        str(output_json),
        "--checkpoint_json",
        str(output_json.with_suffix(".checkpoint.json")),
        "--raw_output_dir",
        str(tmp_path / "raw"),
        "--batch_output_dir",
        str(tmp_path / "batches"),
        "--chunk_output_dir",
        str(tmp_path / "chunks"),
        "--fps",
        "1",
        "--batch_size",
        "10",
        "--batches_per_chunk",
        "5",
        "--model",
        "qwen-vl-plus",
        "--windows_json",
        str(output_json.parent / "qwen_frame_annotation_windows.json"),
        "--resume",
    ]
    windows_payload = (output_json.parent / "qwen_frame_annotation_windows.json").read_text(encoding="utf-8")
    assert '"start": 0.0' in windows_payload
    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "complete",
        "video": str(video),
        "output_json": str(output_json),
        "valid_window_count": 1,
        "window_filter_applied": True,
        "usage": {},
        "usage_summary": {},
        "failed_batch_count": 0,
        "failed_batches": [],
    }


def test_qwen_frame_annotation_node_returns_usage_from_output_json(monkeypatch, tmp_path):
    def fake_run(args, check):
        output_path = Path(args[args.index("--output_json") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "usage": {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                    "usage_summary": {"tokens_per_frame": 14.0},
                    "status": "partial_failed",
                    "failed_batch_count": 1,
                    "failed_batches": [{"batch_idx": 2, "status": "timeout_error"}],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.semantic.nodes.subprocess.run",
        fake_run,
    )

    output_json = tmp_path / "annotation" / "qwen_frame_annotations_left_v4.json"
    result = run_qwen_frame_annotation_node(
        [{"start": 0.0, "end": 2.0, "route_hint": "pass_to_qwen"}],
        {"enabled": True, "video": "left.mp4", "output_json": str(output_json)},
    )

    assert result["usage"] == {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}
    assert result["usage_summary"] == {"tokens_per_frame": 14.0}
    assert result["status"] == "partial_failed"
    assert result["failed_batch_count"] == 1
    assert result["failed_batches"] == [{"batch_idx": 2, "status": "timeout_error"}]


def test_qwen_frame_annotation_node_skips_when_no_valid_windows():
    result = run_qwen_frame_annotation_node(
        [],
        {
            "enabled": True,
            "video": "left.mp4",
            "output_json": "annotations.json",
            "use_video_simple_quality": True,
        },
    )

    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "skipped",
        "reason": "no_valid_video_simple_quality_windows",
        "valid_window_count": 0,
    }


def test_qwen_frame_annotation_node_skips_when_video_simple_quality_failed(monkeypatch):
    calls = []

    def fake_run(args, check):
        calls.append(args)

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.semantic.nodes.subprocess.run",
        fake_run,
    )

    result = run_qwen_frame_annotation_node(
        {"passed": False, "status": "failed", "failed_intervals": [{"label": "frozen_video"}]},
        [],
        {
            "enabled": True,
            "video": "left.mp4",
            "output_json": "annotations.json",
            "require_video_simple_quality_pass": True,
            "use_video_simple_quality": False,
        },
    )

    assert calls == []
    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "skipped",
        "reason": "video_simple_quality_gate_not_passed",
        "failed_interval_count": 1,
    }


def test_qwen_frame_annotation_node_skips_when_raw_quality_blocked_video_quality(monkeypatch):
    calls = []

    def fake_run(args, check):
        calls.append(args)

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.semantic.nodes.subprocess.run",
        fake_run,
    )

    result = run_qwen_frame_annotation_node(
        {"status": "skipped", "reason": "raw_headband_quality_gate_not_passed"},
        [],
        {
            "enabled": True,
            "video": "left.mp4",
            "output_json": "annotations.json",
            "require_video_simple_quality_pass": True,
            "use_video_simple_quality": False,
        },
    )

    assert calls == []
    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "skipped",
        "reason": "raw_headband_quality_gate_not_passed",
    }


def test_qwen_frame_annotation_node_runs_without_video_simple_quality_when_optional(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.semantic.nodes.subprocess.run",
        fake_run,
    )

    video = tmp_path / "session" / "videos" / "left.mp4"
    output_json = tmp_path / "session" / "annotation" / "qwen_frame_annotations_left_v4.json"

    result = run_qwen_frame_annotation_node(
        [],
        {
            "enabled": True,
            "video": str(video),
            "output_json": str(output_json),
            "fps": 1,
            "start_s": 10,
            "duration_s": 20,
            "use_video_simple_quality": False,
        },
    )

    assert "--windows_json" not in calls[0]["args"]
    assert calls[0]["args"] == [
        "python",
        "scripts/quality/run_qwen_frame_annotation.py",
        "--video",
        str(video),
        "--output_json",
        str(output_json),
        "--fps",
        "1",
        "--start_s",
        "10",
        "--duration_s",
        "20",
    ]
    assert result == {
        "stage": "qwen_frame_annotation",
        "status": "complete",
        "video": str(video),
        "output_json": str(output_json),
        "valid_window_count": 0,
        "window_filter_applied": False,
        "usage": {},
        "usage_summary": {},
        "failed_batch_count": 0,
        "failed_batches": [],
    }


def test_stage1_action_extraction_pipeline_can_start_from_persisted_chunks():
    pipeline = create_pipeline()
    stage1_node = next(node for node in pipeline.nodes if node.name == "stage1_action_extraction_chunks")

    assert stage1_node.inputs == ["params:stage1_action_chunks"]


def test_processing_plan_pipeline_can_start_from_persisted_files():
    pipeline = create_pipeline()
    plan_node = next(node for node in pipeline.nodes if node.name == "build_processing_plan")
    windows_node = next(node for node in pipeline.nodes if node.name == "build_hand_estimation_windows")
    visibility_node = next(node for node in pipeline.nodes if node.name == "hand_visibility_gate")

    assert plan_node.inputs == ["params:processing_plan"]
    assert windows_node.inputs == ["params:hand_estimation_windows"]
    assert visibility_node.inputs == ["params:hand_visibility_gate"]


def test_hand_visibility_gate_node_writes_report(tmp_path):
    stage1 = tmp_path / "qwen.json"
    windows = tmp_path / "windows.json"
    processing_plan = tmp_path / "processing_plan.json"
    output = tmp_path / "visibility.json"
    stage1.write_text(
        json.dumps(
            {
                "frames": [
                    {"t": 0, "hands_visible": {"left": True, "right": True}},
                    {"t": 1, "hands_visible": {"left": True, "right": True}},
                ]
            }
        ),
        encoding="utf-8",
    )
    windows.write_text(
        json.dumps({"windows": [{"window_id": "handwin_0000", "start": 0, "end": 1}]}),
        encoding="utf-8",
    )
    processing_plan.write_text(
        json.dumps({"segments": [{"segment_id": "seg_0000", "route": "process", "start": 0, "end": 1}]}),
        encoding="utf-8",
    )

    report = run_hand_visibility_quality_gate_node(
        {"status": "complete"},
        {
            "enabled": True,
            "stage1_json": str(stage1),
            "hand_windows_json": str(windows),
            "processing_plan_json": str(processing_plan),
            "output_json": str(output),
            "min_core_both_hands_visible_rate": 0.9,
            "unit_source": "processing_plan",
        },
    )

    assert output.exists()
    assert report["stage"] == "hand_visibility_quality_gate"
    assert report["passed"] is True
    assert report["passed_window_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["windows"][0]["window_id"] == "seg_0000"


def qwen_chunk_payload(t: int, activity: str = "wearer holds a product") -> dict:
    return {
        "stage": "qwen_ego_frame_annotation_chunk",
        "chunk_idx": 0,
        "batch_indices": [0],
        "time_range": {"start": t, "end": t},
        "frame_count": 1,
        "frames": [
            {
                "t": t,
                "activity": activity,
                "hand_visible": True,
                "hands_visible": {"left": False, "right": True},
                "hand_object_interaction": True,
                "interaction_objects": ["product"],
                "interaction_object_attributes": [{"name": "product", "hand_side": "right", "confidence": 0.9}],
                "attention_objects": ["product"],
                "task_objects": ["product"],
            }
        ],
    }


def test_stage1_action_extraction_chunks_processes_only_missing_outputs(tmp_path):
    qwen_dir = tmp_path / "qwen_chunks"
    action_dir = tmp_path / "action_chunks"
    qwen_dir.mkdir()
    action_dir.mkdir()
    (qwen_dir / "chunk_0000_batches_0000_0000_t000000_000009.json").write_text(
        json.dumps(qwen_chunk_payload(0)),
        encoding="utf-8",
    )
    existing = action_dir / "chunk_0000_batches_0000_0000_t000000_000009.json"
    existing.write_text('{"already": true}', encoding="utf-8")
    (qwen_dir / "chunk_0001_batches_0001_0001_t000010_000019.json").write_text(
        json.dumps(qwen_chunk_payload(10)),
        encoding="utf-8",
    )

    report = run_stage1_action_extraction_chunks_node(
        {"status": "complete"},
        {
            "enabled": True,
            "qwen_chunk_dir": str(qwen_dir),
            "action_chunk_dir": str(action_dir),
            "overwrite": False,
        },
    )

    generated = json.loads((action_dir / "chunk_0001_batches_0001_0001_t000010_000019.json").read_text(encoding="utf-8"))
    assert generated["stage"] == "stage1_action_extraction"
    assert generated["frames"][0]["frame_actions"][0]["action"] == "hold"
    assert report["processed_count"] == 1
    assert report["skipped_existing_count"] == 1


def test_stage1_action_text_review_chunks_builds_review_inputs(tmp_path):
    action_dir = tmp_path / "action_chunks"
    review_dir = tmp_path / "review_chunks"
    action_dir.mkdir()
    (action_dir / "chunk_0000_batches_0000_0000_t000000_000009.json").write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 0,
                        "activity": "wearer rotates something",
                        "hands_visible": {"left": True, "right": True},
                        "interaction_object_attributes": [],
                        "attention_objects": [],
                        "task_objects": [],
                        "frame_actions": [
                            {"action": "rotate", "object_name": "unknown", "hand_side": "unknown", "source": "unbound", "confidence": 0.45}
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = run_stage1_action_text_review_chunks_node(
        {"status": "complete"},
        {
            "enabled": True,
            "action_chunk_dir": str(action_dir),
            "review_chunk_dir": str(review_dir),
        },
    )

    review = json.loads((review_dir / "chunk_0000_batches_0000_0000_t000000_000009.json").read_text(encoding="utf-8"))
    assert review["task"] == "resolve_stage1_hand_action_object_bindings"
    assert review["frame_count"] == 1
    assert report["processed_count"] == 1


def test_stage1_action_text_resolver_chunks_skips_empty_reviews_without_api(tmp_path, monkeypatch):
    action_dir = tmp_path / "action_chunks"
    review_dir = tmp_path / "review_chunks"
    resolved_dir = tmp_path / "resolved_chunks"
    merged_json = tmp_path / "stage1_actions_left_v1_resolved.json"
    action_dir.mkdir()
    review_dir.mkdir()
    action_payload = {"frames": [{"t": 0, "frame_actions": []}]}
    chunk_name = "chunk_0000_batches_0000_0000_t000000_000009.json"
    (action_dir / chunk_name).write_text(json.dumps(action_payload), encoding="utf-8")
    (review_dir / chunk_name).write_text(json.dumps({"frame_count": 0, "frames": []}), encoding="utf-8")

    def fail_call(**kwargs):
        raise AssertionError("resolver API should not be called for empty review chunks")

    monkeypatch.setattr("roboego_hand_vis.pipelines.semantic.nodes.call_qwen_text_resolver", fail_call)

    report = run_stage1_action_text_resolver_chunks_node(
        {"status": "complete"},
        {
            "enabled": True,
            "action_chunk_dir": str(action_dir),
            "review_chunk_dir": str(review_dir),
            "resolved_chunk_dir": str(resolved_dir),
            "merged_output_json": str(merged_json),
            "model": "qwen-plus",
        },
    )

    resolved = json.loads((resolved_dir / chunk_name).read_text(encoding="utf-8"))
    merged = json.loads(merged_json.read_text(encoding="utf-8"))
    assert resolved["frames"] == action_payload["frames"]
    assert merged["stage"] == "stage1_action_resolved_chunks_merged"
    assert merged["source_chunk_dir"] == str(resolved_dir)
    assert merged["source_files"] == [str(resolved_dir / chunk_name)]
    assert merged["frame_count"] == 1
    assert merged["frames"] == action_payload["frames"]
    assert report["merged_output_json"] == str(merged_json)
    assert resolved["text_resolver_frame_count"] == 0
    assert report["empty_review_count"] == 1


def test_build_processing_plan_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.semantic.nodes.subprocess.run", fake_run)

    stage1_json = tmp_path / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    actions_json = tmp_path / "annotation/stage1_action_resolved_chunks/chunk_0000.json"
    output_json = tmp_path / "annotation/processing_plan_left_v1.json"
    report = run_build_processing_plan_node(
        {"status": "complete"},
        {
            "enabled": True,
            "python": "/env/bin/python",
            "stage1_json": str(stage1_json),
            "actions_json": str(actions_json),
            "output_json": str(output_json),
            "min_process_segment_s": 5,
            "context_expand_s": 2,
            "merge_process_gap_s": 3,
        },
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/env/bin/python",
                "scripts/quality/build_processing_plan.py",
                "--stage1_json",
                str(stage1_json),
                "--output_json",
                str(output_json),
                "--actions_json",
                str(actions_json),
                "--context_expand_s",
                "2",
                "--min_process_segment_s",
                "5",
                "--merge_process_gap_s",
                "3",
            ],
        }
    ]
    assert report["status"] == "complete"
    assert report["output_json"] == str(output_json)


def test_build_hand_estimation_windows_node_runs_existing_script(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.semantic.nodes.subprocess.run", fake_run)

    stage1_actions = tmp_path / "annotation/stage1_action_resolved_chunks/chunk_0000.json"
    processing_plan = tmp_path / "annotation/processing_plan_left_v1.json"
    output_json = tmp_path / "annotation/hand_estimation_windows_left_v1.json"
    report = run_build_hand_estimation_windows_node(
        {"status": "complete"},
        {
            "enabled": True,
            "python": "/env/bin/python",
            "stage1_actions": str(stage1_actions),
            "processing_plan": str(processing_plan),
            "output_json": str(output_json),
            "hand_window_mode": "two",
            "context_expand_s": 2,
            "merge_gap_s": 2,
            "merge_after_expand": True,
            "min_window_s": 5,
            "min_priority": "high",
            "min_core_duration_s": 1,
            "fps": 30,
            "require_both_hands": False,
        },
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/env/bin/python",
                "scripts/quality/build_hand_estimation_windows.py",
                "--stage1_actions",
                str(stage1_actions),
                "--processing_plan",
                str(processing_plan),
                "--output_json",
                str(output_json),
                "--context_expand_s",
                "2",
                "--merge_gap_s",
                "2",
                "--merge_after_expand",
                "--min_window_s",
                "5",
                "--min_priority",
                "high",
                "--min_core_duration_s",
                "1",
                "--fps",
                "30",
                "--no-require_both_hands",
                "--hand_window_mode",
                "two",
            ],
        }
    ]
    assert report["status"] == "complete"
    assert report["output_json"] == str(output_json)
