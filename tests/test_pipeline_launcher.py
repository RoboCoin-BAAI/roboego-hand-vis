from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboego_hand_vis.pipeline_launcher import (
    LauncherRequest,
    WorkflowBlockedError,
    build_stage_invocations,
    load_dataset_config,
    render_invocation_command,
    run_launcher_request,
    run_stage_invocations,
    workflow_stage_names,
)
from roboego_hand_vis.pipeline_state import update_stage_state


def test_load_raw_dataset_config_has_hand_window_stage():
    config = load_dataset_config("raw")

    raw_stage = config["stages"]["raw_quality"]
    assert raw_stage["selector"] == "--to-nodes"
    assert raw_stage["node"] == "raw_headband_quality_gate"
    assert raw_stage["extra_params"]["raw_headband_quality_gate"]["strict"] is False

    stage = config["stages"]["depth_anchored_joints21"]
    assert stage["pipeline"] == "egohand"
    assert stage["node"] == "depth_anchored_joints21"
    assert stage["scope"] == "prepared_unit"
    assert stage["params"]["manifest"] == "{unit}/dataset_manifest/dataset_manifest.json"
    assert Path(config["stages"]["qwen_annotation"]["params"]["prompt"]).is_file()


def test_stage_can_use_to_nodes_selector_for_chained_work(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="post_annotation_actions"))[0]
    command = render_invocation_command(invocation)

    assert invocation.selector == "--to-nodes"
    assert "--pipelines semantic" in command
    assert "--to-nodes stage1_action_text_resolver_chunks" in command
    assert "stage1_action_chunks.qwen_chunk_dir=" in command


def test_runtime_kedro_bin_can_use_dataset_path_context(tmp_path: Path):
    config_path = tmp_path / "dataset.yml"
    config_path.write_text(
        """
dataset:
  fps: 30
runtime:
  kedro_bin: "{conda_root}/envs/ego/bin/kedro"
paths:
  conda_root: /opt/miniconda3
stages:
  archive:
    pipeline: depth
    node: archive_depth_cache
    scope: session
    param_namespace: archive_depth_cache
    params:
      enabled: true
      depth_dir: "{session}/depth/fast_foundation"
      output_dir: "{session}/depth/fast_foundation_archive"
""",
        encoding="utf-8",
    )

    invocation = build_stage_invocations(
        LauncherRequest(dataset="custom", config=config_path, session=tmp_path / "session", stage="archive")
    )[0]

    assert invocation.kedro_bin == "/opt/miniconda3/envs/ego/bin/kedro"


def test_raw_dataset_external_tool_paths_render_from_workspace_root(tmp_path: Path):
    session = tmp_path / "session"

    fast_invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="fast_foundation_depth_unit", unit=str(session))
    )[0]
    hamer_invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="hamer", unit=str(session))
    )[0]
    sam_invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="sam3_objects", hand_window=str(session))
    )[0]

    assert fast_invocation.params["fast_foundation_stereo_depth.fast_foundation_root"] == "/home/liuyou/Documents/Fast-FoundationStereo"
    assert hamer_invocation.params["hamer_unit.hamer_root"] == "/home/liuyou/Documents/hamer"
    assert hamer_invocation.params["hamer_unit.hamer_python"] == "/home/liuyou/miniconda3/envs/hamer/bin/python"
    assert sam_invocation.params["sam3_hand_object_segmentation.checkpoint"] == "/home/liuyou/Documents/do-as-i-do/sam3/sam3.pt"


def test_qwen_annotation_gates_on_video_simple_quality_without_window_filter(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="qwen_annotation"))[0]
    command = render_invocation_command(invocation)

    assert invocation.selector == "--to-nodes"
    assert invocation.node == "qwen_frame_annotation"
    assert "raw_headband_quality.overwrite=True" in command
    assert "raw_headband_quality.camera_streams=left%7Cright" in command or "raw_headband_quality.camera_streams=left|right" in command
    assert "pre_semantic_quality.output_json=" in command
    assert "video_simple_quality_left.json" in command
    assert "qwen_frame_annotation.require_video_simple_quality_pass=True" in command
    assert "qwen_frame_annotation.use_video_simple_quality=False" in command
    assert "qwen_frame_annotation.model=qwen3.6-flash" in command
    assert "qwen_frame_annotation.duration_s=" not in command
    assert "pre_semantic_quality_gate_left.json" not in command


def test_direct_hand_window_segments_stage_exports_to_session_hand_windows(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="direct_hand_window_segments"))[0]
    command = render_invocation_command(invocation)

    assert invocation.pipeline == "datasets"
    assert invocation.node == "export_direct_hand_window_segments"
    assert "export_direct_hand_window_segments.hand_windows_json=" in command
    assert f"export_direct_hand_window_segments.output_dir={session.resolve()}/hand_windows" in command
    assert "export_direct_hand_window_segments.camera_streams=left|right" in command
    assert "[left,right]" not in command


def test_egoblur_temporal_face_blur_stage_uses_session_video_defaults(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="egoblur_temporal_face_blur")
    )[0]
    command = render_invocation_command(invocation)

    assert invocation.pipeline == "privacy"
    assert invocation.node == "egoblur_temporal_face_blur"
    assert invocation.params["egoblur_temporal_face_blur.input_video"] == str(session.resolve() / "videos/left.mp4")
    assert invocation.params["egoblur_temporal_face_blur.output_video"] == str(session.resolve() / "videos/left_blur.mp4")
    assert (
        invocation.params["egoblur_temporal_face_blur.output_json"]
        == str(session.resolve() / "annotation/privacy/left_blur_report.json")
    )
    assert "egoblur_temporal_face_blur.detect_every_n_frames=3" in command


def test_materialize_session_stage_uses_session_as_unit(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_session"))[0]

    assert invocation.pipeline == "datasets"
    assert invocation.node == "prepare_stereo_unit"
    assert invocation.params["prepare_stereo_unit.unit_root"] == str(session.resolve())


def test_materialize_hand_windows_stage_exports_direct_windows(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_hand_windows"))[0]
    command = render_invocation_command(invocation)

    assert invocation.pipeline == "datasets"
    assert invocation.node == "export_direct_hand_window_segments"
    assert "export_direct_hand_window_segments.output_dir=" in command
    assert f"{session.resolve()}/hand_windows" in command


def test_materialize_segments_stage_exports_selected_segments(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_segments"))[0]

    assert invocation.pipeline == "datasets"
    assert invocation.node == "export_selected_video_segments"
    assert invocation.params["export_selected_video_segments.output_dir"] == str(session.resolve() / "segments")


def test_materialize_segment_hand_windows_stage_chains_segment_and_window_exports(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="materialize_segment_hand_windows")
    )[0]
    command = render_invocation_command(invocation)

    assert invocation.pipeline == "datasets"
    assert invocation.selector == "--to-nodes"
    assert invocation.node == "export_hand_window_segments"
    assert "export_selected_video_segments.output_dir=" in command
    assert f"prepare_selected_segments.segments_root={session.resolve()}/segments" in command
    assert f"export_hand_window_segments.segments_root={session.resolve()}/segments" in command


def test_materialize_auto_uses_session_for_short_sessions(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_auto"))[0]

    assert invocation.context["materialization_strategy"] == "session"
    assert invocation.context["materialization_target_stage"] == "materialize_session"
    assert invocation.node == "prepare_stereo_unit"
    assert invocation.params["prepare_stereo_unit.unit_root"] == str(session.resolve())


def test_materialize_auto_prefers_actual_video_duration_over_stale_meta(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    (session / "videos").mkdir(parents=True)
    (session / "meta.json").write_text(json.dumps({"duration_sec": 1872.982}), encoding="utf-8")
    (session / "videos/left.mp4").write_bytes(b"fake-video")
    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher._infer_video_duration_s", lambda path: 71.233)

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_auto"))[0]

    assert invocation.context["materialization_strategy"] == "session"
    assert invocation.context["materialization_target_stage"] == "materialize_session"
    assert invocation.context["materialization_duration_s"] == "71.233"
    assert invocation.node == "prepare_stereo_unit"


def test_materialize_auto_uses_hand_windows_for_medium_sessions(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 600}), encoding="utf-8")

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_auto"))[0]

    assert invocation.context["materialization_strategy"] == "hand_windows"
    assert invocation.context["materialization_target_stage"] == "materialize_hand_windows"
    assert invocation.node == "export_direct_hand_window_segments"


def test_materialize_auto_uses_segment_hand_windows_for_long_sessions(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 1200}), encoding="utf-8")

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="materialize_auto"))[0]

    assert invocation.context["materialization_strategy"] == "segment_hand_windows"
    assert invocation.context["materialization_target_stage"] == "materialize_segment_hand_windows"
    assert invocation.selector == "--to-nodes"
    assert invocation.node == "export_hand_window_segments"


def test_base_prepare_workflow_combines_planning_and_stereo_prepare(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")
    calls = []

    def fake_run(invocations, *, dry_run=False):
        for invocation in invocations:
            calls.append((invocation.node, invocation.context.get("materialization_strategy", "")))
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare", dry_run=True))

    assert calls == [
        ("qwen_frame_annotation", ""),
        ("stage1_action_text_resolver_chunks", ""),
        ("build_processing_plan", ""),
        ("build_hand_estimation_windows", ""),
        ("prepare_stereo_unit", "session"),
    ]


def test_base_prepare_workflow_prepares_direct_hand_windows_for_medium_session(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 600}), encoding="utf-8")
    (session / "hand_windows/handwin_0000_t000000_000065").mkdir(parents=True)
    calls = []

    def fake_run(invocations, *, dry_run=False):
        for invocation in invocations:
            calls.append((invocation.node, invocation.context.get("hand_window_id", "")))
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare", dry_run=True))

    assert calls == [
        ("qwen_frame_annotation", ""),
        ("stage1_action_text_resolver_chunks", ""),
        ("build_processing_plan", ""),
        ("build_hand_estimation_windows", ""),
        ("export_direct_hand_window_segments", ""),
        ("prepare_stereo_unit", "handwin_0000_t000000_000065"),
    ]


def test_base_prepare_workflow_skips_successful_steps(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")
    checkpoint = session / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"frames": []}), encoding="utf-8")
    for stage in [
        "raw_quality",
        "video_simple_quality",
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
        "materialize_auto",
        "prepare_stereo",
    ]:
        update_stage_state(session, stage, "success")

    calls = []

    def fake_run(invocations, *, dry_run=False):
        calls.extend(invocations)
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    commands = run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare"))

    assert commands == []
    assert calls == []


def test_base_prepare_workflow_force_reruns_successful_steps(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")
    checkpoint = session / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(json.dumps({"frames": []}), encoding="utf-8")
    for stage in [
        "raw_quality",
        "video_simple_quality",
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
        "materialize_auto",
        "prepare_stereo",
    ]:
        update_stage_state(session, stage, "success")

    calls = []

    def fake_run(invocations, *, dry_run=False):
        for invocation in invocations:
            calls.append(invocation.node)
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare", force=True))

    assert calls == [
        "qwen_frame_annotation",
        "stage1_action_text_resolver_chunks",
        "build_processing_plan",
        "build_hand_estimation_windows",
        "prepare_stereo_unit",
    ]


def test_base_prepare_workflow_exposes_planning_and_stereo_state_stages(tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")

    stages = workflow_stage_names(LauncherRequest(dataset="raw", session=session, stage="base_prepare"))

    assert stages == [
        "raw_quality",
        "video_simple_quality",
        "qwen_annotation",
        "post_annotation_actions",
        "processing_plan",
        "hand_windows",
        "materialize_auto",
        "prepare_stereo",
    ]


def test_base_prepare_stops_with_gate_reason_when_qwen_checkpoint_missing(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")
    report_path = session / "annotation" / "quality_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "raw_headband_quality_gate": {
                    "status": "failed",
                    "passed": False,
                    "checks": [
                        {
                            "name": "frame_count_consistency",
                            "level": "fail",
                            "passed": False,
                            "value": 21,
                            "threshold": 0,
                            "message": "Max frame count delta across videos/timestamps is 21",
                        }
                    ],
                    "failed_checks": ["frame_count_consistency"],
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_run(invocations, *, dry_run=False):
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    with pytest.raises(WorkflowBlockedError) as exc_info:
        run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare"))

    message = str(exc_info.value)
    assert "BASE PREPARE BLOCKED" in message
    assert "base_prepare blocked" in message
    assert "qwen_frame_annotations_left_v4.checkpoint.json" in message
    assert "frame_count_consistency" in message
    assert "value=21" in message
    assert "threshold=0" in message


def test_base_prepare_block_message_includes_failed_video_simple_quality(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 120}), encoding="utf-8")
    report_path = session / "annotation" / "video_simple_quality_left.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "passed": False,
                "status": "failed",
                "failed_intervals": [
                    {
                        "start": 2.0,
                        "end": 6.0,
                        "label": "frozen_video",
                        "reason": "near-identical frames over threshold duration",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(invocations, *, dry_run=False):
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    with pytest.raises(WorkflowBlockedError) as exc_info:
        run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="base_prepare"))

    message = str(exc_info.value)
    assert "video_simple_quality_gate=failed" in message
    assert "frozen_video" in message
    assert "2.000-6.000s" in message


def test_phase1_depth_workflow_runs_medium_session_then_all_hand_windows(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"duration_sec": 600}), encoding="utf-8")
    (session / "hand_windows/handwin_0000_t000000_000065").mkdir(parents=True)
    calls = []

    def fake_run(invocations, *, dry_run=False):
        for invocation in invocations:
            calls.append((invocation.node, invocation.context.get("hand_window_id", "")))
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="phase1_depth"))

    assert calls == [
        ("export_direct_hand_window_segments", ""),
        ("prepare_stereo_unit", "handwin_0000_t000000_000065"),
        ("stereo_summary_quality", "handwin_0000_t000000_000065"),
        ("stereo_segment_quality", "handwin_0000_t000000_000065"),
        ("fast_foundation_stereo_depth", "handwin_0000_t000000_000065"),
        ("depth_numeric_quality", "handwin_0000_t000000_000065"),
        ("archive_depth_cache", "handwin_0000_t000000_000065"),
    ]


def test_depth_estimation_workflow_runs_summary_then_fast_depth(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000065"
    (hand_window / "dataset_manifest").mkdir(parents=True)
    (hand_window / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")
    calls = []

    def fake_run(invocations, *, dry_run=False):
        for invocation in invocations:
            calls.append((invocation.node, invocation.context.get("hand_window_id", "")))
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="depth_estimation"))

    assert calls == [
        ("stereo_summary_quality", "handwin_0000_t000000_000065"),
        ("fast_foundation_stereo_depth", "handwin_0000_t000000_000065"),
    ]


def test_prepare_stereo_stage_runs_for_all_direct_hand_windows(tmp_path: Path):
    session = tmp_path / "session"
    (session / "hand_windows/handwin_0000_t000000_000065").mkdir(parents=True)
    (session / "hand_windows/handwin_0001_t000070_000151").mkdir(parents=True)

    invocations = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="prepare_stereo", all_hand_windows=True)
    )

    assert [item.context["unit"] for item in invocations] == [
        str(session / "hand_windows/handwin_0000_t000000_000065"),
        str(session / "hand_windows/handwin_0001_t000070_000151"),
    ]
    assert all(item.node == "prepare_stereo_unit" for item in invocations)
    assert invocations[0].params["prepare_stereo_unit.unit_root"] == str(
        session / "hand_windows/handwin_0000_t000000_000065"
    )


def test_prepare_stereo_stage_defaults_to_session_unit(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="prepare_stereo"))[0]

    assert invocation.context["unit"] == str(session)
    assert invocation.params["prepare_stereo_unit.unit_root"] == str(session)


def test_stereo_summary_quality_stage_runs_for_hand_windows(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000065"
    hand_window.mkdir(parents=True)

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="stereo_summary_quality", hand_window=hand_window.name)
    )[0]

    assert invocation.pipeline == "quality"
    assert invocation.node == "stereo_summary_quality"
    assert invocation.params["stereo_summary_quality.manifest"] == str(
        hand_window / "dataset_manifest/dataset_manifest.json"
    )
    assert invocation.params["stereo_summary_quality.output_dir"] == str(hand_window / "quality/stereo_summary")


def test_hand_window_short_id_matches_direct_hand_window_prefix(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000002_000065_place_scan_operate_reach"
    hand_window.mkdir(parents=True)

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="hand_filter", hand_window="handwin_0000")
    )[0]

    assert invocation.context["hand_window"] == str(hand_window)
    assert invocation.params["hand_filter.refined_json"] == str(
        hand_window / "hand/hamer_raw/ego_hand_candidates_refined.json"
    )


def test_depth_quality_stage_runs_for_hand_windows(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000065"
    hand_window.mkdir(parents=True)

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="depth_quality", hand_window=hand_window.name)
    )[0]

    assert invocation.pipeline == "quality"
    assert invocation.node == "depth_quality"
    assert invocation.params["depth_quality.manifest"] == str(hand_window / "dataset_manifest/dataset_manifest.json")
    assert invocation.params["depth_quality.depth_dir"] == str(hand_window / "depth/fast_foundation")
    assert invocation.params["depth_quality.output_dir"] == str(hand_window / "quality/depth")
    assert invocation.params["depth_quality.stereo_quality_report"] == str(
        hand_window / "quality/stereo_segments/bad_segments.json"
    )
    assert invocation.params["depth_quality.num_frames"] == "1800"


def test_depth_numeric_quality_stage_runs_for_hand_windows(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000065"
    hand_window.mkdir(parents=True)

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="depth_numeric_quality", hand_window=hand_window.name)
    )[0]

    assert invocation.pipeline == "quality"
    assert invocation.node == "depth_numeric_quality"
    assert invocation.params["depth_numeric_quality.depth_dir"] == str(hand_window / "depth/fast_foundation")
    assert invocation.params["depth_numeric_quality.output_dir"] == str(hand_window / "quality/depth_numeric")
    assert invocation.params["depth_numeric_quality.num_frames"] == "1800"


def test_prepared_unit_stage_defaults_to_direct_hand_windows_when_manifests_exist(tmp_path: Path):
    session = tmp_path / "session"
    for name in ("handwin_0000_t000000_000010", "handwin_0001_t000020_000030"):
        manifest = session / "hand_windows" / name / "dataset_manifest/dataset_manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 300}}}), encoding="utf-8")

    invocations = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="fast_foundation_depth")
    )

    assert [item.context["unit_id"] for item in invocations] == [
        "handwin_0000_t000000_000010",
        "handwin_0001_t000020_000030",
    ]
    assert invocations[0].params["fast_foundation_stereo_depth.manifest"] == str(
        session / "hand_windows/handwin_0000_t000000_000010/dataset_manifest/dataset_manifest.json"
    )
    assert invocations[0].params["fast_foundation_stereo_depth.output_dir"] == str(
        session / "hand_windows/handwin_0000_t000000_000010/depth/fast_foundation"
    )


def test_prepared_unit_stage_defaults_to_segment_hand_windows_before_session_manifest(tmp_path: Path):
    session = tmp_path / "session"
    (session / "dataset_manifest").mkdir(parents=True)
    (session / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")
    hand_window_manifest = (
        session
        / "segments/seg_0000_t000000_000020"
        / "hand_windows/handwin_0000_t000000_000020"
        / "dataset_manifest/dataset_manifest.json"
    )
    hand_window_manifest.parent.mkdir(parents=True)
    hand_window_manifest.write_text("{}", encoding="utf-8")

    invocations = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="depth_numeric_quality")
    )

    assert len(invocations) == 1
    assert invocations[0].context["unit"] == str(hand_window_manifest.parents[1])
    assert invocations[0].params["depth_numeric_quality.depth_dir"] == str(
        hand_window_manifest.parents[1] / "depth/fast_foundation"
    )


def test_prepared_unit_stage_falls_back_to_session_manifest(tmp_path: Path):
    session = tmp_path / "session"
    manifest = session / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 660}}}), encoding="utf-8")

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="fast_foundation_depth")
    )[0]

    assert invocation.context["unit"] == str(session)
    assert invocation.params["fast_foundation_stereo_depth.manifest"] == str(manifest)
    assert invocation.params["fast_foundation_stereo_depth.num_frames"] == "660"


def test_hand_postprocess_stages_default_to_prepared_units(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000010"
    manifest = hand_window / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 300, "height": 1050}}}), encoding="utf-8")

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="hand_filter")
    )[0]

    assert invocation.context["unit"] == str(hand_window)
    assert invocation.params["hand_filter.manifest"] == str(manifest)
    assert invocation.params["hand_filter.output_dir"] == str(hand_window / "hand/filtered")


def test_hamer_stage_defaults_to_prepared_units(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "hand_windows/handwin_0000_t000000_000010"
    manifest = hand_window / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"videos": {"left": {"frame_count": 300}}}), encoding="utf-8")

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="hamer"))[0]

    assert invocation.pipeline == "egohand"
    assert invocation.node == "hamer_unit"
    assert invocation.params["hamer_unit.manifest"] == str(manifest)
    assert invocation.params["hamer_unit.output_dir"] == str(hand_window / "hand/hamer_raw")
    assert invocation.params["hamer_unit.wearer_regions_json"] == str(
        session / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    )
    assert invocation.params["hamer_unit.num_frames"] == "300"


def test_video_simple_quality_stage_alias_maps_to_pre_semantic_gate(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="video_simple_quality"))[0]
    command = render_invocation_command(invocation)

    assert invocation.pipeline == "quality"
    assert invocation.node == "video_simple_quality_gate"
    assert invocation.selector == "--to-nodes"
    assert "--to-nodes video_simple_quality_gate" in command
    assert "raw_headband_quality.enabled=True" in command
    assert "raw_headband_quality.overwrite=True" in command
    assert "raw_headband_quality_gate.strict=False" in command
    assert "pre_semantic_quality.video=" in command
    assert "pre_semantic_quality.output_json=" in command
    assert "video_simple_quality_left.json" in command


def test_builds_depth_anchored_command_from_session_and_hand_window(tmp_path: Path):
    session = tmp_path / "session_20260818_182102"
    hand_window = (
        session
        / "segments/seg_0000_t000000_000065_operate_place_point_reach"
        / "hand_windows/handwin_0000_t000000_000065"
    )
    hand_window.mkdir(parents=True)

    invocations = build_stage_invocations(
        LauncherRequest(
            dataset="raw",
            session=session,
            stage="depth_anchored_joints21",
            hand_window="handwin_0000_t000000_000065",
        )
    )

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.pipeline == "egohand"
    assert invocation.node == "depth_anchored_joints21"
    assert invocation.params["depth_anchored_joints21.enabled"] is True
    assert invocation.params["depth_anchored_joints21.manifest"] == str(
        hand_window / "dataset_manifest/dataset_manifest.json"
    )
    assert invocation.params["depth_anchored_joints21.candidates_json"] == str(
        hand_window / "hand/joints21_pose_gate/final_hand_candidates_gated.json"
    )
    assert invocation.params["depth_anchored_joints21.depth_dir"] == str(hand_window / "depth/fast_foundation")
    assert invocation.params["depth_anchored_joints21.output_dir"] == str(hand_window / "hand/joints21_depth_anchored")


def test_infers_hand_window_frame_count_and_image_height_from_manifest(tmp_path: Path):
    session = tmp_path / "session_20260818_182102"
    hand_window = session / "segments/seg_0000/hand_windows/handwin_0000"
    manifest = hand_window / "dataset_manifest/dataset_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        """
{
  "videos": {
    "left": {"height": 1050, "frame_count": 1960}
  },
  "metadata": {
    "hand_window": {
      "local_frame_range": {"num_frames": 1980}
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="joints21_pose_gate", hand_window="handwin_0000")
    )[0]

    assert invocation.params["joints21_pose_gate.num_frames"] == "1980"
    assert invocation.params["joints21_pose_gate.image_height"] == "1050"


def test_all_hand_windows_builds_one_invocation_per_window(tmp_path: Path):
    session = tmp_path / "session"
    for name in ("handwin_0000_t000000_000010", "handwin_0001_t000020_000030"):
        (session / "segments/seg_0000/hand_windows" / name).mkdir(parents=True)

    invocations = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="final_hand_result", all_hand_windows=True)
    )

    assert [inv.context["hand_window_id"] for inv in invocations] == [
        "handwin_0000_t000000_000010",
        "handwin_0001_t000020_000030",
    ]
    assert all(inv.node == "final_hand_result" for inv in invocations)


def test_sam3_objects_defaults_to_all_hand_windows_for_batch_use(tmp_path: Path):
    session = tmp_path / "session"
    for name in ("handwin_0000_t000000_000010", "handwin_0001_t000020_000030"):
        (session / "hand_windows" / name).mkdir(parents=True)

    invocations = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="sam3_objects"))

    assert [inv.context["hand_window_id"] for inv in invocations] == [
        "handwin_0000_t000000_000010",
        "handwin_0001_t000020_000030",
    ]
    assert all(inv.node == "sam3_hand_object_segmentation" for inv in invocations)


def test_hand_parquet_export_runs_once_for_parent_session(tmp_path: Path):
    session = tmp_path / "session"
    (session / "hand_windows/handwin_0000_t000000_000010").mkdir(parents=True)
    (session / "hand_windows/handwin_0001_t000020_000030").mkdir(parents=True)

    invocations = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="hand_parquet_export"))

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.node == "hand_parquet_export"
    assert invocation.params["hand_parquet_export.session"] == str(session)


def test_finds_direct_session_hand_windows(tmp_path: Path):
    session = tmp_path / "session"
    (session / "hand_windows/handwin_0000_t000000_000065").mkdir(parents=True)

    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="final_hand_result", hand_window="handwin_0000_t000000_000065")
    )[0]

    assert invocation.context["hand_window"] == str(session / "hand_windows/handwin_0000_t000000_000065")
    assert invocation.params["final_hand_result.output_dir"] == str(
        session / "hand_windows/handwin_0000_t000000_000065/hand/final_hand_result"
    )


def test_prepared_unit_stage_requires_manifest_when_multiple_unprepared_windows_exist(tmp_path: Path):
    session = tmp_path / "session"
    (session / "segments/seg_0000/hand_windows/handwin_0000").mkdir(parents=True)
    (session / "segments/seg_0001/hand_windows/handwin_0001").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No prepared dataset_manifest"):
        build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="hand_filter"))


def test_render_invocation_command_quotes_kedro_params(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "segments/seg_0000/hand_windows/handwin_0000"
    hand_window.mkdir(parents=True)
    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="final_hand_result", hand_window=str(hand_window))
    )[0]

    command = render_invocation_command(invocation)

    assert command.startswith("KEDRO_DISABLE_TELEMETRY=1 PYTHONPATH=.")
    assert "--pipelines egohand" in command
    assert "--nodes final_hand_result" in command
    assert "final_hand_result.output_dir=" in command


def test_dry_run_returns_commands_without_running_subprocess(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "segments/seg_0000/hand_windows/handwin_0000"
    hand_window.mkdir(parents=True)
    invocation = build_stage_invocations(
        LauncherRequest(dataset="raw", session=session, stage="final_hand_result", hand_window=str(hand_window))
    )[0]

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run should not execute subprocess")

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.subprocess.run", fail_run)
    commands = run_stage_invocations([invocation], dry_run=True)

    assert len(commands) == 1
    assert "--nodes final_hand_result" in commands[0]
