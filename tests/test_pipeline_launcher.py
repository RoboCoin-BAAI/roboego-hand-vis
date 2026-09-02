from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboego_hand_vis.pipeline_launcher import (
    LauncherRequest,
    build_stage_invocations,
    load_dataset_config,
    render_invocation_command,
    run_launcher_request,
    run_stage_invocations,
)


def test_load_raw_dataset_config_has_hand_window_stage():
    config = load_dataset_config("raw")

    raw_stage = config["stages"]["raw_quality"]
    assert raw_stage["selector"] == "--to-nodes"
    assert raw_stage["node"] == "raw_headband_quality_gate"
    assert raw_stage["extra_params"]["raw_headband_quality_gate"]["strict"] is False

    stage = config["stages"]["depth_anchored_joints21"]
    assert stage["pipeline"] == "egohand"
    assert stage["node"] == "depth_anchored_joints21"
    assert stage["scope"] == "hand_window"
    assert stage["params"]["manifest"] == "{hand_window}/dataset_manifest/dataset_manifest.json"
    assert Path(config["stages"]["qwen_annotation"]["params"]["prompt"]).is_file()


def test_stage_can_use_to_nodes_selector_for_chained_work(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="post_annotation_actions"))[0]
    command = render_invocation_command(invocation)

    assert invocation.selector == "--to-nodes"
    assert "--pipelines semantic" in command
    assert "--to-nodes stage1_action_text_resolver_chunks" in command
    assert "stage1_action_chunks.qwen_chunk_dir=" in command


def test_qwen_annotation_depends_on_video_simple_quality_report(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="qwen_annotation"))[0]
    command = render_invocation_command(invocation)

    assert invocation.selector == "--to-nodes"
    assert invocation.node == "qwen_frame_annotation"
    assert "raw_headband_quality.overwrite=True" in command
    assert "pre_semantic_quality.output_json=" in command
    assert "video_simple_quality_left.json" in command
    assert "qwen_frame_annotation.use_video_simple_quality=True" in command
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


def test_phase1_planning_workflow_runs_first_six_logical_steps(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    calls = []

    def fake_run(invocations, *, dry_run=False):
        calls.extend(invocation.node for invocation in invocations)
        return [render_invocation_command(invocation) for invocation in invocations]

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.run_stage_invocations", fake_run)

    run_launcher_request(LauncherRequest(dataset="raw", session=session, stage="phase1_planning"))

    assert calls == [
        "qwen_frame_annotation",
        "stage1_action_text_resolver_chunks",
        "build_processing_plan",
        "build_hand_estimation_windows",
    ]


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


def test_hamer_stage_uses_direct_hand_windows_root(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="raw", session=session, stage="hamer"))[0]

    assert invocation.pipeline == "egohand"
    assert invocation.node == "hamer_windows"
    assert invocation.params["hamer_windows.segments_root"] == str(session / "hand_windows")
    assert invocation.params["hamer_windows.output_subdir"] == "hand/hamer_raw"
    assert invocation.params["hamer_windows.wearer_regions_json"] == str(
        session / "annotation/qwen_frame_annotations_left_v4.checkpoint.json"
    )


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


def test_hand_window_stage_requires_window_when_multiple_exist(tmp_path: Path):
    session = tmp_path / "session"
    (session / "segments/seg_0000/hand_windows/handwin_0000").mkdir(parents=True)
    (session / "segments/seg_0001/hand_windows/handwin_0001").mkdir(parents=True)

    with pytest.raises(ValueError, match="multiple hand windows"):
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
