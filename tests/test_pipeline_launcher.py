from __future__ import annotations

from pathlib import Path

import pytest

from roboego_hand_vis.pipeline_launcher import (
    LauncherRequest,
    build_stage_invocations,
    load_dataset_config,
    render_invocation_command,
    run_stage_invocations,
)


def test_load_wumei_dataset_config_has_hand_window_stage():
    config = load_dataset_config("wumei")

    stage = config["stages"]["depth_anchored_joints21"]
    assert stage["pipeline"] == "egohand"
    assert stage["node"] == "depth_anchored_joints21"
    assert stage["scope"] == "hand_window"
    assert stage["params"]["manifest"] == "{hand_window}/dataset_manifest/dataset_manifest.json"
    assert Path(config["stages"]["qwen_annotation"]["params"]["prompt"]).is_file()


def test_stage_can_use_to_nodes_selector_for_chained_work(tmp_path: Path):
    session = tmp_path / "session"

    invocation = build_stage_invocations(LauncherRequest(dataset="wumei", session=session, stage="stage1_actions"))[0]
    command = render_invocation_command(invocation)

    assert invocation.selector == "--to-nodes"
    assert "--pipelines semantic" in command
    assert "--to-nodes stage1_action_text_resolver_chunks" in command
    assert "stage1_action_chunks.qwen_chunk_dir=" in command


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
            dataset="wumei",
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
        LauncherRequest(dataset="wumei", session=session, stage="joints21_pose_gate", hand_window="handwin_0000")
    )[0]

    assert invocation.params["joints21_pose_gate.num_frames"] == "1980"
    assert invocation.params["joints21_pose_gate.image_height"] == "1050"


def test_all_hand_windows_builds_one_invocation_per_window(tmp_path: Path):
    session = tmp_path / "session"
    for name in ("handwin_0000_t000000_000010", "handwin_0001_t000020_000030"):
        (session / "segments/seg_0000/hand_windows" / name).mkdir(parents=True)

    invocations = build_stage_invocations(
        LauncherRequest(dataset="wumei", session=session, stage="final_hand_result", all_hand_windows=True)
    )

    assert [inv.context["hand_window_id"] for inv in invocations] == [
        "handwin_0000_t000000_000010",
        "handwin_0001_t000020_000030",
    ]
    assert all(inv.node == "final_hand_result" for inv in invocations)


def test_hand_window_stage_requires_window_when_multiple_exist(tmp_path: Path):
    session = tmp_path / "session"
    (session / "segments/seg_0000/hand_windows/handwin_0000").mkdir(parents=True)
    (session / "segments/seg_0001/hand_windows/handwin_0001").mkdir(parents=True)

    with pytest.raises(ValueError, match="multiple hand windows"):
        build_stage_invocations(LauncherRequest(dataset="wumei", session=session, stage="hand_filter"))


def test_render_invocation_command_quotes_kedro_params(tmp_path: Path):
    session = tmp_path / "session"
    hand_window = session / "segments/seg_0000/hand_windows/handwin_0000"
    hand_window.mkdir(parents=True)
    invocation = build_stage_invocations(
        LauncherRequest(dataset="wumei", session=session, stage="final_hand_result", hand_window=str(hand_window))
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
        LauncherRequest(dataset="wumei", session=session, stage="final_hand_result", hand_window=str(hand_window))
    )[0]

    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run should not execute subprocess")

    monkeypatch.setattr("roboego_hand_vis.pipeline_launcher.subprocess.run", fail_run)
    commands = run_stage_invocations([invocation], dry_run=True)

    assert len(commands) == 1
    assert "--nodes final_hand_result" in commands[0]
