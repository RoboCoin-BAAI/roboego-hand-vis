from __future__ import annotations

import json
from pathlib import Path

from roboego_hand_vis.video_simple_quality import cli


def _write_placeholder_videos(session: Path) -> None:
    videos = session / "videos"
    videos.mkdir(parents=True)
    (videos / "left.mkv").write_bytes(b"left")
    (videos / "right.mkv").write_bytes(b"right")


def test_session_mode_runs_default_left_right_and_writes_aggregate(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session"
    _write_placeholder_videos(session)
    seen: list[tuple[Path, Path]] = []

    def fake_run_pre_semantic_quality_gate(*, video_path, output_json, config):
        seen.append((Path(video_path), Path(output_json)))
        return {
            "source_video": str(video_path),
            "passed": True,
            "status": "passed",
            "checks": {"decode_ok": True, "sampled_frame_count": 3},
            "intervals": [{"route_hint": "pass_to_qwen"}],
        }

    monkeypatch.setattr(cli, "run_pre_semantic_quality_gate", fake_run_pre_semantic_quality_gate)
    args = cli.parse_args(["--session", str(session)])
    config = cli.config_from_yaml({"parameters": {}}, args)

    summary = cli.run_video_simple_quality(config)

    assert summary["passed"] is True
    assert summary["output_json"] == str(session / "annotation" / "video_simple_quality.json")
    assert [item[0].name for item in seen] == ["left.mkv", "right.mkv"]

    aggregate = json.loads((session / "annotation" / "video_simple_quality.json").read_text(encoding="utf-8"))
    assert aggregate["passed"] is True
    assert aggregate["status"] == "passed"
    assert aggregate["stream_count"] == 2
    assert aggregate["drop_interval_count"] == 0
    assert sorted(aggregate["streams"]) == ["left", "right"]


def test_session_mode_fails_when_any_stream_has_drop_interval(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "session"
    _write_placeholder_videos(session)

    def fake_run_pre_semantic_quality_gate(*, video_path, output_json, config):
        is_left = Path(video_path).stem == "left"
        return {
            "source_video": str(video_path),
            "passed": is_left,
            "status": "passed" if is_left else "failed",
            "checks": {"decode_ok": True, "sampled_frame_count": 3},
            "intervals": [
                {"route_hint": "pass_to_qwen"}
                if is_left
                else {"route_hint": "drop_before_qwen", "label": "black_screen"}
            ],
        }

    monkeypatch.setattr(cli, "run_pre_semantic_quality_gate", fake_run_pre_semantic_quality_gate)
    args = cli.parse_args(["--session", str(session)])
    config = cli.config_from_yaml({"parameters": {}}, args)

    summary = cli.run_video_simple_quality(config)

    assert summary["passed"] is False
    assert summary["drop_interval_count"] == 1
    aggregate = json.loads((session / "annotation" / "video_simple_quality.json").read_text(encoding="utf-8"))
    assert aggregate["passed"] is False
    assert aggregate["failed_streams"] == ["right"]
