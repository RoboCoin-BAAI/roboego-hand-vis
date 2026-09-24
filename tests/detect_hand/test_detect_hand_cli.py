from __future__ import annotations

from pathlib import Path

from roboego_hand_vis.detect_hand import cli


def test_config_from_yaml_resolves_default_model_next_to_stage(tmp_path: Path) -> None:
    video = tmp_path / "left.mkv"
    output_json = tmp_path / "hand_detect.json"
    args = cli.parse_args(["--video", str(video), "--output_json", str(output_json)])

    config = cli.config_from_yaml(
        {
            "paths": {"model": "detector.pt"},
            "parameters": {
                "sample_every": 5,
                "conf": 0.15,
                "device": "cpu",
                "imgsz": 640,
                "batch_size": 32,
            },
        },
        args,
    )

    assert config.video == video.resolve()
    assert config.output_json == output_json.resolve()
    assert config.model == Path(cli.__file__).with_name("detector.pt")
    assert config.device == "cpu"


def test_config_from_yaml_defaults_output_json_next_to_session_annotation(tmp_path: Path) -> None:
    video = tmp_path / "session" / "videos" / "left.mkv"
    video.parent.mkdir(parents=True)
    args = cli.parse_args(["--video", str(video)])

    config = cli.config_from_yaml({"paths": {"model": "detector.pt"}, "parameters": {}}, args)

    assert config.output_json == tmp_path / "session" / "annotation" / "detect_hand_left.json"


def test_run_detect_hand_stage_returns_summary(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "left.mkv"
    model = tmp_path / "detector.pt"
    output_json = tmp_path / "hand_detect.json"
    config = cli.DetectHandConfig(
        video=video,
        model=model,
        output_json=output_json,
        sample_every=5,
        conf=0.15,
        device="cpu",
        imgsz=640,
        batch_size=32,
    )
    calls = []

    def fake_process_video(**kwargs):
        calls.append(kwargs)
        return {
            "video": str(video),
            "model": str(model),
            "sample_count": 10,
            "hand_presence_percent": 80.0,
            "left_hand_presence_percent": 70.0,
            "right_hand_presence_percent": 60.0,
        }

    monkeypatch.setattr(cli, "process_video", fake_process_video)

    summary = cli.run_detect_hand_stage(config)

    assert calls[0]["video_path"] == video
    assert calls[0]["model_path"] == model
    assert calls[0]["output_json"] == output_json
    assert summary == {
        "stage": "detect_hand",
        "output_json": str(output_json),
        "sample_count": 10,
        "hand_presence_percent": 80.0,
        "left_hand_presence_percent": 70.0,
        "right_hand_presence_percent": 60.0,
    }
