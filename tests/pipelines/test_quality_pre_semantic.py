from pathlib import Path

import pytest

from roboego_hand_vis.pipelines.quality.nodes import run_pre_semantic_quality_gate_node


def test_pre_semantic_quality_gate_node_skips_when_disabled():
    report, windows = run_pre_semantic_quality_gate_node(
        {"status": "skipped"},
        {"enabled": False},
    )

    assert report == {
        "stage": "pre_semantic_quality_gate",
        "status": "skipped",
        "reason": "disabled",
    }
    assert windows == []


def test_pre_semantic_quality_gate_node_requires_video_when_enabled():
    with pytest.raises(ValueError, match="video"):
        run_pre_semantic_quality_gate_node({"passed": True}, {"enabled": True})


def test_pre_semantic_quality_gate_node_skips_when_raw_gate_did_not_pass():
    report, windows = run_pre_semantic_quality_gate_node(
        {"passed": False, "status": "failed"},
        {"enabled": True, "video": "/tmp/left.mp4"},
    )

    assert report == {
        "stage": "pre_semantic_quality_gate",
        "status": "skipped",
        "reason": "raw_headband_quality_gate_not_passed",
    }
    assert windows == []


def test_pre_semantic_quality_gate_node_returns_valid_windows(monkeypatch, tmp_path):
    calls = []

    def fake_run_pre_semantic_quality_gate(*, video_path, output_json, config):
        calls.append(
            {
                "video_path": video_path,
                "output_json": output_json,
                "sample_every_s": config.sample_every_s,
                "min_freeze_duration_s": config.min_freeze_duration_s,
            }
        )
        return {
            "stage": "pre_semantic_quality_gate",
            "source_video": str(video_path),
            "checks": {"decode_ok": True, "sampled_frame_count": 4},
            "intervals": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "label": "valid_visual_content",
                    "route_hint": "pass_to_qwen",
                    "confidence": 0.98,
                },
                {
                    "start": 5.0,
                    "end": 9.0,
                    "label": "black_screen",
                    "route_hint": "drop_before_qwen",
                    "confidence": 0.99,
                },
                {
                    "start": 9.0,
                    "end": 12.0,
                    "label": "low_motion_static",
                    "route_hint": "pass_to_qwen_with_aux",
                    "confidence": 0.9,
                },
            ],
            "samples": [],
        }

    monkeypatch.setattr(
        "roboego_hand_vis.pipelines.quality.nodes.run_pre_semantic_quality_gate",
        fake_run_pre_semantic_quality_gate,
    )

    video = tmp_path / "session" / "videos" / "left.mp4"
    report, windows = run_pre_semantic_quality_gate_node(
        {"passed": True},
        {
            "enabled": True,
            "video": str(video),
            "output_json": "",
            "sample_every_s": 2.0,
            "min_freeze_duration_s": 8.0,
        }
    )

    assert calls == [
        {
            "video_path": video,
            "output_json": tmp_path / "session" / "annotation" / "pre_semantic_quality_gate_left.json",
            "sample_every_s": 2.0,
            "min_freeze_duration_s": 8.0,
        }
    ]
    assert report["status"] == "complete"
    assert report["output_json"].endswith("annotation/pre_semantic_quality_gate_left.json")
    assert windows == [
        {
            "source_video": str(video),
            "start": 0.0,
            "end": 5.0,
            "label": "valid_visual_content",
            "route_hint": "pass_to_qwen",
            "confidence": 0.98,
        },
        {
            "source_video": str(video),
            "start": 9.0,
            "end": 12.0,
            "label": "low_motion_static",
            "route_hint": "pass_to_qwen_with_aux",
            "confidence": 0.9,
        },
    ]
