from pathlib import Path

import cv2
import numpy as np

from roboego_hand_vis.quality.pre_semantic_gate import (
    PreSemanticGateConfig,
    analyze_sampled_frames,
    frame_gate_label,
    run_pre_semantic_quality_gate,
)


def textured_frame(value: int = 80, width: int = 64, height: int = 48) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    frame[:, :, 1] = (np.arange(width, dtype=np.uint8)[None, :] * 3) % 255
    frame[:, :, 2] = (np.arange(height, dtype=np.uint8)[:, None] * 5) % 255
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: float = 1.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_frame_gate_label_marks_black_overexposed_and_blocked_as_drop_candidates():
    config = PreSemanticGateConfig()

    black = frame_gate_label(np.zeros((32, 32, 3), dtype=np.uint8), config)
    overexposed = frame_gate_label(np.full((32, 32, 3), 255, dtype=np.uint8), config)
    blocked = frame_gate_label(np.full((32, 32, 3), 80, dtype=np.uint8), config)

    assert black["label"] == "black_screen"
    assert black["route_hint"] == "drop_before_qwen"
    assert overexposed["label"] == "overexposed_screen"
    assert overexposed["route_hint"] == "drop_before_qwen"
    assert blocked["label"] == "camera_fully_blocked"
    assert blocked["route_hint"] == "drop_before_qwen"


def test_analyze_sampled_frames_marks_long_freeze_as_drop_before_qwen():
    frame = textured_frame()
    samples = analyze_sampled_frames(
        [frame.copy() for _ in range(6)],
        times_s=[0, 1, 2, 3, 4, 5],
        config=PreSemanticGateConfig(min_freeze_duration_s=5),
    )

    assert samples[-1]["label"] == "frozen_video"
    assert samples[-1]["route_hint"] == "drop_before_qwen"
    assert samples[-1]["aux"]["low_motion"] is True


def test_analyze_sampled_frames_keeps_short_low_motion_static_as_aux_signal():
    frames = [textured_frame(80), textured_frame(81), textured_frame(80)]
    samples = analyze_sampled_frames(
        frames,
        times_s=[0, 1, 2],
        config=PreSemanticGateConfig(min_freeze_duration_s=5),
    )

    assert [sample["route_hint"] for sample in samples] == ["pass_to_qwen", "pass_to_qwen_with_aux", "pass_to_qwen_with_aux"]
    assert samples[1]["label"] == "low_motion_static"
    assert samples[1]["aux"]["low_motion"] is True


def test_run_pre_semantic_quality_gate_writes_report(tmp_path: Path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "quality_gate.json"
    frames = [textured_frame(90 + idx) for idx in range(3)]
    write_video(video, frames)

    report = run_pre_semantic_quality_gate(
        video_path=video,
        output_json=output,
        config=PreSemanticGateConfig(sample_every_s=1.0, min_freeze_duration_s=5),
    )

    assert report["stage"] == "video_simple_quality_gate"
    assert report["checks"]["decode_ok"] is True
    assert report["checks"]["sampled_frame_count"] == 3
    assert report["passed"] is True
    assert report["status"] == "passed"
    assert report["intervals"][0]["route_hint"] in {"pass_to_qwen", "pass_to_qwen_with_aux"}
    assert output.exists()


def test_run_pre_semantic_quality_gate_fails_when_drop_interval_exists(tmp_path: Path):
    video = tmp_path / "input.mp4"
    output = tmp_path / "quality_gate.json"
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    write_video(video, frames)

    report = run_pre_semantic_quality_gate(
        video_path=video,
        output_json=output,
        config=PreSemanticGateConfig(sample_every_s=1.0),
    )

    assert report["passed"] is False
    assert report["status"] == "failed"
    assert report["failed_interval_count"] >= 1
    assert "black_screen" in {interval["label"] for interval in report["failed_intervals"]}
