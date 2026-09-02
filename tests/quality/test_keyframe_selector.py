from pathlib import Path

import cv2
import numpy as np

from tests.manual.qwen_keyframe_selector_lib import (
    KeyframeSelectorConfig,
    select_qwen_keyframes,
)


def textured_frame(value: int = 80, width: int = 64, height: int = 48) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    frame[:, :, 1] = (np.arange(width, dtype=np.uint8)[None, :] * 3) % 255
    frame[:, :, 2] = (np.arange(height, dtype=np.uint8)[:, None] * 5) % 255
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: float = 2.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_select_qwen_keyframes_combines_base_and_visual_change_reasons(tmp_path: Path):
    video = tmp_path / "input.mp4"
    frames = [
        textured_frame(60),
        textured_frame(61),
        textured_frame(62),
        textured_frame(210),
        textured_frame(211),
        textured_frame(212),
    ]
    write_video(video, frames, fps=2.0)
    quality_json = tmp_path / "video_simple_quality_left.json"
    quality_json.write_text(
        """
{
  "intervals": [
    {"start": 0.0, "end": 3.0, "route_hint": "pass_to_qwen"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = select_qwen_keyframes(
        video_path=video,
        video_simple_quality_json=quality_json,
        output_json=tmp_path / "keyframes.json",
        config=KeyframeSelectorConfig(
            analysis_fps=2.0,
            base_interval_s=2.0,
            boundary_margin_s=0.0,
            min_keyframe_gap_s=0.4,
            visual_change_quantile=0.5,
            max_keyframes_per_minute=120,
            max_width=64,
        ),
    )

    keyframes = report["keyframes"]
    assert [item["t"] for item in keyframes] == [0.0, 1.5, 2.0]
    assert keyframes[0]["reasons"] == ["interval_start", "base_interval_sample"]
    assert "visual_change_peak" in keyframes[1]["reasons"]
    assert keyframes[2]["reasons"] == ["base_interval_sample"]


def test_select_qwen_keyframes_uses_only_video_simple_quality_valid_intervals(tmp_path: Path):
    video = tmp_path / "input.mp4"
    write_video(video, [textured_frame(80 + idx) for idx in range(8)], fps=2.0)
    quality_json = tmp_path / "video_simple_quality_left.json"
    quality_json.write_text(
        """
{
  "intervals": [
    {"start": 0.0, "end": 1.0, "route_hint": "drop_before_qwen"},
    {"start": 1.0, "end": 3.0, "route_hint": "pass_to_qwen"}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = select_qwen_keyframes(
        video_path=video,
        video_simple_quality_json=quality_json,
        output_json=tmp_path / "keyframes.json",
        config=KeyframeSelectorConfig(
            analysis_fps=2.0,
            base_interval_s=1.0,
            boundary_margin_s=0.0,
            min_keyframe_gap_s=0.1,
            visual_change_quantile=1.1,
            max_keyframes_per_minute=120,
            max_width=64,
        ),
    )

    assert [item["t"] for item in report["keyframes"]] == [1.0, 2.0]
    assert report["source_intervals"] == [{"start": 1.0, "end": 3.0, "route_hint": "pass_to_qwen"}]
