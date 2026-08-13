from pathlib import Path

from scripts.quality.run_pre_semantic_quality_gate import default_output_json_for_video


def test_default_output_json_for_video_uses_session_annotation_dir():
    video = Path("data/session_20260804_223748/videos/left.mp4")

    output = default_output_json_for_video(video)

    assert output == Path("data/session_20260804_223748/annotation/pre_semantic_quality_gate_left.json")


def test_default_output_json_for_video_uses_video_parent_for_non_session_layout():
    video = Path("tmp/input.mp4")

    output = default_output_json_for_video(video)

    assert output == Path("tmp/pre_semantic_quality_gate_input.json")
