from pathlib import Path

from tests.manual.select_qwen_keyframes import parse_args


def test_select_qwen_keyframes_cli_defaults():
    args = parse_args(
        [
            "--video",
            "left.mp4",
            "--video_simple_quality_json",
            "video_simple_quality_left.json",
            "--output_json",
            "qwen_keyframes_left_v1.json",
        ]
    )

    assert Path(args.video) == Path("left.mp4")
    assert Path(args.video_simple_quality_json) == Path("video_simple_quality_left.json")
    assert Path(args.output_json) == Path("qwen_keyframes_left_v1.json")
    assert args.analysis_fps == 2.0
    assert args.base_interval_s == 5.0
    assert args.max_keyframes_per_minute == 30
