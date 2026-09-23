from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import PreSemanticGateConfig, run_pre_semantic_quality_gate
except ImportError:
    from compute import PreSemanticGateConfig, run_pre_semantic_quality_gate


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


@dataclass(frozen=True)
class VideoSimpleQualityConfig:
    video: Path
    output_json: Path
    gate: PreSemanticGateConfig


def default_output_json_for_video(video_path: Path) -> Path:
    video_path = Path(video_path)
    if video_path.parent.name == "videos":
        return video_path.parent.parent / "annotation" / f"video_simple_quality_{video_path.stem}.json"
    return video_path.parent / f"video_simple_quality_{video_path.stem}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单视频粗质检，输出 video_simple_quality JSON。")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。",
    )
    parser.add_argument("--video", required=True, help="输入视频路径。")
    parser.add_argument(
        "--output_json",
        default="",
        help="输出报告路径；默认写到 session/annotation/video_simple_quality_<camera>.json。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def config_from_yaml(data: dict[str, Any], args: argparse.Namespace) -> VideoSimpleQualityConfig:
    parameters = data.get("parameters", {})
    video = _path(args.video)
    output_json = _path(args.output_json) if args.output_json else default_output_json_for_video(video).resolve()

    return VideoSimpleQualityConfig(
        video=video,
        output_json=output_json,
        gate=PreSemanticGateConfig(
            sample_every_s=float(parameters.get("sample_every_s", 1.0)),
            dark_threshold=int(parameters.get("dark_threshold", 10)),
            bright_threshold=int(parameters.get("bright_threshold", 245)),
            max_black_ratio=float(parameters.get("max_black_ratio", 0.98)),
            max_overexposed_ratio=float(parameters.get("max_overexposed_ratio", 0.98)),
            min_texture_std=float(parameters.get("min_texture_std", 3.0)),
            freeze_absdiff_threshold=float(parameters.get("freeze_absdiff_threshold", 1.5)),
            low_motion_absdiff_threshold=float(parameters.get("low_motion_absdiff_threshold", 4.0)),
            min_freeze_duration_s=float(parameters.get("min_freeze_duration_s", 5.0)),
            min_bad_visual_duration_s=float(parameters.get("min_bad_visual_duration_s", 3.0)),
        ),
    )


def run_video_simple_quality(config: VideoSimpleQualityConfig) -> dict[str, Any]:
    report = run_pre_semantic_quality_gate(
        video_path=config.video,
        output_json=config.output_json,
        config=config.gate,
    )
    return {
        "stage": "video_simple_quality",
        "output_json": str(config.output_json),
        "decode_ok": report["checks"]["decode_ok"],
        "sampled_frame_count": report["checks"]["sampled_frame_count"],
        "drop_interval_count": sum(
            1 for item in report["intervals"] if item["route_hint"] == "drop_before_qwen"
        ),
        "aux_interval_count": sum(
            1 for item in report["intervals"] if item["route_hint"] == "pass_to_qwen_with_aux"
        ),
    }


def main() -> None:
    args = parse_args()
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    result = run_video_simple_quality(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
