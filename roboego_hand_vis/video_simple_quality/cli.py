from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

try:
    from .compute import PreSemanticGateConfig, run_pre_semantic_quality_gate
except ImportError:
    from compute import PreSemanticGateConfig, run_pre_semantic_quality_gate


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


@dataclass(frozen=True)
class VideoSimpleQualityConfig:
    session: Path | None
    video: Path | None
    output_json: Path
    streams: tuple[str, ...]
    video_ext: str
    gate: PreSemanticGateConfig


def default_output_json_for_video(video_path: Path) -> Path:
    video_path = Path(video_path)
    if video_path.parent.name == "videos":
        return video_path.parent.parent / "annotation" / f"video_simple_quality_{video_path.stem}.json"
    return video_path.parent / f"video_simple_quality_{video_path.stem}.json"


def default_output_json_for_session(session: Path) -> Path:
    return Path(session) / "annotation" / "video_simple_quality.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单视频粗质检，输出 video_simple_quality JSON。")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", default="", help="Raw session 目录；默认检查 videos/left.mkv 和 videos/right.mkv。")
    source.add_argument("--video", default="", help="输入单个视频路径；保留旧的单视频用法。")
    parser.add_argument(
        "--output_json",
        default="",
        help="输出报告路径；session 模式默认写到 session/annotation/video_simple_quality.json。",
    )
    return parser.parse_args(argv)


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
    session = _path(args.session) if args.session else None
    video = _path(args.video) if args.video else None
    if session is not None:
        output_json = _path(args.output_json) if args.output_json else default_output_json_for_session(session).resolve()
    elif video is not None:
        output_json = _path(args.output_json) if args.output_json else default_output_json_for_video(video).resolve()
    else:
        raise ValueError("Either --session or --video is required.")

    return VideoSimpleQualityConfig(
        session=session,
        video=video,
        output_json=output_json,
        streams=tuple(parameters.get("camera_streams", ("left", "right"))),
        video_ext=str(parameters.get("video_ext", ".mkv")),
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
    if config.session is not None:
        return run_session_video_simple_quality(config)
    if config.video is None:
        raise ValueError("Either session or video must be configured.")
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


def run_session_video_simple_quality(config: VideoSimpleQualityConfig) -> dict[str, Any]:
    if config.session is None:
        raise ValueError("Session mode requires config.session.")

    stream_reports: dict[str, Any] = {}
    failed_streams: list[str] = []
    drop_interval_count = 0
    aux_interval_count = 0
    sampled_frame_count = 0

    for stream in config.streams:
        video_path = config.session / "videos" / f"{stream}{config.video_ext}"
        report = run_pre_semantic_quality_gate(
            video_path=video_path,
            output_json=config.output_json,
            config=config.gate,
        )
        intervals = list(report.get("intervals") or [])
        stream_drop_count = sum(1 for item in intervals if item.get("route_hint") == "drop_before_qwen")
        stream_aux_count = sum(1 for item in intervals if item.get("route_hint") == "pass_to_qwen_with_aux")
        checks = report.get("checks") or {}
        drop_interval_count += stream_drop_count
        aux_interval_count += stream_aux_count
        sampled_frame_count += int(checks.get("sampled_frame_count") or 0)
        if (not bool(report.get("passed"))) or stream_drop_count:
            failed_streams.append(stream)
        stream_reports[stream] = report

    passed = len(failed_streams) == 0
    aggregate = {
        "stage": "video_simple_quality",
        "mode": "session",
        "source_session": str(config.session),
        "passed": passed,
        "status": "passed" if passed else "failed",
        "can_send_all_to_qwen": passed,
        "stream_count": len(config.streams),
        "streams_checked": list(config.streams),
        "failed_streams": failed_streams,
        "drop_interval_count": drop_interval_count,
        "aux_interval_count": aux_interval_count,
        "sampled_frame_count": sampled_frame_count,
        "streams": stream_reports,
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "stage": "video_simple_quality",
        "mode": "session",
        "output_json": str(config.output_json),
        "passed": passed,
        "status": aggregate["status"],
        "can_send_all_to_qwen": passed,
        "stream_count": len(config.streams),
        "failed_streams": failed_streams,
        "drop_interval_count": drop_interval_count,
        "aux_interval_count": aux_interval_count,
        "sampled_frame_count": sampled_frame_count,
    }


def main() -> None:
    args = parse_args()
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    result = run_video_simple_quality(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
