from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


@dataclass(frozen=True)
class VideoPrivacyBlurConfig:
    input_dir: Path
    output_dir: Path
    python: str
    face_weights: Path
    device: str
    conf: float
    imgsz: int
    scale: float
    blur_strength: float
    method: str
    retries: int
    timeout: float | None
    jobs: str
    threads: int
    track_gap: int
    track_scale: float
    selected_videos: tuple[str, ...]
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the video privacy blur stage on one video directory."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。",
    )
    parser.add_argument("--input_dir", required=True, help="输入视频目录。")
    parser.add_argument("--output_dir", required=True, help="隐私模糊结果输出目录。")
    parser.add_argument(
        "--video",
        action="append",
        default=None,
        help="只处理指定的相对视频路径；可重复传入。",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印资源计划，不处理视频。",
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


def config_from_yaml(
    data: dict[str, Any],
    args: argparse.Namespace,
) -> VideoPrivacyBlurConfig:
    runtime = data.get("runtime", {})
    paths = data.get("paths", {})
    parameters = data.get("parameters", {})
    videos = tuple(args.video or parameters.get("videos", []) or ())
    timeout = parameters.get("timeout")

    python = runtime.get("python") or sys.executable

    return VideoPrivacyBlurConfig(
        input_dir=_path(args.input_dir),
        output_dir=_path(args.output_dir),
        python=str(python),
        face_weights=Path(
            paths.get("face_weights", "conf/models/yolov8n-face.pt")
        ),
        device=str(parameters.get("device", "auto")),
        conf=float(parameters.get("conf", 0.35)),
        imgsz=int(parameters.get("imgsz", 960)),
        scale=float(parameters.get("scale", 1.35)),
        blur_strength=float(parameters.get("blur_strength", 1.0)),
        method=str(parameters.get("method", "gaussian")),
        retries=int(parameters.get("retries", 1)),
        timeout=None if timeout is None else float(timeout),
        jobs=str(parameters.get("jobs", "auto")),
        threads=int(parameters.get("threads", 2)),
        track_gap=int(parameters.get("track_gap", 0)),
        track_scale=float(parameters.get("track_scale", 1.25)),
        selected_videos=videos,
        dry_run=bool(args.dry_run or parameters.get("dry_run", False)),
    )


def build_batch_command(config: VideoPrivacyBlurConfig) -> list[str]:
    command = [
        config.python,
        "-m",
        "roboego_hand_vis.video_privacy_blur.batch",
        str(config.input_dir),
        "--output-dir",
        str(config.output_dir),
        "--face-weights",
        str(config.face_weights),
        "--device",
        config.device,
        "--conf",
        str(config.conf),
        "--imgsz",
        str(config.imgsz),
        "--scale",
        str(config.scale),
        "--blur-strength",
        str(config.blur_strength),
        "--method",
        config.method,
        "--retries",
        str(config.retries),
        "--jobs",
        config.jobs,
        "--threads",
        str(config.threads),
        "--track-gap",
        str(config.track_gap),
        "--track-scale",
        str(config.track_scale),
    ]
    if config.timeout is not None:
        command.extend(["--timeout", str(config.timeout)])
    for video in config.selected_videos:
        command.extend(["--video", video])
    if config.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    args = parse_args()
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    env = os.environ.copy()
    project_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(build_batch_command(config), cwd=project_root, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
