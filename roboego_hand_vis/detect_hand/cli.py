from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


STAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = STAGE_DIR / "recommended.yaml"


@dataclass(frozen=True)
class DetectHandConfig:
    video: Path
    model: Path
    output_json: Path | None
    sample_every: int
    conf: float
    device: str
    imgsz: int
    batch_size: int


def process_video(**kwargs: Any) -> dict[str, Any]:
    try:
        from .compute import process_video as _process_video
    except ImportError:
        from compute import process_video as _process_video
    return _process_video(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO 检测视频中的左右手出现比例。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML。")
    parser.add_argument("--video", required=True, help="输入视频路径。")
    parser.add_argument("--model", default="", help="YOLO 模型路径；默认使用 recommended.yaml。")
    parser.add_argument("--output_json", default="", help="输出 JSON；默认写到 session/annotation/detect_hand_<video>.json。")
    parser.add_argument("--sample_every", type=int, default=None, help="每隔多少帧抽样检测。")
    parser.add_argument("--conf", type=float, default=None, help="YOLO confidence 阈值。")
    parser.add_argument("--device", default="", help="推理设备，例如 0 或 cpu。")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO 输入尺寸。")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小。")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _model_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return STAGE_DIR / path


def default_output_json_for_video(video: Path) -> Path:
    video = Path(video)
    if video.parent.name == "videos":
        return video.parent.parent / "annotation" / f"detect_hand_{video.stem}.json"
    return video.parent / f"detect_hand_{video.stem}.json"


def config_from_yaml(data: dict[str, Any], args: argparse.Namespace) -> DetectHandConfig:
    paths = data.get("paths", {})
    parameters = data.get("parameters", {})
    model_value = args.model or paths.get("model", "detector.pt")
    video = _path(args.video)
    output_json = _path(args.output_json) if args.output_json else default_output_json_for_video(video).resolve()
    return DetectHandConfig(
        video=video,
        model=_model_path(model_value),
        output_json=output_json,
        sample_every=int(args.sample_every if args.sample_every is not None else parameters.get("sample_every", 5)),
        conf=float(args.conf if args.conf is not None else parameters.get("conf", 0.15)),
        device=str(args.device or parameters.get("device", "0")),
        imgsz=int(args.imgsz if args.imgsz is not None else parameters.get("imgsz", 640)),
        batch_size=int(args.batch_size if args.batch_size is not None else parameters.get("batch_size", 32)),
    )


def run_detect_hand_stage(config: DetectHandConfig) -> dict[str, Any]:
    output = process_video(
        video_path=config.video,
        model_path=config.model,
        output_json=config.output_json,
        sample_every=config.sample_every,
        conf=config.conf,
        device=config.device,
        imgsz=config.imgsz,
        batch_size=config.batch_size,
    )
    return {
        "stage": "detect_hand",
        "output_json": str(config.output_json) if config.output_json else None,
        "sample_count": output.get("sample_count"),
        "hand_presence_percent": output.get("hand_presence_percent"),
        "left_hand_presence_percent": output.get("left_hand_presence_percent"),
        "right_hand_presence_percent": output.get("right_hand_presence_percent"),
    }


def main() -> None:
    args = parse_args()
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    result = run_detect_hand_stage(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
