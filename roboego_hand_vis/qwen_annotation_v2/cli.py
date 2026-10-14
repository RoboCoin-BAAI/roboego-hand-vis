from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import QwenAnnotationConfig, run_qwen_annotation
except ImportError:
    from compute import QwenAnnotationConfig, run_qwen_annotation


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行独立 V2 千问时序标注实验。")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。",
    )
    parser.add_argument("--video", required=True, help="输入视频路径。")
    parser.add_argument("--output_json", required=True, help="Qwen 标注输出 JSON。")
    parser.add_argument('--image-mode', choices=('individual', 'grid'), default=None)
    parser.add_argument('--max-concurrency', type=int, default=None)
    parser.add_argument('--overlap-frames', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
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


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _path(text)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def config_from_yaml(
    data: dict[str, Any],
    *,
    video: Path,
    output_json: Path,
) -> QwenAnnotationConfig:
    parameters = data.get("parameters", {})
    output_json = _path(output_json)

    return QwenAnnotationConfig(
        video=_path(video),
        output_json=output_json,
        frames_dir=_path(parameters.get("frames_dir") or output_json.parent / "qwen_frame_annotation_frames"),
        prompt=_path(parameters.get("prompt", "docs/prompts/qwen_frame_annotation_v5_en.md")),
        model=str(parameters.get("model", "qwen3.8-27b")),
        fps=float(parameters.get("fps", 1.0)),
        start_s=float(parameters.get("start_s", 0.0)),
        duration_s=_optional_float(parameters.get("duration_s")),
        windows_json=_optional_path(parameters.get("windows_json")),
        max_width=int(parameters.get("max_width", 768)),
        batch_size=int(parameters.get("batch_size", 10)),
        api_key_env=str(parameters.get("api_key_env", "QWEN_API_KEY")),
        base_url=str(parameters.get("base_url", "")),
        chat_template_kwargs=dict(parameters.get("chat_template_kwargs") or {}),
        timeout_s=float(parameters.get("timeout_s", 120.0)),
        max_retries=int(parameters.get("max_retries", 3)),
        retry_sleep_s=float(parameters.get("retry_sleep_s", 5.0)),
        raw_output_dir=_path(parameters.get("raw_output_dir") or output_json.parent / "qwen_frame_annotation_raw"),
        checkpoint_json=_path(
            parameters.get("checkpoint_json") or output_json.with_suffix(output_json.suffix + ".checkpoint.json")
        ),
        batch_output_dir=_path(parameters.get("batch_output_dir") or output_json.parent / "qwen_frame_annotation_batches"),
        chunk_output_dir=_path(parameters.get("chunk_output_dir") or output_json.parent / "qwen_frame_annotation_chunks"),
        batches_per_chunk=int(parameters.get("batches_per_chunk", 5)),
        resume=bool(parameters.get("resume", True)),
        recover_from_raw=bool(parameters.get("recover_from_raw", False)),
        overlap_frames=int(parameters.get("overlap_frames", 2)),
        max_concurrency=int(parameters.get("max_concurrency", 3)),
        image_mode=str(parameters.get("image_mode", "individual")),
        grid_columns=int(parameters.get("grid_columns", 4)),
        grid_cell_width=int(parameters.get("grid_cell_width", 512)),
    )


def main() -> None:
    args = parse_args()
    data = load_yaml(Path(args.config).expanduser())
    parameters = data.setdefault('parameters', {})
    for name in ('image_mode', 'max_concurrency', 'overlap_frames', 'batch_size'):
        value = getattr(args, name)
        if value is not None:
            parameters[name] = value
    config = config_from_yaml(
        data,
        video=Path(args.video),
        output_json=Path(args.output_json),
    )
    result = run_qwen_annotation(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "complete":
        raise SystemExit(1)
