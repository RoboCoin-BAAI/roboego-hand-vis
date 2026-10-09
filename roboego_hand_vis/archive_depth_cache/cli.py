from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import DepthArchiveConfig, archive_depth_cache
except ImportError:
    from compute import DepthArchiveConfig, archive_depth_cache


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将逐帧深度缓存归档为 uint16 PNG、FFV1、HEVC12 等格式。"
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="归档参数 YAML；默认使用 archive_depth_cache/default.yaml。",
    )

    parser.add_argument(
        "--depth_dir",
        required=True,
        help="深度结果目录。",
    )

    parser.add_argument(
        "--output_dir",
        default="",
        help="归档输出目录；默认与 depth_dir 相同。",
    )
    parser.add_argument("--max_depth_m", type=float, default=None)
    parser.add_argument("--invalid_value", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--ffmpeg_bin", default=None)
    parser.add_argument("--write_png", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write_avi", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write_preview", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preview_mode", choices=["near", "full"], default=None)
    parser.add_argument("--preview_near_min_m", type=float, default=None)
    parser.add_argument("--preview_near_max_m", type=float, default=None)
    parser.add_argument("--preview_contrast_percentile", type=float, default=None)
    parser.add_argument("--preview_codec", default=None)
    parser.add_argument("--ffv1_container", choices=["avi", "mkv"], default=None)
    parser.add_argument("--dry_run_avi", action="store_true", default=None)
    parser.add_argument("--delete_npy", action="store_true", default=None)
    parser.add_argument("--delete_png", action="store_true", default=None)
    parser.add_argument("--write_hevc12", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hevc12_crf", type=int, default=None)
    parser.add_argument("--hevc12_preset", default=None)
    parser.add_argument("--hevc12_lossless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--write_valid_mask", action=argparse.BooleanOptionalAction, default=None)

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")

    return config


def config_from_yaml(data: dict[str, Any]) -> DepthArchiveConfig:
    parameters = data.get("parameters", {})
    paths = data.get("paths", {})

    contrast = parameters.get(
        "preview_contrast_percentile",
        2.0,
    )

    if contrast is not None:
        contrast = float(contrast)

    return DepthArchiveConfig(
        max_depth_m=float(
            parameters.get("max_depth_m", 4.0)
        ),
        invalid_value=int(
            parameters.get("invalid_value", 0)
        ),
        fps=float(
            parameters.get("fps", 30.0)
        ),

        write_png=bool(
            parameters.get("write_png", True)
        ),
        write_avi=bool(
            parameters.get("write_avi", False)
        ),
        write_preview=bool(
            parameters.get("write_preview", False)
        ),

        preview_mode=str(
            parameters.get("preview_mode", "near")
        ),
        preview_near_min_m=float(
            parameters.get("preview_near_min_m", 0.15)
        ),
        preview_near_max_m=float(
            parameters.get("preview_near_max_m", 3.0)
        ),
        preview_contrast_percentile=contrast,
        preview_codec=str(
            parameters.get("preview_codec", "h264")
        ),

        ffv1_container=str(
            parameters.get("ffv1_container", "avi")
        ),
        dry_run_avi=bool(
            parameters.get("dry_run_avi", False)
        ),

        delete_npy=bool(
            parameters.get("delete_npy", False)
        ),
        delete_png=bool(
            parameters.get("delete_png", False)
        ),

        ffmpeg_bin=str(
            paths.get("ffmpeg_bin", "ffmpeg")
        ),

        write_hevc12=bool(
            parameters.get("write_hevc12", False)
        ),
        hevc12_crf=int(
            parameters.get("hevc12_crf", 0)
        ),
        hevc12_preset=str(
            parameters.get("hevc12_preset", "slow")
        ),
        hevc12_lossless=bool(
            parameters.get("hevc12_lossless", True)
        ),

        write_valid_mask=bool(
            parameters.get("write_valid_mask", False)
        ),
    )


def summarize_report_for_cli(report: dict) -> dict:
    outputs = report.get("outputs", {})
    compression = report.get("compression", {})

    return {
        "stage": report.get("stage"),
        "frame_count": report.get("frame_count"),
        "source_depth_format": report.get("source_depth_format"),
        "output_dir": report.get("output_dir"),
        "hevc12_video": outputs.get("hevc12_video"),
        "valid_mask_video": outputs.get("valid_mask_video"),
        "metadata_json": outputs.get("metadata_json"),
        "hevc12_bytes": compression.get("hevc12_bytes"),
        "valid_mask_bytes": compression.get("valid_mask_bytes"),
        "total_bytes": compression.get("hevc12_with_valid_mask_bytes"),
    }


def config_with_cli_overrides(config: DepthArchiveConfig, args: argparse.Namespace) -> DepthArchiveConfig:
    overrides = {
        name: getattr(args, name)
        for name in [
            "max_depth_m",
            "invalid_value",
            "fps",
            "write_png",
            "write_avi",
            "write_preview",
            "preview_mode",
            "preview_near_min_m",
            "preview_near_max_m",
            "preview_contrast_percentile",
            "preview_codec",
            "ffv1_container",
            "dry_run_avi",
            "delete_npy",
            "delete_png",
            "write_hevc12",
            "hevc12_crf",
            "hevc12_preset",
            "hevc12_lossless",
            "write_valid_mask",
        ]
        if getattr(args, name) is not None
    }

    if args.ffmpeg_bin is not None:
        overrides["ffmpeg_bin"] = args.ffmpeg_bin

    return replace(config, **overrides)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    yaml_config = load_yaml(config_path)

    # -----------------------------
    # 路径：只从 CLI 获取
    # -----------------------------
    depth_dir = Path(
        args.depth_dir
    ).expanduser().resolve()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else depth_dir
    )

    config = config_with_cli_overrides(config_from_yaml(yaml_config), args)

    # runtime 只做提示，不自动切换环境
    recommended_env = (
        yaml_config
        .get("runtime", {})
        .get("recommended_env", "")
    )

    print(f"python          : {sys.executable}")
    print(f"recommended_env : {recommended_env}")
    print(f"config          : {config_path}")
    print(f"depth_dir       : {depth_dir}")
    print(f"output_dir      : {output_dir}")
    print(f"ffmpeg          : {config.ffmpeg_bin}")

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=config,
    )

    print(
        json.dumps(
            summarize_report_for_cli(report),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
