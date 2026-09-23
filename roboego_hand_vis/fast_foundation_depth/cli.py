from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    try:
        from .compute import FastFoundationDepthConfig
    except ImportError:
        from compute import FastFoundationDepthConfig


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 Fast-FoundationStereo，对左右目视频估计逐帧深度并保存为米制深度结果。"
    )

    # 配置文件
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Fast-FoundationStereo 参数 YAML；默认使用当前目录下 recommended.yaml。",
    )

    # 路径只走 CLI
    parser.add_argument(
        "--manifest",
        required=True,
        help="输入数据清单 JSON。",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="深度结果输出目录。",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")

    return config


def config_from_yaml(
    data: dict[str, Any],
    *,
    manifest: Path,
    output_dir: Path,
) -> "FastFoundationDepthConfig":
    try:
        from .compute import FastFoundationDepthConfig
    except ImportError:
        from compute import FastFoundationDepthConfig

    parameters = data.get("parameters", {})
    paths = data.get("paths", {})
    # num_frames: auto -> None
    num_frames_value = parameters.get("num_frames", "auto")

    if isinstance(num_frames_value, str) and num_frames_value.lower() == "auto":
        num_frames = None
    else:
        num_frames = int(num_frames_value)

    return FastFoundationDepthConfig(
        # CLI
        manifest=manifest,
        output_dir=output_dir,

        # YAML paths
        fast_foundation_root=Path(
            paths.get(
                "fast_foundation_root",
                "../Fast-FoundationStereo",
            )
        ),
        ckpt=Path(
            paths.get(
                "ckpt",
                "../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth",
            )
        ),

        # YAML parameters
        start=int(
            parameters.get("start", 0)
        ),
        num_frames=num_frames,
        scale=float(
            parameters.get("scale", 0.5)
        ),
        hiera=int(
            parameters.get("hiera", 0)
        ),
        valid_iters=int(
            parameters.get("valid_iters", 4)
        ),
        max_disp=int(
            parameters.get("max_disp", 192)
        ),
        min_disp=float(
            parameters.get("min_disp", 1e-3)
        ),
        max_depth_m=float(
            parameters.get("max_depth_m", 4.0)
        ),

        save_depth_png=bool(
            parameters.get("save_depth_png", True)
        ),
        save_npy=bool(
            parameters.get("save_npy", False)
        ),
        save_vis=bool(
            parameters.get("save_vis", True)
        ),

        preview_every_n_frames=int(
            parameters.get("preview_every_n_frames", 5)
        ),
        progress_every_n_frames=int(
            parameters.get("progress_every_n_frames", 20)
        ),

        verbose=bool(
            parameters.get("verbose", False)
        ),
        overwrite=bool(
            parameters.get("overwrite", False)
        ),
    )


def summarize_result_for_cli(result: dict) -> dict:
    summary = {
        key: value
        for key, value in result.items()
        if key != "processed_frames"
    }

    processed_frames = result.get("processed_frames") or []

    if processed_frames:
        summary["processed_frame_start"] = processed_frames[0]
        summary["processed_frame_end"] = processed_frames[-1]

    return summary


def main() -> None:
    args = parse_args()

    try:
        from .compute import run_fast_foundation_depth
    except ImportError:
        from compute import run_fast_foundation_depth

    config_path = Path(args.config).expanduser().resolve()
    yaml_config = load_yaml(config_path)

    # 路径来自 CLI
    manifest = Path(
        args.manifest
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    # 其他参数来自 YAML
    config = config_from_yaml(
        yaml_config,
        manifest=manifest,
        output_dir=output_dir,
    )

    result = run_fast_foundation_depth(config)

    print(
        json.dumps(
            summarize_result_for_cli(result),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
