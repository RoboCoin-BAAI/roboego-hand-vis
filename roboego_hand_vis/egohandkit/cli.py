from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import EgoHandKitConfig, run_egohandkit
except ImportError:
    from compute import EgoHandKitConfig, run_egohandkit


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用外部 EgoHandKit 仓库运行手部估计。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="EgoHandKit stage 参数 YAML。")
    parser.add_argument("--input", required=True, help="输入视频、图片目录或 EgoHandKit 支持的输入路径。")
    parser.add_argument("--output_root", required=True, help="EgoHandKit 输出根目录。")
    parser.add_argument("--depth_dir", default="", help="深度目录；为空则不传入。")
    parser.add_argument("--prediction", default="", help="已有预测文件；为空则不传入。")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def config_from_yaml(
    data: dict[str, Any],
    *,
    input: Path,
    output_root: Path,
    depth_dir: Path | None = None,
    prediction: Path | None = None,
) -> EgoHandKitConfig:
    paths = data.get("paths", {})
    parameters = data.get("parameters", {})
    return EgoHandKitConfig(
        input=input,
        output_root=output_root,
        egohandkit_root=Path(paths.get("egohandkit_root", "../EgoHandKit")).expanduser(),
        python=str(paths.get("python", "python")),
        run_script=str(paths.get("run_script", "run.py")),
        depth_dir=depth_dir,
        prediction=prediction,
        frontend=str(parameters.get("frontend", "canonical")),
        observations=str(parameters.get("observations", "observations.jsonl")),
        backend=str(parameters.get("backend", "hamer")),
        gpu=parameters.get("gpu", 0),
        batch_size=parameters.get("batch_size", 16),
        depth_gate=bool(parameters.get("depth_gate", True)),
        motion_gate=bool(parameters.get("motion_gate", True)),
        mint_3d_consistency_gate=bool(parameters.get("mint_3d_consistency_gate", True)),
        hand_tracking_parquet=bool(parameters.get("hand_tracking_parquet", True)),
        render_hand_tracking_parquet=bool(parameters.get("render_hand_tracking_parquet", True)),
        no_omega_world=bool(parameters.get("no_omega_world", True)),
        extra_args=[str(value) for value in parameters.get("extra_args", [])],
    )


def main() -> None:
    args = parse_args()
    data = load_yaml(Path(args.config).expanduser().resolve())
    config = config_from_yaml(
        data,
        input=Path(args.input).expanduser().resolve(),
        output_root=Path(args.output_root).expanduser().resolve(),
        depth_dir=Path(args.depth_dir).expanduser().resolve() if args.depth_dir else None,
        prediction=Path(args.prediction).expanduser().resolve() if args.prediction else None,
    )
    print(json.dumps(run_egohandkit(config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

