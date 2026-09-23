from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

try:
    from .runner import OpenVinsRunConfig, run_openvins
except ImportError:
    from runner import OpenVinsRunConfig, run_openvins

DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoboEgo VIO OpenVINS stage.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。")
    parser.add_argument("--session", required=True, help="输入 session 目录。")
    parser.add_argument("--output_dir", default=None, help="可选覆盖输出目录；默认来自 YAML。")
    parser.add_argument("--start_s", type=float, default=None, help="可选覆盖回放起始时间。")
    parser.add_argument("--duration_s", type=float, default=None, help="可选覆盖回放时长。")
    parser.add_argument("--input_mode", choices=("raw", "rectified"), default=None, help="VIO 输入模式；raw 使用原始视频，rectified 使用去畸变/立体矫正视频。")
    parser.add_argument("--openvins_root", default=None, help="可选覆盖 OpenVINS 根目录。")
    parser.add_argument("--openvins_binary", default=None, help="可选覆盖 OpenVINS offline runner 二进制路径。")
    parser.add_argument("--no-downsample_cameras", action="store_true", help="不让 OpenVINS 下采样相机图像。")
    parser.add_argument("--prepare_only", action="store_true", help="只生成 OpenVINS 输入和配置，不启动 offline runner。")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return data


def _template_path(value: str, *, session: Path, openvins_root: Path) -> Path:
    return Path(
        value.format(session=str(session), openvins_root=str(openvins_root))
    ).expanduser().resolve()


def config_from_yaml_and_args(data: dict[str, Any], args: argparse.Namespace) -> OpenVinsRunConfig:
    session = Path(args.session).expanduser().resolve()
    io = data.get("io", {})
    prepare = data.get("prepare", {})
    run = data.get("run", {})
    openvins = data.get("openvins", {})
    runtime = data.get("runtime", {})

    openvins_root = Path(
        args.openvins_root or openvins.get("root", "/home/user/open_vins")
    ).expanduser().resolve()
    binary_value = args.openvins_binary or openvins.get("binary", "{openvins_root}/build/roboego_openvins_offline")
    output_value = args.output_dir or io.get("output_dir", "{session}/vio")

    input_mode = str(getattr(args, "input_mode", None) or prepare.get("input_mode", "raw"))
    if input_mode == "rectified":
        downsample_default = bool(prepare.get("rectified_downsample_cameras", False))
    else:
        downsample_default = bool(prepare.get("downsample_cameras", True))
    downsample = False if args.no_downsample_cameras else downsample_default

    return OpenVinsRunConfig(
        session_dir=session,
        output_dir=_template_path(str(output_value), session=session, openvins_root=openvins_root),
        openvins_root=openvins_root,
        openvins_binary=_template_path(str(binary_value), session=session, openvins_root=openvins_root),
        system_python=Path(runtime.get("system_python", "/usr/bin/python3")).expanduser().resolve(),
        start_s=float(args.start_s if args.start_s is not None else run.get("start_s", 0.0)),
        duration_s=args.duration_s if args.duration_s is not None else run.get("duration_s"),
        input_mode=input_mode,
        downsample_cameras=downsample,
        stereo_tolerance_s=float(prepare.get("stereo_tolerance_s", 0.001)),
        max_imu_gap_s=float(prepare.get("max_imu_gap_s", 0.05)),
        max_imu_boundary_gap_s=float(prepare.get("max_imu_boundary_gap_s", 0.1)),
        prepare_only=bool(args.prepare_only),
    )


def main() -> None:
    args = parse_args()
    config = config_from_yaml_and_args(load_yaml(Path(args.config)), args)
    run_openvins(config)
