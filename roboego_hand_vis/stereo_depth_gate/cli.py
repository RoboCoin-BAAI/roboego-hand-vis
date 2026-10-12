from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import (
        SummaryQualityConfig,
        DepthUsableThresholds,
        build_sample_row,
        inspect_video,
        load_manifest_video_paths,
        read_frame,
        sample_frame_indices,
        summarize,
    )
except ImportError:
    from compute import (
        SummaryQualityConfig,
        DepthUsableThresholds,
        build_sample_row,
        inspect_video,
        load_manifest_video_paths,
        read_frame,
        sample_frame_indices,
        summarize,
    )

DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


@dataclass(frozen=True)
class StereoDepthGateRunConfig:
    manifest_path: Path
    output_dir: Path
    summary: SummaryQualityConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-depth stereo video quality gate for one session.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML；默认使用当前 stage 目录下 recommended.yaml。")
    parser.add_argument("--session", required=True, help="Session directory containing dataset_manifest/dataset_manifest.json.")
    parser.add_argument("--manifest", default="", help="可选覆盖输入 manifest 路径。")
    parser.add_argument("--output_dir", default="", help="可选覆盖输出目录。")
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--min_orb_matches", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def _resolve_template(value: str, *, session: Path) -> Path:
    return Path(value.format(session=str(session))).expanduser().resolve()


def config_from_yaml_and_args(data: dict[str, Any], args: argparse.Namespace) -> StereoDepthGateRunConfig:
    session = Path(args.session).expanduser().resolve()
    io_config = data.get("io", {})
    summary_config = data.get("summary", {})
    thresholds = data.get("thresholds", {})
    depth_thresholds = data.get("depth_usable_thresholds", {})

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else _resolve_template(str(io_config.get("manifest", "{session}/dataset_manifest/dataset_manifest.json")), session=session)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _resolve_template(str(io_config.get("output_dir", "{session}/quality/stereo_depth_gate")), session=session)
    )

    threshold_values = {
        "max_brightness_gap": float(thresholds.get("max_brightness_gap", 35.0)),
        "min_sharpness": float(thresholds.get("min_sharpness", 50.0)),
        "max_dark_ratio": float(thresholds.get("max_dark_ratio", 0.25)),
        "max_bright_ratio": float(thresholds.get("max_bright_ratio", 0.25)),
        "min_orb_matches": int(args.min_orb_matches if args.min_orb_matches is not None else thresholds.get("min_orb_matches", 30)),
        "max_median_abs_dy": float(thresholds.get("max_median_abs_dy", 2.0)),
        "max_p90_abs_dy": float(thresholds.get("max_p90_abs_dy", 5.0)),
    }

    return StereoDepthGateRunConfig(
        manifest_path=manifest_path,
        output_dir=output_dir,
        summary=SummaryQualityConfig(
            depth_usable_thresholds=DepthUsableThresholds(
                max_median_abs_dy=float(depth_thresholds.get("max_median_abs_dy", 8.0)),
                max_p90_abs_dy=float(depth_thresholds.get("max_p90_abs_dy", 70.0)),
            ),
            sample_count=int(args.sample_count if args.sample_count is not None else summary_config.get("sample_count", 12)),
            dark_threshold=int(summary_config.get("dark_threshold", 10)),
            bright_threshold=int(summary_config.get("bright_threshold", 245)),
            orb_features=int(summary_config.get("orb_features", 1500)),
            **threshold_values,
        ),
    )


def run_stereo_depth_gate(config: StereoDepthGateRunConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    left_path, right_path, manifest = load_manifest_video_paths(config.manifest_path)
    left_info = inspect_video(left_path)
    right_info = inspect_video(right_path)
    frame_count = min(left_info["frame_count"], right_info["frame_count"])
    sample_indices = sample_frame_indices(frame_count, config.summary.sample_count)

    sample_frames = []
    for frame_idx in sample_indices:
        left_frame = read_frame(left_path, frame_idx)
        right_frame = read_frame(right_path, frame_idx)
        sample_frames.append(
            build_sample_row(
                frame_idx,
                left_info["fps"],
                left_frame,
                right_frame,
                config.summary.dark_threshold,
                config.summary.bright_threshold,
                config.summary.orb_features,
            )
        )

    summary = summarize(sample_frames, config.summary)
    gate_report = {
        "stage": "stereo_depth_gate",
        "manifest_path": str(config.manifest_path),
        "dataset_name": manifest.get("dataset_name"),
        "sequence_name": manifest.get("sequence_name"),
        "output_dir": str(config.output_dir),
        "left_video": left_info,
        "right_video": right_info,
        "sample_indices": sample_indices,
        "metrics": {
            "brightness_gap_avg": summary.get("brightness_gap_avg"),
            "left_dark_ratio_max": summary.get("left_dark_ratio_max"),
            "right_dark_ratio_max": summary.get("right_dark_ratio_max"),
            "left_bright_ratio_max": summary.get("left_bright_ratio_max"),
            "right_bright_ratio_max": summary.get("right_bright_ratio_max"),
            "left_sharp_avg": summary.get("left_sharp_avg"),
            "right_sharp_avg": summary.get("right_sharp_avg"),
            "orb_matches_min": summary.get("orb_matches_min"),
            "median_abs_dy_avg": summary.get("median_abs_dy_avg"),
            "p90_abs_dy_max": summary.get("p90_abs_dy_max"),
        },
        "checks": summary.get("checks", {}),
        "sample_frames": sample_frames,
        "summary_passed": bool(summary["passed"]),
        "usable_for_depth": bool(summary.get("usable_for_depth", False)),
        "depth_risk_level": summary.get("depth_risk_level"),
        "depth_risk_reasons": summary.get("depth_risk_reasons", []),
        "thresholds": {
            "max_brightness_gap": config.summary.max_brightness_gap,
            "min_sharpness": config.summary.min_sharpness,
            "max_dark_ratio": config.summary.max_dark_ratio,
            "max_bright_ratio": config.summary.max_bright_ratio,
            "min_orb_matches": config.summary.min_orb_matches,
            "max_median_abs_dy": config.summary.max_median_abs_dy,
            "max_p90_abs_dy": config.summary.max_p90_abs_dy,
        },
        "outputs": {
            "gate_report": str(config.output_dir / "stereo_depth_gate_report.json"),
        },
    }
    # Keep the original fields while making the actual release policy explicit.
    gate_report["strict_thresholds"] = dict(gate_report["thresholds"])
    gate_report["depth_usable_thresholds"] = {
        **gate_report["thresholds"], **summary["usable_for_depth_thresholds"],
    }
    gate_report["depth_usable_checks"] = summary["depth_usable_checks"]
    gate_report["strict_passed"] = bool(summary["passed"])
    if summary["passed"] and summary["usable_for_depth"]:
        explanation = "严格质检与深度可用性检查均通过，允许继续深度估计，风险等级为 low。"
    elif summary["usable_for_depth"]:
        explanation = ("严格几何阈值未通过，但满足宽松几何阈值，且亮度、清晰度、曝光和匹配数量检查均通过。"
                       "允许继续深度估计，风险等级为 high_risk；门禁放行不代表深度结果质量可靠。")
    else:
        explanation = "深度可用性检查未通过，阻止继续深度估计；失败项见 depth_usable_checks。"
    gate_report["gate_decision"] = {
        "based_on": "usable_for_depth",
        "passed": bool(summary["usable_for_depth"]),
        "database_status": "completed" if summary["usable_for_depth"] else "block",
        "explanation": explanation,
        "strict_thresholds_source": "config.thresholds",
        "depth_usable_thresholds_source": "config.depth_usable_thresholds（几何）；其他项目继承 config.thresholds",
        "metric_definitions": {
            "median_abs_dy_avg": "各抽样帧 ORB 匹配点垂直误差绝对值中位数的平均值，单位像素。",
            "p90_abs_dy_max": "各抽样帧 ORB 匹配点垂直误差绝对值 P90 的最大值，单位像素。",
        },
    }
    (config.output_dir / "stereo_depth_gate_report.json").write_text(
        json.dumps(gate_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return gate_report


def main() -> None:
    args = parse_args()
    config = config_from_yaml_and_args(load_yaml(Path(args.config).expanduser().resolve()), args)
    report = run_stereo_depth_gate(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
