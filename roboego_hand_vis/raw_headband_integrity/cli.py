from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import DEFAULT_CAMERA_STREAMS, DEFAULT_VIDEO_EXT, export_quality_report
    from .rules import evaluate_quality_report
except ImportError:
    from compute import DEFAULT_CAMERA_STREAMS, DEFAULT_VIDEO_EXT, export_quality_report
    from rules import evaluate_quality_report


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


@dataclass(frozen=True)
class RawHeadbandIntegrityConfig:
    session: Path
    output_root: Path
    camera_streams: tuple[str, ...]
    start: int
    num_frames: int | None
    timestamp_tolerance_s: float | None
    video_ext: str
    overwrite: bool
    environment_name: str
    repair_timestamp_files: bool
    max_repair_duplicate_rate: float
    max_repair_consecutive_duplicate_frames: int
    gate_enabled: bool
    gate_rules: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 Raw 头环数据完整性，并输出统一 quality_report.json。")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。",
    )
    parser.add_argument("--session", required=True, help="Raw session 目录。")
    parser.add_argument("--output_root", default="", help="报告输出根目录；默认写回 session。")
    parser.add_argument(
        "--camera_stream",
        action="append",
        choices=DEFAULT_CAMERA_STREAMS,
        default=None,
        help="只检查指定相机流；可重复传入。默认使用 YAML 中的 camera_streams。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 quality_report.json。")
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return float(value)


def config_from_yaml(data: dict[str, Any], args: argparse.Namespace) -> RawHeadbandIntegrityConfig:
    parameters = data.get("parameters", {})
    gate = data.get("gate", {})
    session = _path(args.session)
    output_root = _path(args.output_root) if args.output_root else session
    camera_streams = tuple(args.camera_stream or parameters.get("camera_streams", DEFAULT_CAMERA_STREAMS))

    return RawHeadbandIntegrityConfig(
        session=session,
        output_root=output_root,
        camera_streams=camera_streams,
        start=int(parameters.get("start", 0)),
        num_frames=_optional_int(parameters.get("num_frames")),
        timestamp_tolerance_s=_optional_float(parameters.get("timestamp_tolerance_s")),
        video_ext=str(parameters.get("video_ext", DEFAULT_VIDEO_EXT)),
        overwrite=bool(args.overwrite or parameters.get("overwrite", False)),
        environment_name=str(parameters.get("environment_name", "py312")),
        repair_timestamp_files=bool(parameters.get("repair_timestamp_files", False)),
        max_repair_duplicate_rate=float(parameters.get("max_repair_duplicate_rate", 0.001)),
        max_repair_consecutive_duplicate_frames=int(
            parameters.get("max_repair_consecutive_duplicate_frames", 5)
        ),
        gate_enabled=bool(gate.get("enabled", True)),
        gate_rules=dict(gate.get("rules") or {}),
    )


def run_raw_headband_integrity(config: RawHeadbandIntegrityConfig) -> dict[str, Any]:
    report_path = export_quality_report(
        session_dir=config.session,
        output_root=config.output_root,
        camera_streams=config.camera_streams,
        start=config.start,
        num_frames=config.num_frames,
        timestamp_tolerance_s=config.timestamp_tolerance_s,
        video_ext=config.video_ext,
        overwrite=config.overwrite,
        environment_name=config.environment_name,
        repair_timestamp_files=config.repair_timestamp_files,
        max_repair_duplicate_rate=config.max_repair_duplicate_rate,
        max_repair_consecutive_duplicate_frames=config.max_repair_consecutive_duplicate_frames,
    )

    summary: dict[str, Any] = {
        "stage": "raw_headband_integrity",
        "report_path": str(report_path),
    }
    if config.gate_enabled:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gate_result = evaluate_quality_report(report, config.gate_rules)
        report["raw_headband_quality_gate"] = gate_result
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["gate"] = {
            "passed": gate_result.get("passed"),
            "risk_level": gate_result.get("risk_level"),
            "blocking_check_count": len(gate_result.get("blocking_checks") or ()),
            "warning_check_count": len(gate_result.get("warning_checks") or ()),
        }
    return summary


def main() -> None:
    args = parse_args()
    config = config_from_yaml(load_yaml(Path(args.config).expanduser()), args)
    result = run_raw_headband_integrity(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
