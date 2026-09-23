from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import HandWindowsStageConfig, run_hand_windows_stage
except ImportError:
    from compute import HandWindowsStageConfig, run_hand_windows_stage


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行手部估计窗口切分阶段。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。")
    parser.add_argument("--stage1_actions", required=True, help="stage1_actions_left_v1_resolved.json 路径。")
    parser.add_argument("--processing_plan", required=True, help="processing_plan_left_v1.json 路径。")
    parser.add_argument("--output_json", default="", help="hand windows 输出 JSON；默认写到 stage1_actions 同目录。")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def config_from_yaml(
    data: dict[str, Any],
    *,
    stage1_actions: Path,
    processing_plan: Path,
    output_json: Path | None = None,
) -> HandWindowsStageConfig:
    parameters = data.get("parameters", {})
    stage1_actions = _path(stage1_actions)
    processing_plan = _path(processing_plan)
    base_dir = stage1_actions.parent
    return HandWindowsStageConfig(
        stage1_actions=stage1_actions,
        processing_plan=processing_plan,
        output_json=_path(output_json) if output_json is not None else _path(parameters.get("output_json") or base_dir / "hand_estimation_windows_left_v1.json"),
        require_both_hands=bool(parameters.get("require_both_hands", True)),
        hand_window_mode=str(parameters.get("hand_window_mode", "two")),
        context_expand_s=int(parameters.get("context_expand_s", 1)),
        merge_gap_s=int(parameters.get("merge_gap_s", 2)),
        merge_after_expand=bool(parameters.get("merge_after_expand", False)),
        min_window_s=int(parameters.get("min_window_s", 1)),
        min_priority=_optional_str(parameters.get("min_priority")),
        min_core_duration_s=int(parameters.get("min_core_duration_s", 0)),
        fps=int(parameters.get("fps", 30)),
    )


def main() -> None:
    args = parse_args()
    config = config_from_yaml(
        load_yaml(Path(args.config).expanduser()),
        stage1_actions=Path(args.stage1_actions),
        processing_plan=Path(args.processing_plan),
        output_json=Path(args.output_json).expanduser() if args.output_json else None,
    )
    result = run_hand_windows_stage(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

