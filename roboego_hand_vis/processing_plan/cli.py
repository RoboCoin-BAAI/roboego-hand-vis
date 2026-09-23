from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import ProcessingPlanStageConfig, run_processing_plan_stage
except ImportError:
    from compute import ProcessingPlanStageConfig, run_processing_plan_stage


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行粗语义处理片段切分阶段。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。")
    parser.add_argument("--stage1_actions", required=True, help="stage1_actions_left_v1_resolved.json 路径。")
    parser.add_argument("--output_json", default="", help="processing plan 输出 JSON；默认写到 stage1_actions 同目录。")
    parser.add_argument("--actions_json", default="", help="可选动作 JSON 或动作 chunk 目录，用于覆盖 stage1 中的 frame_actions。")
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
    output_json: Path | None = None,
    actions_json: Path | None = None,
) -> ProcessingPlanStageConfig:
    parameters = data.get("parameters", {})
    stage1_actions = _path(stage1_actions)
    base_dir = stage1_actions.parent
    output = _path(output_json) if output_json is not None else _path(parameters.get("output_json") or base_dir / "processing_plan_left_v1.json")
    actions = _optional_path(actions_json or parameters.get("actions_json"))
    return ProcessingPlanStageConfig(
        stage1_actions=stage1_actions,
        output_json=output,
        actions_json=actions,
        min_hold_interruption_s=int(parameters.get("min_hold_interruption_s", 8)),
        min_walk_interruption_s=int(parameters.get("min_walk_interruption_s", 5)),
        min_idle_interruption_s=int(parameters.get("min_idle_interruption_s", 8)),
        context_expand_s=int(parameters.get("context_expand_s", 3)),
        min_process_segment_s=int(parameters.get("min_process_segment_s", 1)),
        merge_process_gap_s=int(parameters.get("merge_process_gap_s", 5)),
    )


def main() -> None:
    args = parse_args()
    config = config_from_yaml(
        load_yaml(Path(args.config).expanduser()),
        stage1_actions=Path(args.stage1_actions),
        output_json=Path(args.output_json).expanduser() if args.output_json else None,
        actions_json=Path(args.actions_json).expanduser() if args.actions_json else None,
    )
    result = run_processing_plan_stage(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _path(text)

