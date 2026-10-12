from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

try:
    from .compute import DEFAULT_POST_ANNOTATION_BASE_URL, PostAnnotationActionsConfig, run_post_annotation_actions
except ImportError:
    from compute import DEFAULT_POST_ANNOTATION_BASE_URL, PostAnnotationActionsConfig, run_post_annotation_actions


DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Qwen 标注后的动作标准化阶段。")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="阶段参数 YAML；默认使用当前目录下 recommended.yaml。")
    parser.add_argument("--qwen_chunk_dir", required=True, help="Qwen 帧标注 chunk 目录。")
    parser.add_argument("--output_dir", default="", help="输出目录；默认写到 qwen_chunk_dir 同级 annotation 目录。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 resolved chunk。")
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
    qwen_chunk_dir: Path,
    output_dir: Path | None = None,
    overwrite: bool | None = None,
) -> PostAnnotationActionsConfig:
    parameters = data.get("parameters", {})
    base_url = (str(parameters.get("base_url") or "").strip()
                or os.environ.get("DASHSCOPE_BASE_URL", "").strip()
                or DEFAULT_POST_ANNOTATION_BASE_URL)
    qwen_chunk_dir = _path(qwen_chunk_dir)
    base_dir = _path(output_dir) if output_dir is not None else qwen_chunk_dir.parent
    return PostAnnotationActionsConfig(
        qwen_chunk_dir=qwen_chunk_dir,
        action_chunk_dir=_path(parameters.get("action_chunk_dir") or base_dir / "stage1_action_chunks"),
        review_chunk_dir=_path(parameters.get("review_chunk_dir") or base_dir / "stage1_action_text_review_chunks"),
        resolved_chunk_dir=_path(parameters.get("resolved_chunk_dir") or base_dir / "stage1_action_resolved_chunks"),
        merged_output_json=_path(parameters.get("merged_output_json") or base_dir / "stage1_actions_left_v1_resolved.json"),
        subtask_output_json=_path(parameters.get("subtask_output_json") or base_dir / "stage1_subtasks_left_v1.json"),
        frames_dir=_optional_path(parameters.get("frames_dir") or base_dir / "qwen_frame_annotation_frames"),
        base_url=base_url,
        text_model=str(parameters.get("text_model", "qwen3.8-27b")),
        subtask_model=str(parameters.get("subtask_model", parameters.get("text_model", "qwen3.8-27b"))),
        subtask_prompt=_path(parameters.get("subtask_prompt", "docs/prompts/stage1_subtask_segmentation_en.md")),
        subtask_object_prompt=_path(
            parameters.get("subtask_object_prompt", "docs/prompts/stage1_subtask_object_resolution_en.md")
        ),
        subtask_chunk_size=int(parameters.get("subtask_chunk_size", 8)),
        subtask_overlap=int(parameters.get("subtask_overlap", 1)),
        generate_subtasks=bool(parameters.get("generate_subtasks", True)),
        use_vl_object_resolver=bool(parameters.get("use_vl_object_resolver", True)),
        vl_object_model=str(parameters.get("vl_object_model", "qwen3.8-27b")),
        max_vl_frames_per_chunk=int(parameters.get("max_vl_frames_per_chunk", 30)),
        api_key_env=str(parameters.get("api_key_env", "DASHSCOPE_API_KEY")),
        timeout_s=float(parameters.get("timeout_s", 120.0)),
        overwrite=bool(parameters.get("overwrite", False) if overwrite is None else overwrite),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    config = config_from_yaml(
        load_yaml(Path(args.config).expanduser()),
        qwen_chunk_dir=Path(args.qwen_chunk_dir),
        output_dir=output_dir,
        overwrite=args.overwrite if args.overwrite else None,
    )
    result = run_post_annotation_actions(config)
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
