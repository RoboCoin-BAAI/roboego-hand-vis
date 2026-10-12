from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboego_hand_vis.post_annotation_actions.robolabel_actions import standardize_stage1_actions
from roboego_hand_vis.post_annotation_actions.text_resolver import (
    call_qwen_text_resolver,
    merge_text_resolver_results,
    write_text_resolver_review_input,
)
from roboego_hand_vis.post_annotation_actions.stage1_actions import write_stage1_action_output
from roboego_hand_vis.post_annotation_actions.model_subtasks import (
    call_qwen_object_resolver,
    call_qwen_subtask_resolver,
    write_model_subtask_output,
)


DEFAULT_POST_ANNOTATION_BASE_URL = "https://llm-g3o8d3j71xbf6prc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"


@dataclass(frozen=True)
class PostAnnotationActionsConfig:
    qwen_chunk_dir: Path
    action_chunk_dir: Path
    review_chunk_dir: Path
    resolved_chunk_dir: Path
    merged_output_json: Path
    subtask_output_json: Path
    frames_dir: Path | None
    text_model: str
    subtask_model: str
    subtask_prompt: Path
    subtask_object_prompt: Path
    subtask_chunk_size: int
    subtask_overlap: int
    generate_subtasks: bool
    use_vl_object_resolver: bool
    vl_object_model: str
    max_vl_frames_per_chunk: int
    api_key_env: str
    timeout_s: float
    overwrite: bool
    base_url: str = DEFAULT_POST_ANNOTATION_BASE_URL


def run_post_annotation_actions(config: PostAnnotationActionsConfig) -> dict[str, Any]:
    processed = []
    skipped_existing = 0
    empty_review_count = 0
    for qwen_chunk in _chunk_paths(config.qwen_chunk_dir):
        action_chunk = config.action_chunk_dir / qwen_chunk.name
        review_chunk = config.review_chunk_dir / qwen_chunk.name
        resolved_chunk = config.resolved_chunk_dir / qwen_chunk.name
        if resolved_chunk.exists() and not config.overwrite:
            skipped_existing += 1
            continue

        write_stage1_action_output(input_json=qwen_chunk, output_json=action_chunk)
        review_payload = write_text_resolver_review_input(input_json=action_chunk, output_json=review_chunk)
        action_payload = json.loads(action_chunk.read_text(encoding="utf-8"))
        if int(review_payload.get("frame_count", len(review_payload.get("frames", [])))) == 0:
            merged = dict(action_payload)
            merged["text_resolver_frame_count"] = 0
            empty_review_count += 1
        else:
            parsed = call_qwen_text_resolver(
                review_payload=review_payload,
                model=config.text_model,
                api_key_env=config.api_key_env,
                base_url=config.base_url,
                timeout_s=config.timeout_s,
            )
            if parsed.get("status") != "ok":
                raise RuntimeError(f"Qwen text resolver failed for {review_chunk.name}: {parsed.get('status')}")
            merged = merge_text_resolver_results(action_payload, parsed)

        resolved = standardize_stage1_actions(
            merged,
            frames_dir=config.frames_dir,
            use_vl_object_resolver=config.use_vl_object_resolver,
            vl_model=config.vl_object_model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            timeout_s=config.timeout_s,
            max_vl_frames=config.max_vl_frames_per_chunk,
        )
        resolved_chunk.parent.mkdir(parents=True, exist_ok=True)
        resolved_chunk.write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
        processed.append(str(resolved_chunk))

    merged_payload = merge_resolved_chunks(config.resolved_chunk_dir)
    config.merged_output_json.parent.mkdir(parents=True, exist_ok=True)
    config.merged_output_json.write_text(json.dumps(merged_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    subtask_payload = None
    if config.generate_subtasks:
        subtask_payload = _write_subtasks(config)
    return {
        "stage": "post_annotation_actions",
        "status": "complete",
        "qwen_chunk_dir": str(config.qwen_chunk_dir),
        "action_chunk_dir": str(config.action_chunk_dir),
        "review_chunk_dir": str(config.review_chunk_dir),
        "resolved_chunk_dir": str(config.resolved_chunk_dir),
        "merged_output_json": str(config.merged_output_json),
        "subtask_output_json": str(config.subtask_output_json) if subtask_payload is not None else "",
        "processed_count": len(processed),
        "processed_outputs": processed,
        "skipped_existing_count": skipped_existing,
        "empty_review_count": empty_review_count,
        "merged_frame_count": int(merged_payload.get("frame_count", 0) or 0),
        "use_vl_object_resolver": config.use_vl_object_resolver,
        "vl_object_model": config.vl_object_model,
    }


def _write_subtasks(config: PostAnnotationActionsConfig) -> dict[str, Any]:
    def resolver(review_payload: dict[str, Any]) -> dict[str, Any]:
        return call_qwen_subtask_resolver(
            review_payload,
            prompt_path=config.subtask_prompt,
            model=config.subtask_model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            timeout_s=config.timeout_s,
        )

    def object_resolver(review_payload: dict[str, Any]) -> dict[str, Any]:
        return call_qwen_object_resolver(
            review_payload,
            prompt_path=config.subtask_object_prompt,
            model=config.subtask_model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            timeout_s=config.timeout_s,
        )

    return write_model_subtask_output(
        input_json=config.merged_output_json,
        output_json=config.subtask_output_json,
        resolver=resolver,
        object_resolver=object_resolver,
        prompt_path=config.subtask_prompt,
        model=config.subtask_model,
        chunk_size=config.subtask_chunk_size,
        overlap=config.subtask_overlap,
    )


def merge_resolved_chunks(chunk_dir: Path) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    source_files = []
    action_frame_count = 0
    for chunk_path in _chunk_paths(chunk_dir):
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        source_files.append(str(chunk_path))
        for frame in payload.get("frames", []):
            if not isinstance(frame, dict):
                continue
            frame_out = dict(frame)
            if not isinstance(frame_out.get("frame_actions"), list):
                frame_out["frame_actions"] = []
            if frame_out["frame_actions"]:
                action_frame_count += 1
            frames.append(frame_out)
    frames = sorted(frames, key=lambda frame: float(frame.get("t", 0)))
    return {
        "stage": "robolabel_stage1_actions_merged",
        "source_chunk_dir": str(chunk_dir),
        "source_files": source_files,
        "frame_count": len(frames),
        "action_frame_count": action_frame_count,
        "frames": frames,
    }


def _chunk_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("chunk_*.json") if path.is_file())
