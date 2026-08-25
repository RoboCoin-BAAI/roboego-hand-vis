"""Nodes for semantic annotation pipelines."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from roboego_hand_vis.quality.stage1_action_text_resolver import (
    call_qwen_text_resolver,
    merge_text_resolver_results,
    write_text_resolver_review_input,
)
from roboego_hand_vis.quality.stage1_actions import write_stage1_action_output


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _append_optional_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def run_qwen_frame_annotation_node(
    pre_semantic_quality_windows: list[dict[str, Any]] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "qwen_frame_annotation", "status": "skipped", "reason": "disabled"}

    video = _optional_str(config.get("video"))
    output_json = _optional_str(config.get("output_json"))
    if not video:
        raise ValueError("qwen_frame_annotation.video is required when enabled")
    if not output_json:
        raise ValueError("qwen_frame_annotation.output_json is required when enabled")
    windows = pre_semantic_quality_windows or []
    if not windows:
        return {
            "stage": "qwen_frame_annotation",
            "status": "skipped",
            "reason": "no_valid_pre_semantic_windows",
            "valid_window_count": 0,
        }

    windows_json = Path(_optional_str(config.get("windows_json")) or Path(output_json).parent / "qwen_frame_annotation_windows.json")
    windows_json.parent.mkdir(parents=True, exist_ok=True)
    windows_json.write_text(json.dumps(windows, indent=2, ensure_ascii=False), encoding="utf-8")

    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/quality/run_qwen_frame_annotation.py",
        "--video",
        video,
        "--output_json",
        output_json,
    ]
    _append_optional_arg(args, "--frames_dir", config.get("frames_dir"))
    _append_optional_arg(args, "--prompt", config.get("prompt"))
    _append_optional_arg(args, "--checkpoint_json", config.get("checkpoint_json"))
    _append_optional_arg(args, "--raw_output_dir", config.get("raw_output_dir"))
    _append_optional_arg(args, "--batch_output_dir", config.get("batch_output_dir"))
    _append_optional_arg(args, "--chunk_output_dir", config.get("chunk_output_dir"))
    _append_optional_arg(args, "--fps", config.get("fps"))
    _append_optional_arg(args, "--start_s", config.get("start_s"))
    _append_optional_arg(args, "--duration_s", config.get("duration_s"))
    _append_optional_arg(args, "--max_width", config.get("max_width"))
    _append_optional_arg(args, "--batch_size", config.get("batch_size"))
    _append_optional_arg(args, "--batches_per_chunk", config.get("batches_per_chunk"))
    _append_optional_arg(args, "--model", config.get("model"))
    _append_optional_arg(args, "--windows_json", windows_json)
    _append_optional_arg(args, "--api_key_env", config.get("api_key_env"))
    _append_optional_arg(args, "--timeout_s", config.get("timeout_s"))
    _append_optional_arg(args, "--max_retries", config.get("max_retries"))
    _append_optional_arg(args, "--retry_sleep_s", config.get("retry_sleep_s"))
    if bool(config.get("resume", False)):
        args.append("--resume")
    if bool(config.get("recover_from_raw", False)):
        args.append("--recover_from_raw")

    subprocess.run(args, check=True)
    return {
        "stage": "qwen_frame_annotation",
        "status": "complete",
        "video": video,
        "output_json": output_json,
        "valid_window_count": len(windows),
        "window_filter_applied": True,
    }


def run_stage1_action_extraction_chunks_node(
    qwen_frame_annotation_report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return _skipped_chunk_report("stage1_action_extraction_chunks", "disabled")
    qwen_chunk_dir = _required_dir(config, "qwen_chunk_dir", "stage1_action_chunks")
    action_chunk_dir = Path(_required_value(config, "action_chunk_dir", "stage1_action_chunks"))
    overwrite = bool(config.get("overwrite", False))

    processed: list[str] = []
    skipped_existing = 0
    for input_path in _chunk_paths(qwen_chunk_dir):
        output_path = action_chunk_dir / input_path.name
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        write_stage1_action_output(input_json=input_path, output_json=output_path)
        processed.append(str(output_path))
    return _chunk_report(
        stage="stage1_action_extraction_chunks",
        input_dir=qwen_chunk_dir,
        output_dir=action_chunk_dir,
        processed=processed,
        skipped_existing_count=skipped_existing,
        upstream=qwen_frame_annotation_report,
    )


def run_stage1_action_text_review_chunks_node(
    stage1_action_chunks_report: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return _skipped_chunk_report("stage1_action_text_review_chunks", "disabled")
    action_chunk_dir = _required_dir(config, "action_chunk_dir", "stage1_action_chunks")
    review_chunk_dir = Path(_required_value(config, "review_chunk_dir", "stage1_action_chunks"))
    overwrite = bool(config.get("overwrite", False))

    processed: list[str] = []
    skipped_existing = 0
    for input_path in _chunk_paths(action_chunk_dir):
        output_path = review_chunk_dir / input_path.name
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue
        write_text_resolver_review_input(input_json=input_path, output_json=output_path)
        processed.append(str(output_path))
    return _chunk_report(
        stage="stage1_action_text_review_chunks",
        input_dir=action_chunk_dir,
        output_dir=review_chunk_dir,
        processed=processed,
        skipped_existing_count=skipped_existing,
        upstream=stage1_action_chunks_report,
    )


def run_stage1_action_text_resolver_chunks_node(
    stage1_action_text_review_chunks_report: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return _skipped_chunk_report("stage1_action_text_resolver_chunks", "disabled")
    action_chunk_dir = _required_dir(config, "action_chunk_dir", "stage1_action_chunks")
    review_chunk_dir = _required_dir(config, "review_chunk_dir", "stage1_action_chunks")
    resolved_chunk_dir = Path(_required_value(config, "resolved_chunk_dir", "stage1_action_chunks"))
    raw_output_dir = Path(_optional_str(config.get("raw_output_dir")) or resolved_chunk_dir.parent / "stage1_action_text_resolver_raw")
    resolver_response_dir = Path(_optional_str(config.get("resolver_response_dir")) or resolved_chunk_dir.parent / "stage1_action_text_resolver_responses")
    overwrite = bool(config.get("overwrite", False))
    model = _optional_str(config.get("model")) or "qwen-plus"
    api_key_env = _optional_str(config.get("api_key_env")) or "DASHSCOPE_API_KEY"
    timeout_s = float(config.get("timeout_s", 120.0))

    processed: list[str] = []
    skipped_existing = 0
    empty_review_count = 0
    for review_path in _chunk_paths(review_chunk_dir):
        action_path = action_chunk_dir / review_path.name
        if not action_path.exists():
            raise FileNotFoundError(f"Missing action chunk for review chunk: {action_path}")
        output_path = resolved_chunk_dir / review_path.name
        if output_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        action_payload = json.loads(action_path.read_text(encoding="utf-8"))
        if int(review_payload.get("frame_count", len(review_payload.get("frames", [])))) == 0:
            merged = dict(action_payload)
            merged["text_resolver_frame_count"] = 0
            empty_review_count += 1
        else:
            parsed = call_qwen_text_resolver(
                review_payload=review_payload,
                model=model,
                api_key_env=api_key_env,
                timeout_s=timeout_s,
            )
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            (raw_output_dir / f"{review_path.stem}.raw.txt").write_text(str(parsed.get("raw_response", "")), encoding="utf-8")
            resolver_response_dir.mkdir(parents=True, exist_ok=True)
            (resolver_response_dir / review_path.name).write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
            if parsed.get("status") != "ok":
                raise RuntimeError(f"Qwen text resolver failed for {review_path.name}: {parsed.get('status')}")
            merged = merge_text_resolver_results(action_payload, parsed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        processed.append(str(output_path))
    report = _chunk_report(
        stage="stage1_action_text_resolver_chunks",
        input_dir=review_chunk_dir,
        output_dir=resolved_chunk_dir,
        processed=processed,
        skipped_existing_count=skipped_existing,
        upstream=stage1_action_text_review_chunks_report,
    )
    report["empty_review_count"] = empty_review_count
    return report


def run_build_processing_plan_node(
    stage1_action_text_resolver_chunks_report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "build_processing_plan", "status": "skipped", "reason": "disabled"}

    stage1_json = _required_value(config, "stage1_json", "processing_plan")
    output_json = _required_value(config, "output_json", "processing_plan")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/quality/build_processing_plan.py",
        "--stage1_json",
        stage1_json,
        "--output_json",
        output_json,
    ]
    _append_optional_arg(args, "--actions_json", config.get("actions_json"))
    _append_optional_arg(args, "--min_hold_interruption_s", config.get("min_hold_interruption_s"))
    _append_optional_arg(args, "--min_walk_interruption_s", config.get("min_walk_interruption_s"))
    _append_optional_arg(args, "--min_idle_interruption_s", config.get("min_idle_interruption_s"))
    _append_optional_arg(args, "--context_expand_s", config.get("context_expand_s"))
    _append_optional_arg(args, "--min_process_segment_s", config.get("min_process_segment_s"))
    _append_optional_arg(args, "--merge_process_gap_s", config.get("merge_process_gap_s"))

    subprocess.run(args, check=True)
    return {
        "stage": "build_processing_plan",
        "status": "complete",
        "stage1_json": stage1_json,
        "actions_json": _optional_str(config.get("actions_json")),
        "output_json": output_json,
        "upstream_status": (stage1_action_text_resolver_chunks_report or {}).get("status", ""),
    }


def run_build_hand_estimation_windows_node(
    build_processing_plan_report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "build_hand_estimation_windows", "status": "skipped", "reason": "disabled"}

    stage1_actions = _required_value(config, "stage1_actions", "hand_estimation_windows")
    processing_plan = _required_value(config, "processing_plan", "hand_estimation_windows")
    output_json = _required_value(config, "output_json", "hand_estimation_windows")
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/quality/build_hand_estimation_windows.py",
        "--stage1_actions",
        stage1_actions,
        "--processing_plan",
        processing_plan,
        "--output_json",
        output_json,
    ]
    _append_optional_arg(args, "--context_expand_s", config.get("context_expand_s"))
    _append_optional_arg(args, "--merge_gap_s", config.get("merge_gap_s"))
    if bool(config.get("merge_after_expand", False)):
        args.append("--merge_after_expand")
    _append_optional_arg(args, "--min_window_s", config.get("min_window_s"))
    _append_optional_arg(args, "--min_priority", config.get("min_priority"))
    _append_optional_arg(args, "--min_core_duration_s", config.get("min_core_duration_s"))
    _append_optional_arg(args, "--fps", config.get("fps"))
    if "require_both_hands" in config and not bool(config.get("require_both_hands")):
        args.append("--no-require_both_hands")
    _append_optional_arg(args, "--hand_window_mode", config.get("hand_window_mode"))

    subprocess.run(args, check=True)
    return {
        "stage": "build_hand_estimation_windows",
        "status": "complete",
        "stage1_actions": stage1_actions,
        "processing_plan": processing_plan,
        "output_json": output_json,
        "upstream_status": (build_processing_plan_report or {}).get("status", ""),
    }


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value


def _required_dir(config: dict[str, Any], key: str, namespace: str) -> Path:
    return Path(_required_value(config, key, namespace))


def _chunk_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("chunk_*.json") if path.is_file())


def _skipped_chunk_report(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "status": "skipped", "reason": reason}


def _chunk_report(
    *,
    stage: str,
    input_dir: Path,
    output_dir: Path,
    processed: list[str],
    skipped_existing_count: int,
    upstream: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "complete",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_count": len(processed),
        "processed_outputs": processed,
        "skipped_existing_count": skipped_existing_count,
        "upstream_status": (upstream or {}).get("status", ""),
    }
