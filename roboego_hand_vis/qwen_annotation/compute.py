from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_run import prepare_task_metadata
from .task_context import context_prompts, parse_context_frame_response, validate_task_objects
from roboego_hand_vis.qwen_annotation.frame_annotation import (
    build_data_inspection_placeholder_frames,
    build_qwen_frame_annotation_messages,
    call_qwen_frame_annotations,
    completed_batch_indices,
    extract_video_frames,
    extract_video_frames_from_windows,
    is_data_inspection_failed,
    load_checkpoint_frames,
    merge_qwen_usage,
    read_prompt_markdown,
    recover_batches_from_raw,
    recover_checkpoint_from_raw,
    write_batch_annotation,
    write_checkpoint,
    write_chunk_annotations,
    write_qwen_frame_annotation_output,
)


@dataclass(frozen=True)
class QwenAnnotationConfig:
    video: Path
    output_json: Path
    frames_dir: Path
    prompt: Path
    model: str
    fps: float
    start_s: float
    duration_s: float | None
    windows_json: Path | None
    max_width: int
    batch_size: int
    api_key_env: str
    base_url: str
    chat_template_kwargs: dict[str, Any]
    timeout_s: float
    max_retries: int
    retry_sleep_s: float
    raw_output_dir: Path
    checkpoint_json: Path
    batch_output_dir: Path
    chunk_output_dir: Path
    batches_per_chunk: int
    resume: bool
    recover_from_raw: bool


def run_qwen_annotation(config: QwenAnnotationConfig) -> dict[str, Any]:
    system_prompt, user_prompt = read_prompt_markdown(config.prompt)
    task_metadata = prepare_task_metadata(config)
    context_options = {"task_metadata": task_metadata} if task_metadata is not None else {}
    parser_options = {}
    if task_metadata is not None:
        system_prompt, user_prompt = context_prompts(system_prompt, user_prompt, task_metadata)
        parser_options["response_parser"] = lambda text: parse_context_frame_response(text, task_metadata)
    config.raw_output_dir.mkdir(parents=True, exist_ok=True)
    if config.windows_json is not None:
        windows_payload = json.loads(config.windows_json.read_text(encoding="utf-8"))
        frame_paths, timestamps_s = extract_video_frames_from_windows(
            video_path=config.video,
            output_dir=config.frames_dir,
            windows=windows_payload,
            fps=config.fps,
            max_width=config.max_width,
        )
    else:
        frame_paths, timestamps_s = extract_video_frames(
            video_path=config.video,
            output_dir=config.frames_dir,
            start_s=config.start_s,
            duration_s=config.duration_s,
            fps=config.fps,
            max_width=config.max_width,
        )
    if not frame_paths:
        raise RuntimeError("No frames extracted for Qwen annotation")

    if config.recover_from_raw:
        recover_checkpoint_from_raw(
            **context_options,
            **parser_options,
            raw_output_dir=config.raw_output_dir,
            checkpoint_path=config.checkpoint_json,
            video_path=config.video,
            prompt_path=config.prompt,
            model=config.model,
            sampling_fps=config.fps,
        )
        recover_batches_from_raw(
            **context_options,
            **parser_options,
            raw_output_dir=config.raw_output_dir,
            batch_output_dir=config.batch_output_dir,
            video_path=config.video,
            prompt_path=config.prompt,
            model=config.model,
            sampling_fps=config.fps,
        )
        write_chunk_annotations(
            batch_output_dir=config.batch_output_dir,
            chunk_output_dir=config.chunk_output_dir,
            batches_per_chunk=config.batches_per_chunk,
        )

    checkpoint_usage = {}
    if config.resume and config.checkpoint_json.exists():
        checkpoint_usage = json.loads(config.checkpoint_json.read_text(encoding="utf-8")).get("usage", {})
    all_frames: list[dict[str, Any]] = load_checkpoint_frames(config.checkpoint_json) if config.resume else []
    if task_metadata is not None:
        validate_task_objects(all_frames, task_metadata)
    completed_batches = completed_batch_indices(config.checkpoint_json) if config.resume else set()
    failed_batches: list[dict[str, Any]] = []
    usage_by_batch: dict[int, dict[str, Any]] = {}
    existing_batches = sorted(config.batch_output_dir.glob("batch_*.json")) if config.resume and config.batch_output_dir.exists() else []
    for batch_path in existing_batches:
        try:
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        usage_by_batch[int(batch_payload.get("batch_idx", len(usage_by_batch)))] = batch_payload.get("usage", {})

    batch_size = max(1, config.batch_size)
    for batch_idx, start in enumerate(range(0, len(frame_paths), batch_size)):
        if batch_idx in completed_batches:
            continue
        batch_paths = frame_paths[start : start + batch_size]
        batch_times = timestamps_s[start : start + batch_size]
        messages = build_qwen_frame_annotation_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frame_paths=batch_paths,
            timestamps_s=batch_times,
        )
        parsed = call_qwen_frame_annotations(
            **parser_options,
            messages=messages,
            model=config.model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            chat_template_kwargs=config.chat_template_kwargs,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            retry_sleep_s=config.retry_sleep_s,
        )
        (config.raw_output_dir / f"batch_{batch_idx:04d}.raw.txt").write_text(
            str(parsed.get("raw_response", "")),
            encoding="utf-8",
        )
        if parsed.get("status") != "ok":
            error_path = config.raw_output_dir / f"batch_{batch_idx:04d}.error.json"
            error_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
            if is_data_inspection_failed(parsed):
                parsed["frames"] = build_data_inspection_placeholder_frames(batch_times, parsed)
                parsed["status"] = "ok"
            else:
                failed_batches.append(
                    {
                        "batch_idx": int(batch_idx),
                        "time_range": {"start": float(min(batch_times)), "end": float(max(batch_times))},
                        "status": str(parsed.get("status") or "unknown_error"),
                        "error": str(parsed.get("error") or parsed.get("raw_response") or ""),
                        "error_json": str(error_path),
                        "retry_count": int(parsed.get("retry_count", 0) or 0),
                        "attempt_count": int(parsed.get("attempt_count", 1) or 1),
                    }
                )
                total_usage = merge_qwen_usage(list(usage_by_batch.values()) or [checkpoint_usage])
                write_checkpoint(
                    **context_options,
                    checkpoint_path=config.checkpoint_json,
                    frames=all_frames,
                    completed_batches=sorted(completed_batches),
                    video_path=config.video,
                    prompt_path=config.prompt,
                    model=config.model,
                    sampling_fps=config.fps,
                    usage=total_usage,
                    failed_batches=failed_batches,
                )
                continue
        write_batch_annotation(
            **context_options,
            batch_output_dir=config.batch_output_dir,
            batch_idx=batch_idx,
            frames=parsed["frames"],
            video_path=config.video,
            prompt_path=config.prompt,
            model=config.model,
            sampling_fps=config.fps,
            usage=parsed.get("usage", {}),
            retry_count=int(parsed.get("retry_count", 0) or 0),
            attempt_count=int(parsed.get("attempt_count", 1) or 1),
        )
        all_frames.extend(parsed["frames"])
        completed_batches.add(batch_idx)
        usage_by_batch[batch_idx] = parsed.get("usage", {})
        total_usage = merge_qwen_usage(list(usage_by_batch.values()) or [checkpoint_usage])
        write_checkpoint(
            **context_options,
            checkpoint_path=config.checkpoint_json,
            frames=all_frames,
            completed_batches=sorted(completed_batches),
            video_path=config.video,
            prompt_path=config.prompt,
            model=config.model,
            sampling_fps=config.fps,
            usage=total_usage,
            failed_batches=failed_batches,
        )
        write_chunk_annotations(
            batch_output_dir=config.batch_output_dir,
            chunk_output_dir=config.chunk_output_dir,
            batches_per_chunk=config.batches_per_chunk,
        )

    total_usage = merge_qwen_usage(list(usage_by_batch.values()) or [checkpoint_usage])
    payload = write_qwen_frame_annotation_output(
        **context_options,
        output_json=config.output_json,
        parsed_frames=all_frames,
        video_path=config.video,
        prompt_path=config.prompt,
        model=config.model,
        sampling_fps=config.fps,
        usage=total_usage,
        failed_batches=failed_batches,
    )
    return {
        "output_json": str(config.output_json),
        "frames_dir": str(config.frames_dir),
        "raw_output_dir": str(config.raw_output_dir),
        "checkpoint_json": str(config.checkpoint_json),
        "batch_output_dir": str(config.batch_output_dir),
        "chunk_output_dir": str(config.chunk_output_dir),
        "model": config.model,
        "frame_count": len(payload["frames"]),
        "completed_batch_count": len(completed_batches),
        "failed_batch_count": len(failed_batches),
        "status": payload.get("status"),
        "usage": total_usage,
        "max_retries": config.max_retries,
        "retry_sleep_s": config.retry_sleep_s,
    }
