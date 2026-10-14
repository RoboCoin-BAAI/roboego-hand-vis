from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_run import prepare_task_metadata
from .run_settings import bind_run_settings
from .grid_messages import build_messages
from .temporal_batches import batch_prompt, iter_results, plan_batches, select_target_frames
from roboego_hand_vis.qwen_annotation.task_context import context_prompts, parse_context_frame_response, validate_task_objects
from roboego_hand_vis.qwen_annotation_v2.frame_annotation import (
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
    parse_qwen_frame_annotation_response,
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
    overlap_frames: int = 2
    max_concurrency: int = 3
    image_mode: str = "individual"
    grid_columns: int = 4
    grid_cell_width: int = 512


def run_qwen_annotation(config: QwenAnnotationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    if (config.batch_size < 1 or config.overlap_frames < 0 or config.max_concurrency < 1
            or not math.isfinite(config.fps) or config.fps <= 0
            or config.image_mode not in ('individual', 'grid')
            or config.grid_columns < 1 or config.grid_cell_width < 1):
        raise ValueError('batch_size, max_concurrency and fps must be positive; overlap_frames must be nonnegative')
    bind_run_settings(config)
    system_prompt, user_prompt = read_prompt_markdown(config.prompt)
    system_prompt += "\nSampling may include fractional seconds. Use the exact supplied timestamps, not an assumed one-frame-per-second clock."
    task_metadata = prepare_task_metadata(config)
    context_options = {"task_metadata": task_metadata} if task_metadata is not None else {}
    parser_options = {}
    if task_metadata is not None:
        system_prompt, user_prompt = context_prompts(system_prompt, user_prompt, task_metadata)
        parser_options["response_parser"] = lambda text: parse_context_frame_response(text, task_metadata)
    config.raw_output_dir.mkdir(parents=True, exist_ok=True)
    extract_started = time.perf_counter()
    windows_payload = None
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

    extract_seconds = time.perf_counter() - extract_started
    plans = plan_batches(timestamps_s, config.batch_size, config.overlap_frames, config.fps, windows=windows_payload)
    batch_times = {plan.index: ([timestamps_s[i] for i in plan.targets],
                               [timestamps_s[i] for i in plan.context]) for plan in plans}

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
            batch_times=batch_times,
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
            batch_times=batch_times,
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
    available_batches = set()
    for batch_path in existing_batches:
        try:
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        idx = int(batch_payload['batch_idx'])
        if idx in available_batches or idx not in batch_times:
            raise ValueError('Batch artifacts differ from the current plan; use a fresh output directory')
        targets = batch_times[idx][0]
        select_target_frames(batch_payload['frames'], targets, targets)
        available_batches.add(idx)
        usage_by_batch[idx] = batch_payload.get('usage', {})

    if not completed_batches.issubset(batch_times):
        raise ValueError('Checkpoint batch indices differ; use a fresh output directory')
    completed_times = sorted(t for index in completed_batches for t in batch_times[index][0])
    if completed_batches:
        all_frames = select_target_frames(all_frames, completed_times, completed_times)
    elif all_frames:
        raise ValueError('Checkpoint frames have no completed batch ownership; use a fresh output directory')
    # A checkpoint may have survived while a batch file was lost; restore only that batch.
    missing_indices = completed_batches - available_batches
    missing_times = {t for idx in missing_indices for t in batch_times[idx][0]}
    all_frames = [frame for frame in all_frames if frame['t'] not in missing_times]
    completed_batches -= missing_indices
    pending_plans = [plan for plan in plans if plan.index not in completed_batches]
    parser = parser_options.get('response_parser', parse_qwen_frame_annotation_response)

    def request(plan):
        targets, allowed = batch_times[plan.index]
        context = [t for t in allowed if t not in targets]
        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=batch_prompt(user_prompt, targets, context),
            frame_paths=[frame_paths[i] for i in plan.context],
            timestamps_s=allowed, image_mode=config.image_mode,
            grid_columns=config.grid_columns, grid_cell_width=config.grid_cell_width,
        )
        def parse_response(text):
            parsed = parser(text)
            if parsed.get('status') == 'ok':
                try:
                    parsed['frames'] = select_target_frames(parsed['frames'], targets, allowed)
                except ValueError as exc:
                    return {'status': 'validation_failed', 'error': str(exc), 'raw_response': text}
            return parsed
        request_started = time.perf_counter()
        try:
            parsed = call_qwen_frame_annotations(
                response_parser=parse_response, messages=messages, model=config.model,
                api_key_env=config.api_key_env, base_url=config.base_url,
                chat_template_kwargs=config.chat_template_kwargs, timeout_s=config.timeout_s,
                max_retries=config.max_retries, retry_sleep_s=config.retry_sleep_s,
            )
        except Exception as exc:
            parsed = {'status': 'execution_failed', 'error': str(exc)}
        return parsed | {'request_seconds': time.perf_counter() - request_started}

    annotation_started = time.perf_counter()
    request_seconds = 0.0
    for plan, parsed in iter_results(pending_plans, config.max_concurrency, request):
        batch_idx = plan.index
        batch_times_target = batch_times[batch_idx][0]
        request_seconds += parsed['request_seconds']
        (config.raw_output_dir / f"batch_{batch_idx:04d}.raw.txt").write_text(
            str(parsed.get("raw_response", "")),
            encoding="utf-8",
        )
        if parsed.get("status") != "ok":
            error_path = config.raw_output_dir / f"batch_{batch_idx:04d}.error.json"
            error_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
            if is_data_inspection_failed(parsed):
                parsed["frames"] = build_data_inspection_placeholder_frames(batch_times_target, parsed)
                parsed["status"] = "ok"
            else:
                failed_batches.append(
                    {
                        "batch_idx": int(batch_idx),
                        "time_range": {"start": float(min(batch_times_target)), "end": float(max(batch_times_target))},
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

    # Rebuild derived chunks even when no requests were needed on this resume.
    write_chunk_annotations(batch_output_dir=config.batch_output_dir,
                            chunk_output_dir=config.chunk_output_dir,
                            batches_per_chunk=config.batches_per_chunk)
    all_frames.sort(key=lambda frame: float(frame["t"]))
    annotation_seconds = time.perf_counter() - annotation_started
    failed_batches.sort(key=lambda batch: batch["batch_idx"])
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
        "max_concurrency": config.max_concurrency,
        "overlap_frames": config.overlap_frames,
        "batch_size": config.batch_size,
        "image_mode": config.image_mode,
        "request_count": len(pending_plans),
        "timings": {"extract_seconds": round(extract_seconds, 3),
                    "annotation_seconds": round(annotation_seconds, 3),
                    "request_seconds_sum": round(request_seconds, 3),
                    "elapsed_seconds": round(time.perf_counter() - started, 3)},
    }
