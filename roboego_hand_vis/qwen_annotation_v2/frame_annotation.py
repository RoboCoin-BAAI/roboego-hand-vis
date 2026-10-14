"""V2 sampling/recovery; reuse unchanged Qwen HTTP, schema and output helpers."""

import json
import re
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .sampled_frames import sampled_frame_name
from .temporal_batches import select_target_frames
from roboego_hand_vis.qwen_annotation.frame_annotation import (
    build_data_inspection_placeholder_frames, build_qwen_frame_annotation_messages,
    call_qwen_frame_annotations, completed_batch_indices, is_data_inspection_failed,
    load_checkpoint_frames, merge_qwen_usage, read_prompt_markdown,
    parse_qwen_frame_annotation_response, write_batch_annotation as _legacy_batch, write_checkpoint,
    write_chunk_annotations as _legacy_chunks, write_qwen_frame_annotation_output as _legacy_output,
    _resize_max_width, _deduplicate_frames,
)

def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    start_s: float,
    duration_s: float | None,
    fps: float,
    max_width: int,
) -> tuple[list[Path], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / video_fps if video_fps > 0 else 0.0
    end_s = video_duration if duration_s is None else min(video_duration, start_s + duration_s)
    step = 1.0 / max(fps, 1e-6)
    times = list(np.arange(start_s, end_s, step))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    timestamps: list[float] = []
    for time_s in times:
        frame_idx = int(round(time_s * video_fps)) if video_fps > 0 else 0
        frame_idx = max(0, min(frame_count - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _resize_max_width(frame, max_width=max_width)
        timestamp = round(float(time_s), 6)
        path = output_dir / sampled_frame_name(timestamp)
        cv2.imwrite(str(path), frame)
        paths.append(path)
        timestamps.append(timestamp)
    cap.release()
    ordered = sorted(zip(timestamps, paths))
    return [path for _, path in ordered], [timestamp for timestamp, _ in ordered]

def extract_video_frames_from_windows(
    *,
    video_path: Path,
    output_dir: Path,
    windows: list[dict[str, Any]],
    fps: float,
    max_width: int,
) -> tuple[list[Path], list[float]]:
    """Extract frames from valid windows while keeping absolute video timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / video_fps if video_fps > 0 else 0.0
    step = 1.0 / max(fps, 1e-6)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    timestamps: list[float] = []
    seen_timestamps: set[float] = set()
    for window in sorted(windows, key=lambda item: float(item["start"])):
        start_s = max(0.0, float(window["start"]))
        end_s = min(video_duration, float(window["end"]))
        if end_s <= start_s:
            continue
        for time_s in np.arange(start_s, end_s, step):
            timestamp = round(float(time_s), 6)
            if timestamp in seen_timestamps:
                continue
            seen_timestamps.add(timestamp)
            frame_idx = int(round(timestamp * video_fps)) if video_fps > 0 else 0
            frame_idx = max(0, min(frame_count - 1, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_max_width(frame, max_width=max_width)
            path = output_dir / sampled_frame_name(timestamp)
            cv2.imwrite(str(path), frame)
            paths.append(path)
            timestamps.append(float(timestamp))
    cap.release()
    ordered = sorted(zip(timestamps, paths))
    return [path for _, path in ordered], [timestamp for timestamp, _ in ordered]

def recover_checkpoint_from_raw(
    *,
    raw_output_dir: Path,
    checkpoint_path: Path,
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
    task_metadata: dict[str, Any] | None = None,
    response_parser: Callable[[str], dict[str, Any]] | None = None,
    batch_times: dict[int, tuple[list[float], list[float]]] | None = None,
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    completed_batches: list[int] = []
    for raw_path in sorted(Path(raw_output_dir).glob("batch_*.raw.txt")):
        match = re.search(r"batch_(\d+)\.raw\.txt$", raw_path.name)
        if match is None:
            continue
        parsed = (response_parser or parse_qwen_frame_annotation_response)(raw_path.read_text(encoding="utf-8"))
        if parsed.get("status") != "ok":
            continue
        if batch_times is not None:
            idx = int(match.group(1))
            if idx not in batch_times:
                continue
            try:
                targets, allowed = batch_times[idx]
                parsed['frames'] = select_target_frames(parsed['frames'], targets, allowed)
            except ValueError:
                continue
        completed_batches.append(int(match.group(1)))
        frames.extend(parsed["frames"])
    return write_checkpoint(
        checkpoint_path=checkpoint_path,
        frames=_deduplicate_frames(frames),
        task_metadata=task_metadata,
        completed_batches=completed_batches,
        video_path=video_path,
        prompt_path=prompt_path,
        model=model,
        sampling_fps=sampling_fps,
    )

def recover_batches_from_raw(
    *,
    raw_output_dir: Path,
    batch_output_dir: Path,
    video_path: Path,
    prompt_path: Path,
    model: str,
    sampling_fps: float,
    task_metadata: dict[str, Any] | None = None,
    response_parser: Callable[[str], dict[str, Any]] | None = None,
    batch_times: dict[int, tuple[list[float], list[float]]] | None = None,
) -> list[Path]:
    outputs: list[Path] = []
    for raw_path in sorted(Path(raw_output_dir).glob("batch_*.raw.txt")):
        match = re.search(r"batch_(\d+)\.raw\.txt$", raw_path.name)
        if match is None:
            continue
        batch_idx = int(match.group(1))
        parsed = (response_parser or parse_qwen_frame_annotation_response)(raw_path.read_text(encoding="utf-8"))
        if parsed.get("status") != "ok":
            continue
        if batch_times is not None:
            idx = int(match.group(1))
            if idx not in batch_times:
                continue
            try:
                targets, allowed = batch_times[idx]
                parsed['frames'] = select_target_frames(parsed['frames'], targets, allowed)
            except ValueError:
                continue
        outputs.append(
            write_batch_annotation(
                batch_output_dir=batch_output_dir,
                batch_idx=batch_idx,
                frames=parsed["frames"],
                task_metadata=task_metadata,
                video_path=video_path,
                prompt_path=prompt_path,
                model=model,
                sampling_fps=sampling_fps,
            )
        )
    return outputs


def write_batch_annotation(**kwargs):
    destination = Path(kwargs['batch_output_dir'])
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.batch-write-', dir=destination) as directory:
        path = _legacy_batch(**(kwargs | {'batch_output_dir': Path(directory)}))
        final = destination / path.name
        os.replace(path, final)
    return final


def write_chunk_annotations(*, batch_output_dir, chunk_output_dir, batches_per_chunk):
    destination = Path(chunk_output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with tempfile.TemporaryDirectory(prefix='.chunk-write-', dir=destination) as directory:
        paths = _legacy_chunks(batch_output_dir=batch_output_dir, chunk_output_dir=Path(directory),
                               batches_per_chunk=batches_per_chunk)
        for path in paths:
            final = destination / path.name
            os.replace(path, final)
            prefix = path.name.split('_batches_', 1)[0] + '_'
            for stale in destination.glob(prefix + '*.json'):
                if stale != final:
                    stale.unlink()
            outputs.append(final)
    return outputs


def write_qwen_frame_annotation_output(**kwargs):
    destination = Path(kwargs['output_json'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.output-write-', dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        payload = _legacy_output(**(kwargs | {'output_json': temporary}))
        os.replace(temporary, destination)
    return payload
