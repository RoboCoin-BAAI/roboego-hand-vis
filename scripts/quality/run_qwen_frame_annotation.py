#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.quality.qwen_frame_annotation import (
    build_qwen_frame_annotation_messages,
    build_data_inspection_placeholder_frames,
    call_qwen_frame_annotations,
    completed_batch_indices,
    extract_video_frames,
    load_checkpoint_frames,
    read_prompt_markdown,
    recover_batches_from_raw,
    recover_checkpoint_from_raw,
    is_data_inspection_failed,
    write_batch_annotation,
    write_checkpoint,
    write_chunk_annotations,
    write_qwen_frame_annotation_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen V4 ego frame annotation on 1 FPS sampled video frames.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--frames_dir", default="")
    parser.add_argument("--prompt", default="docs/prompts/qwen_frame_annotation_v4.md")
    parser.add_argument("--model", default="qwen-vl-plus")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--max_width", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--raw_output_dir", default="")
    parser.add_argument("--checkpoint_json", default="")
    parser.add_argument("--batch_output_dir", default="")
    parser.add_argument("--chunk_output_dir", default="")
    parser.add_argument("--batches_per_chunk", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover_from_raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    frames_dir = Path(args.frames_dir) if args.frames_dir else output_json.parent / "qwen_frame_annotation_frames"
    raw_output_dir = Path(args.raw_output_dir) if args.raw_output_dir else output_json.parent / "qwen_frame_annotation_raw"
    checkpoint_json = (
        Path(args.checkpoint_json)
        if args.checkpoint_json
        else output_json.with_suffix(output_json.suffix + ".checkpoint.json")
    )
    batch_output_dir = Path(args.batch_output_dir) if args.batch_output_dir else output_json.parent / "qwen_frame_annotation_batches"
    chunk_output_dir = Path(args.chunk_output_dir) if args.chunk_output_dir else output_json.parent / "qwen_frame_annotation_chunks"
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    system_prompt, user_prompt = read_prompt_markdown(Path(args.prompt))
    frame_paths, timestamps_s = extract_video_frames(
        video_path=Path(args.video),
        output_dir=frames_dir,
        start_s=args.start_s,
        duration_s=args.duration_s,
        fps=args.fps,
        max_width=args.max_width,
    )
    if not frame_paths:
        raise RuntimeError("No frames extracted for Qwen annotation")

    if args.recover_from_raw:
        recover_checkpoint_from_raw(
            raw_output_dir=raw_output_dir,
            checkpoint_path=checkpoint_json,
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
        )
        recover_batches_from_raw(
            raw_output_dir=raw_output_dir,
            batch_output_dir=batch_output_dir,
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
        )
        write_chunk_annotations(
            batch_output_dir=batch_output_dir,
            chunk_output_dir=chunk_output_dir,
            batches_per_chunk=args.batches_per_chunk,
        )

    all_frames: list[dict] = load_checkpoint_frames(checkpoint_json) if args.resume else []
    completed_batches = completed_batch_indices(checkpoint_json) if args.resume else set()
    batch_size = max(1, args.batch_size)
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
            messages=messages,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout_s=args.timeout_s,
        )
        (raw_output_dir / f"batch_{batch_idx:04d}.raw.txt").write_text(
            str(parsed.get("raw_response", "")),
            encoding="utf-8",
        )
        if parsed.get("status") != "ok":
            (raw_output_dir / f"batch_{batch_idx:04d}.error.json").write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if is_data_inspection_failed(parsed):
                parsed["frames"] = build_data_inspection_placeholder_frames(batch_times, parsed)
                parsed["status"] = "ok"
            else:
                raise RuntimeError(f"Qwen annotation failed for batch {batch_idx}: {parsed.get('status')}")
        write_batch_annotation(
            batch_output_dir=batch_output_dir,
            batch_idx=batch_idx,
            frames=parsed["frames"],
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
        )
        all_frames.extend(parsed["frames"])
        completed_batches.add(batch_idx)
        write_checkpoint(
            checkpoint_path=checkpoint_json,
            frames=all_frames,
            completed_batches=sorted(completed_batches),
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
        )
        write_chunk_annotations(
            batch_output_dir=batch_output_dir,
            chunk_output_dir=chunk_output_dir,
            batches_per_chunk=args.batches_per_chunk,
        )

    payload = write_qwen_frame_annotation_output(
        output_json=output_json,
        parsed_frames=all_frames,
        video_path=Path(args.video),
        prompt_path=Path(args.prompt),
        model=args.model,
        sampling_fps=args.fps,
    )
    summary = {
        "output_json": str(output_json),
        "frames_dir": str(frames_dir),
        "raw_output_dir": str(raw_output_dir),
        "checkpoint_json": str(checkpoint_json),
        "batch_output_dir": str(batch_output_dir),
        "chunk_output_dir": str(chunk_output_dir),
        "model": args.model,
        "frame_count": len(payload["frames"]),
        "completed_batch_count": len(completed_batches),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
