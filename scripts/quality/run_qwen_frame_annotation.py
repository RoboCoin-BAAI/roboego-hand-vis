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
    extract_video_frames_from_windows,
    load_checkpoint_frames,
    read_prompt_markdown,
    recover_batches_from_raw,
    recover_checkpoint_from_raw,
    is_data_inspection_failed,
    merge_qwen_usage,
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
    parser.add_argument("--prompt", default="docs/prompts/qwen_frame_annotation_v5_en.md")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument("--windows_json", default="", help="Optional valid windows JSON; timestamps remain absolute to the source video.")
    parser.add_argument("--max_width", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep_s", type=float, default=2.0)
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
    if args.windows_json:
        windows_payload = json.loads(Path(args.windows_json).read_text(encoding="utf-8"))
        frame_paths, timestamps_s = extract_video_frames_from_windows(
            video_path=Path(args.video),
            output_dir=frames_dir,
            windows=windows_payload,
            fps=args.fps,
            max_width=args.max_width,
        )
    else:
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

    checkpoint_usage = {}
    if args.resume and checkpoint_json.exists():
        checkpoint_usage = json.loads(checkpoint_json.read_text(encoding="utf-8")).get("usage", {})
    all_frames: list[dict] = load_checkpoint_frames(checkpoint_json) if args.resume else []
    completed_batches = completed_batch_indices(checkpoint_json) if args.resume else set()
    failed_batches: list[dict] = []
    usage_by_batch: dict[int, dict] = {}
    for batch_path in sorted(batch_output_dir.glob("batch_*.json")) if args.resume and batch_output_dir.exists() else []:
        try:
            batch_payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        usage_by_batch[int(batch_payload.get("batch_idx", len(usage_by_batch)))] = batch_payload.get("usage", {})
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
            max_retries=args.max_retries,
            retry_sleep_s=args.retry_sleep_s,
        )
        (raw_output_dir / f"batch_{batch_idx:04d}.raw.txt").write_text(
            str(parsed.get("raw_response", "")),
            encoding="utf-8",
        )
        if parsed.get("status") != "ok":
            error_path = raw_output_dir / f"batch_{batch_idx:04d}.error.json"
            error_path.write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
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
                    checkpoint_path=checkpoint_json,
                    frames=all_frames,
                    completed_batches=sorted(completed_batches),
                    video_path=Path(args.video),
                    prompt_path=Path(args.prompt),
                    model=args.model,
                    sampling_fps=args.fps,
                    usage=total_usage,
                    failed_batches=failed_batches,
                )
                continue
        write_batch_annotation(
            batch_output_dir=batch_output_dir,
            batch_idx=batch_idx,
            frames=parsed["frames"],
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
            usage=parsed.get("usage", {}),
            retry_count=int(parsed.get("retry_count", 0) or 0),
            attempt_count=int(parsed.get("attempt_count", 1) or 1),
        )
        all_frames.extend(parsed["frames"])
        completed_batches.add(batch_idx)
        usage_by_batch[batch_idx] = parsed.get("usage", {})
        total_usage = merge_qwen_usage(list(usage_by_batch.values()) or [checkpoint_usage])
        write_checkpoint(
            checkpoint_path=checkpoint_json,
            frames=all_frames,
            completed_batches=sorted(completed_batches),
            video_path=Path(args.video),
            prompt_path=Path(args.prompt),
            model=args.model,
            sampling_fps=args.fps,
            usage=total_usage,
            failed_batches=failed_batches,
        )
        write_chunk_annotations(
            batch_output_dir=batch_output_dir,
            chunk_output_dir=chunk_output_dir,
            batches_per_chunk=args.batches_per_chunk,
        )

    total_usage = merge_qwen_usage(list(usage_by_batch.values()) or [checkpoint_usage])
    payload = write_qwen_frame_annotation_output(
        output_json=output_json,
        parsed_frames=all_frames,
        video_path=Path(args.video),
        prompt_path=Path(args.prompt),
        model=args.model,
        sampling_fps=args.fps,
        usage=total_usage,
        failed_batches=failed_batches,
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
        "failed_batch_count": len(failed_batches),
        "status": payload.get("status"),
        "usage": total_usage,
        "max_retries": args.max_retries,
        "retry_sleep_s": args.retry_sleep_s,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
