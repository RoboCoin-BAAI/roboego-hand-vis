#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


def split_frame_side_by_side(frame):
    height, width = frame.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"Side-by-side stereo width must be even, got {width}")
    mid = width // 2
    return frame[:, :mid].copy(), frame[:, mid:].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a horizontal side-by-side stereo video into left/right videos.")
    parser.add_argument("--input", required=True, help="Input side-by-side stereo video.")
    parser.add_argument("--left_output", required=True, help="Output left-eye video.")
    parser.add_argument("--right_output", required=True, help="Output right-eye video.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc codec, default mp4v.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    left_output = Path(args.left_output).expanduser().resolve()
    right_output = Path(args.right_output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not args.overwrite:
        existing = [p for p in (left_output, right_output) if p.exists()]
        if existing:
            raise FileExistsError(f"Output exists; pass --overwrite: {existing}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width % 2 != 0:
        raise ValueError(f"Side-by-side stereo width must be even, got {width}")

    left_output.parent.mkdir(parents=True, exist_ok=True)
    right_output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    size = (width // 2, height)
    left_writer = cv2.VideoWriter(str(left_output), fourcc, fps, size)
    right_writer = cv2.VideoWriter(str(right_output), fourcc, fps, size)
    if not left_writer.isOpened() or not right_writer.isOpened():
        raise RuntimeError(f"Failed to open video writers: {left_output}, {right_output}")

    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            left, right = split_frame_side_by_side(frame)
            left_writer.write(left)
            right_writer.write(right)
            written += 1
    finally:
        cap.release()
        left_writer.release()
        right_writer.release()

    summary = {
        "input": str(input_path),
        "left_output": str(left_output),
        "right_output": str(right_output),
        "input_width": width,
        "input_height": height,
        "output_width": size[0],
        "output_height": size[1],
        "fps": fps,
        "input_frame_count": frame_count,
        "written_frame_count": written,
        "codec": args.codec,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
