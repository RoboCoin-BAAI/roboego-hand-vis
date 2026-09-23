from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoClipResult:
    input_path: Path
    output_path: Path
    start_frame: int
    end_frame: int
    source_frame_count: int
    written_frame_count: int
    fps: float
    width: int
    height: int


def clip_video_by_frame_range(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    overwrite: bool = False,
    fourcc: str = "mp4v",
) -> VideoClipResult:
    """Write frames in the half-open interval [start_frame, end_frame)."""
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")
    if len(fourcc) != 4:
        raise ValueError("fourcc must be exactly 4 characters")

    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output video already exists: {target}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    writer: cv2.VideoWriter | None = None
    try:
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_frame_count <= 0:
            raise ValueError(f"Input video has no frames: {source}")
        if start_frame >= source_frame_count:
            raise ValueError(
                f"start_frame {start_frame} is outside source frame count {source_frame_count}"
            )
        if end_frame > source_frame_count:
            raise ValueError(
                f"end_frame {end_frame} exceeds source frame count {source_frame_count}"
            )
        if fps <= 0:
            raise ValueError(f"Input video has invalid fps: {fps}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Input video has invalid resolution: {width}x{height}")

        target.parent.mkdir(parents=True, exist_ok=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {target}")

        written = 0
        for _frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame {start_frame + written} from {source}"
                )
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return VideoClipResult(
        input_path=source,
        output_path=target,
        start_frame=start_frame,
        end_frame=end_frame,
        source_frame_count=source_frame_count,
        written_frame_count=written,
        fps=fps,
        width=width,
        height=height,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip a video by frame interval [start_frame, end_frame)."
    )
    parser.add_argument("--input", required=True, help="输入视频路径。")
    parser.add_argument("--output", required=True, help="输出视频路径。")
    parser.add_argument("--start_frame", type=int, required=True, help="起始帧，包含。")
    parser.add_argument("--end_frame", type=int, required=True, help="结束帧，不包含。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出。")
    parser.add_argument("--fourcc", default="mp4v", help="输出视频编码 fourcc，默认 mp4v。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = clip_video_by_frame_range(
        args.input,
        args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        overwrite=args.overwrite,
        fourcc=args.fourcc,
    )
    print(
        f"Wrote {result.written_frame_count} frames to {result.output_path} "
        f"from [{result.start_frame}, {result.end_frame})."
    )


if __name__ == "__main__":
    main()
