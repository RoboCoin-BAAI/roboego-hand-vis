from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2


DEFAULT_CAMERA_STREAMS = ("left", "right")


def export_selected_video_segments(
    *,
    session_dir: Path,
    processing_plan_json: Path,
    output_dir: Path,
    camera_streams: tuple[str, ...] = DEFAULT_CAMERA_STREAMS,
    fps: float | None = None,
    overwrite: bool = False,
    codec: str = "h264",
) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    processing_plan_json = Path(processing_plan_json).resolve()
    output_dir = Path(output_dir).resolve()
    plan = json.loads(processing_plan_json.read_text(encoding="utf-8"))
    fps_value = float(fps if fps is not None else _session_fps(session_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for segment in plan.get("segments", []):
        segment_dir = output_dir / segment_dir_name(segment)
        if segment_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Segment directory exists; pass overwrite=True: {segment_dir}")
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True)

        start_frame = int(round(float(segment["start"]) * fps_value))
        end_frame_exclusive = int(round((float(segment["end"]) + 1.0) * fps_value))
        frame_count = max(0, end_frame_exclusive - start_frame)
        videos: dict[str, str] = {}
        for stream in camera_streams:
            source = session_dir / "videos" / f"{stream}.mp4"
            target = segment_dir / f"{stream}.mp4"
            cut_video(
                source_video=source,
                output_video=target,
                start_frame=start_frame,
                frame_count=frame_count,
                fps=fps_value,
                codec=codec,
            )
            videos[stream] = str(target)

        segment_payload = {
            **segment,
            "videos": videos,
            "source_session": str(session_dir),
            "source_processing_plan": str(processing_plan_json),
            "source_start_frame": start_frame,
            "segment_frame_count": frame_count,
            "fps": fps_value,
        }
        (segment_dir / "segment.json").write_text(
            json.dumps(segment_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        exported.append(
            {
                "segment_id": segment.get("segment_id"),
                "segment_dir": str(segment_dir),
                "start": int(segment["start"]),
                "end": int(segment["end"]),
                "source_start_frame": start_frame,
                "segment_frame_count": frame_count,
                "videos": videos,
            }
        )

    summary = {
        "source_session": str(session_dir),
        "source_processing_plan": str(processing_plan_json),
        "output_dir": str(output_dir),
        "fps": fps_value,
        "camera_streams": list(camera_streams),
        "segment_count": len(exported),
        "segments": exported,
    }
    (output_dir / "selected_video_segments_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def segment_dir_name(segment: dict[str, Any]) -> str:
    segment_id = str(segment.get("segment_id", "seg_0000"))
    start = int(segment["start"])
    end = int(segment["end"])
    label = _safe_label(str(segment.get("semantic_label") or "segment"))
    return f"{segment_id}_t{start:06d}_{end:06d}_{label}"


def cut_video(
    *,
    source_video: Path,
    output_video: Path,
    start_frame: int,
    frame_count: int,
    fps: float,
    codec: str = "h264",
) -> dict[str, Any]:
    source_video = Path(source_video)
    if not source_video.exists():
        raise FileNotFoundError(f"Missing source video: {source_video}")
    if codec.lower() in {"h264", "avc1", "libx264"} and shutil.which("ffmpeg"):
        return _cut_video_ffmpeg_h264(
            source_video=source_video,
            output_video=output_video,
            start_frame=start_frame,
            frame_count=frame_count,
            fps=fps,
        )
    if codec.lower() in {"h264", "avc1", "libx264"}:
        codec = "mp4v"
    cap = cv2.VideoCapture(str(source_video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source video: {source_video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video size for {source_video}: {width}x{height}")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            (width, height),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open output video writer: {output_video}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            written = 0
            for _ in range(int(frame_count)):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
    finally:
        cap.release()
    return {
        "source_video": str(source_video),
        "output_video": str(output_video),
        "start_frame": int(start_frame),
        "requested_frame_count": int(frame_count),
        "written_frame_count": int(written),
        "fps": float(fps),
        "codec": codec,
        "encoder": "opencv",
    }


def _cut_video_ffmpeg_h264(
    *,
    source_video: Path,
    output_video: Path,
    start_frame: int,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    start_seconds = float(start_frame) / float(fps)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        str(source_video),
        "-frames:v",
        str(int(frame_count)),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(command, check=True)
    return {
        "source_video": str(source_video),
        "output_video": str(output_video),
        "start_frame": int(start_frame),
        "requested_frame_count": int(frame_count),
        "written_frame_count": _video_frame_count(output_video),
        "fps": float(fps),
        "codec": "h264",
        "encoder": "ffmpeg",
    }


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _session_fps(session_dir: Path) -> float:
    meta_path = Path(session_dir) / "meta.json"
    if not meta_path.exists():
        return 30.0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return float(meta.get("fps", 30.0))


def _safe_label(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip()]
    label = "".join(chars).strip("_")
    while "__" in label:
        label = label.replace("__", "_")
    return label or "segment"
