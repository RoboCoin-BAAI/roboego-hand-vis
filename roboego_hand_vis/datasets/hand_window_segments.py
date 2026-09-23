from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from roboego_hand_vis.common.video_paths import find_named_video
from roboego_hand_vis.datasets.common import manifest_summary, probe_video
from roboego_hand_vis.datasets.video_segments import (
    DEFAULT_CAMERA_STREAMS,
    cut_video,
    _copy_tree_if_exists,
    _session_fps,
    _write_segment_imu,
    _write_segment_timestamps,
)


def export_hand_window_segments(
    *,
    segments_root: Path,
    hand_windows_json: Path,
    overwrite: bool = False,
    codec: str = "h264",
) -> dict[str, Any]:
    segments_root = Path(segments_root).resolve()
    windows_path = Path(hand_windows_json).resolve()
    payload = json.loads(windows_path.read_text(encoding="utf-8"))
    exported: list[dict[str, Any]] = []
    for window in payload.get("windows", []):
        segment_dir = _find_segment_dir(segments_root, str(window["source_segment_id"]))
        if segment_dir is None:
            continue
        exported.append(
            export_hand_window_segment(
                segment_dir=segment_dir,
                window=window,
                overwrite=overwrite,
                codec=codec,
            )
        )
    return {
        "stage": "export_hand_window_segments",
        "segments_root": str(segments_root),
        "source_hand_windows": str(windows_path),
        "window_count": len(exported),
        "windows": exported,
    }


def export_direct_hand_window_segments(
    *,
    session_dir: Path,
    hand_windows_json: Path,
    output_dir: Path,
    camera_streams: tuple[str, ...] = DEFAULT_CAMERA_STREAMS,
    fps: float | None = None,
    overwrite: bool = False,
    codec: str = "h264",
) -> dict[str, Any]:
    session_dir = Path(session_dir).resolve()
    windows_path = Path(hand_windows_json).resolve()
    output_dir = Path(output_dir).resolve()
    payload = json.loads(windows_path.read_text(encoding="utf-8"))
    fps_value = float(fps if fps is not None else _session_fps(session_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    for window in payload.get("windows", []):
        exported.append(
            export_direct_hand_window_segment(
                session_dir=session_dir,
                output_dir=output_dir,
                window=window,
                camera_streams=camera_streams,
                fps=fps_value,
                overwrite=overwrite,
                codec=codec,
            )
        )
    summary = {
        "stage": "export_direct_hand_window_segments",
        "source_session": str(session_dir),
        "source_hand_windows": str(windows_path),
        "output_dir": str(output_dir),
        "fps": fps_value,
        "camera_streams": list(camera_streams),
        "window_count": len(exported),
        "windows": exported,
    }
    (output_dir / "direct_hand_window_segments_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def export_direct_hand_window_segment(
    *,
    session_dir: Path,
    output_dir: Path,
    window: dict[str, Any],
    camera_streams: tuple[str, ...] = DEFAULT_CAMERA_STREAMS,
    fps: float,
    overwrite: bool = False,
    codec: str = "h264",
) -> dict[str, Any]:
    start_frame = int(round(float(window["start"]) * float(fps)))
    end_frame_exclusive = int(round((float(window["end"]) + 1.0) * float(fps)))
    frame_count = max(0, end_frame_exclusive - start_frame)
    window_dir = Path(output_dir) / direct_hand_window_dir_name(window)
    if window_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Hand window directory exists; pass overwrite=True: {window_dir}")
        shutil.rmtree(window_dir)
    (window_dir / "videos").mkdir(parents=True, exist_ok=True)
    (window_dir / "annotation").mkdir(parents=True, exist_ok=True)

    videos: dict[str, str] = {}
    for stream in camera_streams:
        source = find_named_video(Path(session_dir) / "videos", stream)
        target = window_dir / "videos" / f"{stream}.mp4"
        cut_video(
            source_video=source,
            output_video=target,
            start_frame=start_frame,
            frame_count=frame_count,
            fps=float(fps),
            codec=codec,
        )
        videos[stream] = str(target)

    time_bounds = _write_segment_timestamps(
        session_dir=Path(session_dir),
        segment_dir=window_dir,
        camera_streams=camera_streams,
        start_frame=start_frame,
        frame_count=frame_count,
    )
    _write_segment_imu(session_dir=Path(session_dir), segment_dir=window_dir, time_bounds=time_bounds)
    _copy_tree_if_exists(Path(session_dir) / "calibrations", window_dir / "calibrations")
    window_payload = {
        **window,
        "source_session": str(Path(session_dir).resolve()),
        "source_hand_window_dir": str(window_dir.resolve()),
        "source_start_frame": start_frame,
        "segment_frame_count": frame_count,
        "local_frame_range": {"start": 0, "end": frame_count - 1, "num_frames": frame_count},
        "fps": float(fps),
        "videos": videos,
    }
    (window_dir / "annotation" / "window.json").write_text(
        json.dumps(window_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "window_id": window["window_id"],
        "window_dir": str(window_dir),
        "start": int(window["start"]),
        "end": int(window["end"]),
        "source_start_frame": start_frame,
        "frame_count": frame_count,
        "videos": videos,
    }


def export_hand_window_segment(
    *,
    segment_dir: Path,
    window: dict[str, Any],
    overwrite: bool = False,
    codec: str = "h264",
) -> dict[str, Any]:
    segment_dir = Path(segment_dir).resolve()
    frame_range = window["segment_frame_range"]
    start_frame = int(frame_range["start"])
    end_frame = int(frame_range["end"])
    frame_count = int(frame_range.get("num_frames") or max(0, end_frame - start_frame + 1))
    window_dir = segment_dir / "hand_windows" / hand_window_dir_name(window)
    if window_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Hand window directory exists; pass overwrite=True: {window_dir}")
        shutil.rmtree(window_dir)
    (window_dir / "videos").mkdir(parents=True, exist_ok=True)
    (window_dir / "annotation").mkdir(parents=True, exist_ok=True)
    (window_dir / "timestamps").mkdir(parents=True, exist_ok=True)

    rectified_manifest_path = segment_dir / "rectified" / "dataset_manifest" / "dataset_manifest.json"
    manifest = json.loads(rectified_manifest_path.read_text(encoding="utf-8"))
    fps = float(manifest["videos"]["left"].get("fps") or 30.0)

    left_out = window_dir / "videos" / "left_rectified.mp4"
    right_out = window_dir / "videos" / "right_rectified.mp4"
    cut_video(
        source_video=Path(manifest["videos"]["left"]["path"]),
        output_video=left_out,
        start_frame=start_frame,
        frame_count=frame_count,
        fps=fps,
        codec=codec,
    )
    cut_video(
        source_video=Path(manifest["videos"]["right"]["path"]),
        output_video=right_out,
        start_frame=start_frame,
        frame_count=frame_count,
        fps=fps,
        codec=codec,
    )
    _write_window_timestamps(segment_dir=segment_dir, window_dir=window_dir, start_frame=start_frame, frame_count=frame_count)
    window_payload = {
        **window,
        "source_segment_dir": str(segment_dir),
        "source_rectified_manifest": str(rectified_manifest_path),
        "local_frame_range": {"start": 0, "end": frame_count - 1, "num_frames": frame_count},
    }
    (window_dir / "annotation" / "window.json").write_text(
        json.dumps(window_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_out = _write_window_manifest(
        source_manifest=manifest,
        window=window_payload,
        window_dir=window_dir,
        left_video=left_out,
        right_video=right_out,
    )
    return {
        "window_id": window["window_id"],
        "source_segment_id": window["source_segment_id"],
        "window_dir": str(window_dir),
        "manifest": str(manifest_out),
        "left_video": str(left_out),
        "right_video": str(right_out),
        "frame_count": frame_count,
    }


def hand_window_dir_name(window: dict[str, Any]) -> str:
    return f"{window['window_id']}_t{int(window['start']):06d}_{int(window['end']):06d}"


def direct_hand_window_dir_name(window: dict[str, Any]) -> str:
    label = str(window.get("semantic_label") or "hand_window").replace(" ", "_")
    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in label).strip("_")
    return f"{hand_window_dir_name(window)}_{safe_label or 'hand_window'}"


def _find_segment_dir(segments_root: Path, segment_id: str) -> Path | None:
    matches = sorted(path for path in segments_root.iterdir() if path.is_dir() and path.name.startswith(f"{segment_id}_"))
    return matches[0] if matches else None


def _write_window_timestamps(*, segment_dir: Path, window_dir: Path, start_frame: int, frame_count: int) -> None:
    source_dir = segment_dir / "timestamps"
    if not source_dir.exists():
        return
    for source in source_dir.glob("*_timestamps.txt"):
        rows = _read_timestamp_rows(source)
        selected = rows[start_frame : start_frame + frame_count]
        target = window_dir / "timestamps" / source.name
        with target.open("w", encoding="utf-8") as handle:
            for idx, timestamp in enumerate(selected):
                handle.write(f"{idx} {timestamp}\n")


def _read_timestamp_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            rows.append(parts[1])
    return rows


def _write_window_manifest(
    *,
    source_manifest: dict[str, Any],
    window: dict[str, Any],
    window_dir: Path,
    left_video: Path,
    right_video: Path,
) -> Path:
    out = copy.deepcopy(source_manifest)
    out["sequence_name"] = f"{source_manifest.get('sequence_name', 'segment')}_{window['window_id']}"
    out["root_dir"] = str(window_dir)
    out["videos"]["left"] = probe_video(left_video)
    out["videos"]["right"] = probe_video(right_video)
    out["metadata"] = copy.deepcopy(source_manifest.get("metadata", {}))
    out["metadata"]["hand_window"] = {
        "window_id": window["window_id"],
        "source_segment_id": window["source_segment_id"],
        "time_s": {"start": window.get("start"), "end": window.get("end")},
        "segment_frame_range": window["segment_frame_range"],
        "local_frame_range": window["local_frame_range"],
        "source_manifest": window["source_rectified_manifest"],
        "actions": window.get("actions", []),
        "objects": window.get("objects", []),
    }
    output_dir = window_dir / "dataset_manifest"
    output_dir.mkdir(parents=True, exist_ok=True)
    out["outputs"] = {
        "manifest_path": str(output_dir / "dataset_manifest.json"),
        "summary_path": str(output_dir / "dataset_summary.txt"),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "dataset_summary.txt"
    manifest_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(manifest_summary(out), encoding="utf-8")
    return manifest_path
