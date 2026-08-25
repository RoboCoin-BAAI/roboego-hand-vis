from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from roboego_hand_vis.datasets.common import manifest_summary, probe_video
from roboego_hand_vis.quality.video_segments import cut_video


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
