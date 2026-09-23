#!/usr/bin/env python3
"""Repair the remaining right timestamp alignment failures after duplicate repair."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.raw_headband_integrity.logic import (  # noqa: E402
    TimestampSeries,
    assert_aligned_timestamps,
    parse_timestamp_file,
    write_timestamp_file,
)


ROOT = Path("/mnt/nas/synnas/ego/baai_ego_task")
TOLERANCE_S = 0.005


def _backup(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.name + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        return backup
    index = 1
    while True:
        candidate = path.with_name(path.name + f"{suffix}.{index}")
        if not candidate.exists():
            shutil.copy2(path, candidate)
            return candidate
        index += 1


def _right_alignment_sessions() -> list[Path]:
    sessions: list[Path] = []
    for report_path in sorted(ROOT.rglob("quality_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        gate = report.get("raw_headband_quality_gate") or {}
        for check in gate.get("checks") or []:
            if not isinstance(check, dict):
                continue
            if check.get("name") == "timestamp_alignment" and check.get("stream") == "right":
                if check.get("passed") is False or check.get("level") == "fail":
                    sessions.append(Path(report.get("source_session") or report_path.parent.parent))
                    break
    return sessions


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _renumber(series: TimestampSeries) -> TimestampSeries:
    return TimestampSeries(indices=np.arange(len(series.timestamps), dtype=np.int64), timestamps=series.timestamps.copy())


def _is_right_leading_one_frame(left: TimestampSeries, right: TimestampSeries) -> bool:
    if len(right.timestamps) != len(left.timestamps) + 1:
        return False
    deltas = np.abs(right.timestamps[1:] - left.timestamps)
    return bool(len(deltas) and float(np.max(deltas)) <= TOLERANCE_S)


def _trim_right_first_frame_video(video_path: Path, suffix: str) -> dict:
    before_count = _video_frame_count(video_path)
    backup = _backup(video_path, suffix)
    tmp = video_path.with_name(video_path.stem + ".drop_first.tmp.mp4")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(backup),
        "-vf",
        "select='gte(n,1)',setpts=N/FRAME_RATE/TB",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(video_path)
    after_count = _video_frame_count(video_path)
    return {"backup": str(backup), "before_count": before_count, "after_count": after_count}


def _update_json_file(path: Path, suffix: str, updater) -> str | None:
    if not path.exists():
        return None
    backup = _backup(path, suffix)
    payload = json.loads(path.read_text(encoding="utf-8"))
    updater(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(backup)


def _update_meta_for_right_trim(session: Path, frame_count: int, suffix: str) -> str | None:
    def updater(meta: dict) -> None:
        topics = meta.get("topics") or {}
        if "right" in topics:
            topics["right"]["written_frames"] = frame_count
        meta["synced_frames"] = frame_count
        meta["total_written_frames"] = frame_count
        meta["duration_sec"] = frame_count / float(meta.get("fps") or 30)
        meta["span_sec"] = max((frame_count - 1) / float(meta.get("fps") or 30), 0)
        meta["avg_fps_per_cam"] = frame_count / meta["span_sec"] if meta["span_sec"] else 0

    return _update_json_file(session / "meta.json", suffix, updater)


def _update_cut_info_for_right_trim(session: Path, frame_count: int, suffix: str) -> str | None:
    def updater(cut_info: dict) -> None:
        if "timestamp_counts" in cut_info and "right" in cut_info["timestamp_counts"]:
            cut_info["timestamp_counts"]["right"] = frame_count
        if "expected_frame_counts" in cut_info and "right" in cut_info["expected_frame_counts"]:
            cut_info["expected_frame_counts"]["right"] = frame_count
        if "frame_ranges" in cut_info and "right" in cut_info["frame_ranges"]:
            right_range = cut_info["frame_ranges"]["right"]
            if isinstance(right_range, list) and len(right_range) == 2:
                right_range[0] = int(right_range[0]) + 1
        cut_info["expected_frames"] = frame_count

    return _update_json_file(session / "cut_info.json", suffix, updater)


def _repair_local_right_to_left(session: Path, left: TimestampSeries, right: TimestampSeries, suffix: str) -> dict:
    n = min(len(left.timestamps), len(right.timestamps))
    deltas = np.abs(right.timestamps[:n] - left.timestamps[:n])
    over = np.where(deltas > TOLERANCE_S)[0]
    repaired = TimestampSeries(indices=right.indices.copy(), timestamps=right.timestamps.copy())
    repaired.timestamps[over] = left.timestamps[over]
    timestamp_path = session / "timestamps" / "right_timestamps.txt"
    backup = _backup(timestamp_path, suffix)
    write_timestamp_file(timestamp_path, repaired)
    return {"strategy": "copy_left_timestamp_at_over_tolerance_indices", "backup": str(backup), "indices": over.tolist()}


def main() -> int:
    suffix = ".bak_alignment_repair_" + datetime.now().strftime("%Y%m%d%H%M%S")
    results = []
    for session in _right_alignment_sessions():
        left = parse_timestamp_file(session / "timestamps" / "left_timestamps.txt")
        right = parse_timestamp_file(session / "timestamps" / "right_timestamps.txt")
        if _is_right_leading_one_frame(left, right):
            timestamp_path = session / "timestamps" / "right_timestamps.txt"
            timestamp_backup = _backup(timestamp_path, suffix)
            trimmed = _renumber(TimestampSeries(indices=right.indices[1:].copy(), timestamps=right.timestamps[1:].copy()))
            write_timestamp_file(timestamp_path, trimmed)
            video_result = _trim_right_first_frame_video(session / "videos" / "right.mp4", suffix)
            meta_backup = _update_meta_for_right_trim(session, len(trimmed.timestamps), suffix)
            cut_info_backup = _update_cut_info_for_right_trim(session, len(trimmed.timestamps), suffix)
            repaired_right = trimmed
            strategy = "drop_right_first_frame"
            extra = {
                "timestamp_backup": str(timestamp_backup),
                "video": video_result,
                "meta_backup": meta_backup,
                "cut_info_backup": cut_info_backup,
            }
        else:
            extra = _repair_local_right_to_left(session, left, right, suffix)
            repaired_right = parse_timestamp_file(session / "timestamps" / "right_timestamps.txt")
            strategy = extra["strategy"]

        alignment = assert_aligned_timestamps(
            {"left": left, "right": repaired_right},
            "left",
            0,
            min(len(left.timestamps), len(repaired_right.timestamps)),
            TOLERANCE_S,
        )
        results.append(
            {
                "session": str(session),
                "strategy": strategy,
                "left_timestamps": len(left.timestamps),
                "right_timestamps": len(repaired_right.timestamps),
                "right_alignment": alignment["right"],
                **extra,
            }
        )

    print(json.dumps({"backup_suffix": suffix, "session_count": len(results), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
