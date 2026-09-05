#!/usr/bin/env python3
"""Repair duplicate timestamp runs reported by raw headband quality reports."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from roboego_hand_vis.quality.raw_headband_integrity import (
    assert_aligned_timestamps,
    parse_timestamp_file,
    summarize_timestamp_duplicates,
    TimestampSeries,
    write_timestamp_file,
)


def _timestamp_paths(report: dict, session_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for stream, value in (report.get("timestamps") or {}).items():
        paths[str(stream)] = Path(value)
    if paths:
        return paths
    for path in sorted((session_dir / "timestamps").glob("*_timestamps.txt")):
        stream = path.name.removesuffix("_timestamps.txt")
        paths[stream] = path
    return paths


def _copy_backup(path: Path, suffix: str) -> Path:
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


def repair_jump_duplicate_timestamps(series: TimestampSeries, fps: float) -> tuple[TimestampSeries, dict]:
    """Repair duplicate timestamps caused by an earlier skipped timestamp tick."""
    timestamps = np.asarray(series.timestamps, dtype=np.float64).copy()
    nominal_step = round(1.0 / float(fps), 4) if fps > 0 else 0.0333
    runs = []
    fallback_runs = []

    while True:
        duplicate_positions = np.where(np.diff(timestamps) == 0)[0] + 1
        if len(duplicate_positions) == 0:
            break
        duplicate_pos = int(duplicate_positions[0])
        diffs = np.diff(timestamps[: duplicate_pos + 1])
        gap_candidates = np.where(diffs > nominal_step * 1.5)[0] + 1
        if len(gap_candidates):
            gap_pos = int(gap_candidates[-1])
            original = timestamps.copy()
            timestamps[gap_pos + 1 : duplicate_pos + 1] = original[gap_pos:duplicate_pos]
            timestamps[gap_pos] = original[gap_pos] - nominal_step
            runs.append(
                {
                    "method": "shift_from_previous_large_gap",
                    "gap_index": int(series.indices[gap_pos]),
                    "duplicate_index": int(series.indices[duplicate_pos]),
                    "filled_timestamp": float(timestamps[gap_pos]),
                    "anchor_timestamp": float(original[duplicate_pos - 1]),
                }
            )
            continue

        run_start = duplicate_pos - 1
        run_end = duplicate_pos + 1
        while run_end < len(timestamps) and timestamps[run_end] == timestamps[run_start]:
            run_end += 1
        missing = run_end - run_start - 1
        window_start = max(0, run_start - missing)
        anchor_pos = run_end - 1
        anchor = float(timestamps[anchor_pos])
        for pos in range(window_start, anchor_pos + 1):
            timestamps[pos] = anchor - nominal_step * float(anchor_pos - pos)
        fallback_runs.append(
            {
                "method": "backfill_from_duplicate_run_anchor",
                "start_index": int(series.indices[window_start]),
                "end_index": int(series.indices[anchor_pos]),
                "frame_count": int(anchor_pos - window_start + 1),
            }
        )

    return (
        TimestampSeries(indices=series.indices.copy(), timestamps=timestamps),
        {
            "applied": bool(runs or fallback_runs),
            "method": "repair_duplicate_timestamps_as_skipped_ticks",
            "runs": runs,
            "fallback_runs": fallback_runs,
        },
    )


def repair_report(report_path: Path, backup_suffix: str, dry_run: bool) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    session_dir = Path(report.get("source_session") or report_path.parent.parent)
    timestamp_paths = _timestamp_paths(report, session_dir)
    series_by_stream = {stream: parse_timestamp_file(path) for stream, path in timestamp_paths.items()}
    duplicate_summaries = {
        stream: summarize_timestamp_duplicates(series) for stream, series in series_by_stream.items()
    }
    streams_to_repair = {
        stream for stream, summary in duplicate_summaries.items() if int(summary.get("duplicate_count", 0)) > 0
    }
    if not streams_to_repair:
        return {"report": str(report_path), "changed": False, "reason": "no_duplicate_streams"}

    before = {stream: summarize_timestamp_duplicates(series_by_stream[stream]) for stream in streams_to_repair}

    fps = float(report.get("fps") or (report.get("meta") or {}).get("fps") or 30.0)
    repaired_by_stream = dict(series_by_stream)

    changed_streams: dict[str, dict] = {}
    for stream in sorted(streams_to_repair):
        path = timestamp_paths.get(stream)
        series = series_by_stream.get(stream)
        if path is None or series is None:
            changed_streams[stream] = {"changed": False, "reason": "missing_path_or_repaired_series"}
            continue
        repaired, repair_summary = repair_jump_duplicate_timestamps(series, fps)
        repaired_by_stream[stream] = repaired
        after = summarize_timestamp_duplicates(repaired)
        changed = int(before[stream].get("duplicate_count", 0)) > 0 and int(after.get("duplicate_count", 0)) == 0
        changed_streams[stream] = {
            "changed": changed,
            "path": str(path),
            "before": before[stream],
            "after": after,
            "repair": repair_summary,
        }
        if changed and not dry_run:
            backup = _copy_backup(path, backup_suffix)
            write_timestamp_file(path, repaired)
            changed_streams[stream]["backup"] = str(backup)

    alignment_after = {}
    frame_window = report.get("frame_window") or {}
    try:
        start = int(frame_window.get("start", 0))
        num_frames = int(frame_window.get("num_frames") or min(len(s.timestamps) for s in repaired_by_stream.values()))
        tolerance_s = float(report.get("timestamp_tolerance_s") or 0.005)
        alignment_after = assert_aligned_timestamps(repaired_by_stream, "left", start, num_frames, tolerance_s)
    except Exception as exc:
        alignment_after = {"error": str(exc)}

    return {
        "report": str(report_path),
        "changed": any(item.get("changed") for item in changed_streams.values()),
        "streams": changed_streams,
        "alignment_after": alignment_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backup_suffix = ".bak_duplicate_repair_" + datetime.now().strftime("%Y%m%d%H%M%S")
    results = []
    for report_path in sorted(args.root.rglob("quality_report.json")):
        result = repair_report(report_path, backup_suffix, args.dry_run)
        if result.get("changed") or result.get("reason") != "no_duplicate_streams":
            results.append(result)

    changed_files = sum(
        1
        for result in results
        for stream in (result.get("streams") or {}).values()
        if stream.get("changed")
    )
    changed_reports = sum(1 for result in results if result.get("changed"))
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "reports_with_duplicates": len(results),
                "changed_reports": changed_reports,
                "changed_timestamp_files": changed_files,
                "backup_suffix": None if args.dry_run else backup_suffix,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
