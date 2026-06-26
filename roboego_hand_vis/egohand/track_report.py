from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from roboego_hand_vis.egohand.tracking_interface import build_track_report_payload


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != "source_hand"}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_tracks_json(
    tracks: Dict[str, List[Optional[Dict[str, Any]]]],
    summary: Dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
    manifest_inputs: Any = None,
    video_path: Path | None = None,
    alignment_report: Path | None = None,
    depth_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    alignment_report = alignment_report or Path(args.alignment_report).expanduser().resolve()
    depth_dir = depth_dir or Path(args.depth_dir).expanduser().resolve()
    output_dir = output_dir or output_path.parent.resolve()
    parameters = {
        "start": args.start,
        "num_frames": args.num_frames,
        "max_interp_gap": args.max_interp_gap,
        "smooth_alpha": args.smooth_alpha,
        "jump_threshold_m": args.jump_threshold_m,
        "mesh_error_threshold_m": args.mesh_error_threshold_m,
        "min_valid_depth_ratio": args.min_valid_depth_ratio,
        "drop_unstable_recovered": args.drop_unstable_recovered,
        "unstable_recovered_jump_threshold_m": args.unstable_recovered_jump_threshold_m,
        "unstable_recovered_max_area_ratio": args.unstable_recovered_max_area_ratio,
        "apply_recovered_depth_overrides": args.apply_recovered_depth_overrides,
        "final_smoothing_window": args.final_smoothing_window,
    }
    payload = build_track_report_payload(
        inputs=manifest_inputs,
        alignment_report=alignment_report,
        depth_dir=depth_dir,
        output_dir=output_dir,
        parameters=parameters,
        summary=summary,
        tracks=jsonable(tracks),
        fallback_video=video_path or Path(args.video).expanduser().resolve(),
    )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_tracks_csv(tracks: Dict[str, List[Optional[Dict[str, Any]]]], output_path: Path, start: int) -> None:
    fieldnames = [
        "frame_idx",
        "side",
        "status",
        "candidate_id",
        "mesh_obj",
        "translation_x",
        "translation_y",
        "translation_z",
        "smoothed_translation_x",
        "smoothed_translation_y",
        "smoothed_translation_z",
        "raw_translation_jump_m",
        "smoothed_translation_jump_m",
        "mesh_to_depth_median_m",
        "depth_to_mesh_median_m",
        "anomaly_flags",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for side, track in tracks.items():
            for offset, obs in enumerate(track):
                frame_idx = start + offset
                if obs is None:
                    writer.writerow({"frame_idx": frame_idx, "side": side, "status": "missing"})
                    continue
                tr = obs["translation"]
                sm = obs["smoothed_translation"]
                writer.writerow(
                    {
                        "frame_idx": frame_idx,
                        "side": side,
                        "status": obs.get("status"),
                        "candidate_id": obs.get("candidate_id"),
                        "mesh_obj": obs.get("mesh_obj"),
                        "translation_x": tr[0],
                        "translation_y": tr[1],
                        "translation_z": tr[2],
                        "smoothed_translation_x": sm[0],
                        "smoothed_translation_y": sm[1],
                        "smoothed_translation_z": sm[2],
                        "raw_translation_jump_m": obs.get("raw_translation_jump_m"),
                        "smoothed_translation_jump_m": obs.get("smoothed_translation_jump_m"),
                        "mesh_to_depth_median_m": obs.get("mesh_to_depth_median_m"),
                        "depth_to_mesh_median_m": obs.get("depth_to_mesh_median_m"),
                        "anomaly_flags": ",".join(obs.get("anomaly_flags", [])),
                    }
                )
