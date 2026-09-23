from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboego_hand_vis.common.depth_io import find_depth_path_for_frame, load_depth_frame


@dataclass(frozen=True)
class DepthNumericQualityConfig:
    start: int = 0
    num_frames: int = 300
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    min_valid_depth_ratio: float = 0.5
    min_depth_std_m: float = 0.02
    min_depth_iqr_m: float = 0.03
    max_near_ratio: float = 0.8
    max_far_ratio: float = 0.8
    edge_band_m: float = 0.05
    max_median_delta_m: float = 0.5
    min_valid_mask_iou: float = 0.5


def _depth_stats(depth_m: np.ndarray, config: DepthNumericQualityConfig) -> dict[str, Any]:
    finite = np.isfinite(depth_m)
    valid = finite & (depth_m >= config.min_depth_m) & (depth_m <= config.max_depth_m)
    values = depth_m[valid]
    if values.size == 0:
        return {
            "valid_depth_ratio": 0.0,
            "depth_min_m": None,
            "depth_p05_m": None,
            "depth_median_m": None,
            "depth_p95_m": None,
            "depth_max_m": None,
            "depth_std_m": None,
            "depth_iqr_m": None,
            "near_ratio": 0.0,
            "far_ratio": 0.0,
            "valid_mask": valid,
        }

    p05, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    near_threshold = config.min_depth_m + config.edge_band_m
    far_threshold = config.max_depth_m - config.edge_band_m
    return {
        "valid_depth_ratio": float(np.mean(valid)),
        "depth_min_m": float(np.min(values)),
        "depth_p05_m": float(p05),
        "depth_median_m": float(median),
        "depth_p95_m": float(p95),
        "depth_max_m": float(np.max(values)),
        "depth_std_m": float(np.std(values)),
        "depth_iqr_m": float(p75 - p25),
        "near_ratio": float(np.mean(values <= near_threshold)),
        "far_ratio": float(np.mean(values >= far_threshold)),
        "valid_mask": valid,
    }


def _valid_mask_iou(current: np.ndarray | None, previous: np.ndarray | None) -> float | None:
    if current is None or previous is None or current.shape != previous.shape:
        return None
    union = current | previous
    if not union.any():
        return None
    return float(np.mean(current & previous) / np.mean(union))


def _frame_warnings(frame: dict[str, Any], config: DepthNumericQualityConfig) -> list[str]:
    warnings: list[str] = []
    if frame.get("missing_depth"):
        warnings.append("missing_depth")
        return warnings
    if frame["valid_depth_ratio"] < config.min_valid_depth_ratio:
        warnings.append("low_valid_depth_ratio")
    if frame["depth_median_m"] is None:
        warnings.append("missing_valid_depth")
    if (frame["depth_std_m"] is not None and frame["depth_std_m"] < config.min_depth_std_m) or (
        frame["depth_iqr_m"] is not None and frame["depth_iqr_m"] < config.min_depth_iqr_m
    ):
        warnings.append("flat_depth_distribution")
    if frame["near_ratio"] > config.max_near_ratio:
        warnings.append("near_depth_saturation")
    if frame["far_ratio"] > config.max_far_ratio:
        warnings.append("far_depth_saturation")
    if frame.get("median_depth_delta_m") is not None and frame["median_depth_delta_m"] > config.max_median_delta_m:
        warnings.append("large_median_depth_delta")
    if frame.get("valid_mask_iou") is not None and frame["valid_mask_iou"] < config.min_valid_mask_iou:
        warnings.append("low_valid_mask_iou")
    return warnings


def _bad_segments_from_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for frame in frames:
        warnings = frame.get("warnings") or []
        if warnings and active is None:
            active = {"start_frame": frame["frame"], "end_frame": frame["frame"], "reasons": set(warnings)}
        elif warnings and active is not None:
            active["end_frame"] = frame["frame"]
            active["reasons"].update(warnings)
        elif not warnings and active is not None:
            segments.append({**active, "reasons": sorted(active["reasons"])})
            active = None
    if active is not None:
        segments.append({**active, "reasons": sorted(active["reasons"])})
    return segments


def _summary_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(frame[key]) for frame in frames if frame.get(key) is not None]

    return {
        "frame_count": len(frames),
        "warning_frame_count": sum(1 for frame in frames if frame.get("warnings")),
        "warnings": sorted({warning for frame in frames for warning in frame.get("warnings", [])}),
        "valid_depth_ratio_avg": float(np.mean(vals("valid_depth_ratio"))) if vals("valid_depth_ratio") else None,
        "depth_median_avg_m": float(np.mean(vals("depth_median_m"))) if vals("depth_median_m") else None,
        "median_depth_delta_avg_m": float(np.mean(vals("median_depth_delta_m"))) if vals("median_depth_delta_m") else None,
        "valid_mask_iou_avg": float(np.mean(vals("valid_mask_iou"))) if vals("valid_mask_iou") else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "frame",
        "depth_path",
        "missing_depth",
        "valid_depth_ratio",
        "depth_min_m",
        "depth_p05_m",
        "depth_median_m",
        "depth_p95_m",
        "depth_max_m",
        "depth_std_m",
        "depth_iqr_m",
        "near_ratio",
        "far_ratio",
        "median_depth_delta_m",
        "valid_mask_iou",
        "warnings",
    ]
    keys = {key for row in rows for key in row if key != "valid_mask"}
    fieldnames = [key for key in preferred if key in keys] + sorted(keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            out = {key: value for key, value in row.items() if key != "valid_mask"}
            if isinstance(out.get("warnings"), list):
                out["warnings"] = ";".join(out["warnings"])
            writer.writerow(out)


def run_depth_numeric_quality_check(
    *,
    depth_dir: Path,
    output_dir: Path,
    config: DepthNumericQualityConfig,
) -> dict[str, Any]:
    depth_dir = Path(depth_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    frames: list[dict[str, Any]] = []
    previous_median: float | None = None
    previous_mask: np.ndarray | None = None

    for frame_idx in range(config.start, config.start + config.num_frames):
        path = find_depth_path_for_frame(depth_dir, frame_idx)
        if not path.exists():
            frame = {
                "frame": frame_idx,
                "depth_path": str(path),
                "missing_depth": True,
                "median_depth_delta_m": None,
                "valid_mask_iou": None,
                "warnings": ["missing_depth"],
            }
            frames.append(frame)
            previous_median = None
            previous_mask = None
            continue

        stats = _depth_stats(load_depth_frame(depth_dir, frame_idx), config)
        current_median = stats["depth_median_m"]
        current_mask = stats.pop("valid_mask")
        frame = {
            "frame": frame_idx,
            "depth_path": str(path),
            "missing_depth": False,
            **stats,
            "median_depth_delta_m": abs(float(current_median) - previous_median)
            if current_median is not None and previous_median is not None
            else None,
            "valid_mask_iou": _valid_mask_iou(current_mask, previous_mask),
        }
        frame["warnings"] = _frame_warnings(frame, config)
        frames.append(frame)
        previous_median = float(current_median) if current_median is not None else None
        previous_mask = current_mask

    bad_segments = _bad_segments_from_frames(frames)
    report = {
        "stage": "depth_numeric_quality",
        "depth_dir": str(depth_dir),
        "config": {
            "start": config.start,
            "num_frames": config.num_frames,
            "min_depth_m": config.min_depth_m,
            "max_depth_m": config.max_depth_m,
            "min_valid_depth_ratio": config.min_valid_depth_ratio,
            "min_depth_std_m": config.min_depth_std_m,
            "min_depth_iqr_m": config.min_depth_iqr_m,
            "max_near_ratio": config.max_near_ratio,
            "max_far_ratio": config.max_far_ratio,
            "max_median_delta_m": config.max_median_delta_m,
            "min_valid_mask_iou": config.min_valid_mask_iou,
        },
        "summary": _summary_from_frames(frames),
        "bad_segments": bad_segments,
        "frames": frames,
        "outputs": {
            "report": str(output_dir / "depth_numeric_quality_report.json"),
            "csv": str(output_dir / "depth_numeric_quality_frames.csv"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_frames = [{key: value for key, value in frame.items() if key != "valid_mask"} for frame in frames]
    report["frames"] = serializable_frames
    (output_dir / "depth_numeric_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "depth_numeric_quality_frames.csv", serializable_frames)
    return report
