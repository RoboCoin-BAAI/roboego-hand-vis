"""OpenVINS timing log summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TIMING_STAGES = (
    "tracking",
    "propagation",
    "msckf_update",
    "slam_update",
    "slam_delayed",
    "retriangulation_and_marginalization",
    "total",
)


def summarize_openvins_timing(path: str | Path) -> dict[str, object]:
    rows = np.atleast_2d(np.genfromtxt(Path(path), delimiter=",", comments="#"))
    if rows.size == 0 or rows.shape[1] != len(TIMING_STAGES) + 1:
        raise ValueError(f"Unexpected OpenVINS timing layout in {path}")
    durations = rows[:, 1:]
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError(f"Invalid duration in OpenVINS timing log {path}")
    total = durations[:, -1]
    mean_total_s = float(np.mean(total))
    return {
        "sample_count": len(rows),
        "mean_total_ms": mean_total_s * 1000.0,
        "median_total_ms": float(np.median(total) * 1000.0),
        "p95_total_ms": float(np.percentile(total, 95) * 1000.0),
        "equivalent_rate_hz": float(1.0 / mean_total_s) if mean_total_s > 0 else 0.0,
        "mean_stage_ms": {
            stage: float(value * 1000.0)
            for stage, value in zip(TIMING_STAGES, np.mean(durations, axis=0))
        },
    }
