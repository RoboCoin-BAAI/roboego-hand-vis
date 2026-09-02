from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class KeyframeSelectorConfig:
    analysis_fps: float = 2.0
    base_interval_s: float = 5.0
    boundary_margin_s: float = 0.0
    min_keyframe_gap_s: float = 1.0
    visual_change_quantile: float = 0.85
    max_keyframes_per_minute: int = 30
    max_width: int = 320


def load_valid_intervals(video_simple_quality_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(video_simple_quality_json).read_text(encoding="utf-8"))
    intervals = []
    for interval in payload.get("intervals", []):
        if interval.get("route_hint") not in {"pass_to_qwen", "pass_to_qwen_with_aux"}:
            continue
        intervals.append(
            {
                "start": float(interval["start"]),
                "end": float(interval["end"]),
                "route_hint": str(interval.get("route_hint", "")),
            }
        )
    return intervals


def select_qwen_keyframes(
    *,
    video_path: Path,
    video_simple_quality_json: Path,
    output_json: Path,
    config: KeyframeSelectorConfig = KeyframeSelectorConfig(),
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser()
    output_json = Path(output_json).expanduser()
    intervals = load_valid_intervals(Path(video_simple_quality_json).expanduser())
    samples = sample_video_metrics(video_path=video_path, intervals=intervals, config=config)
    candidates = _base_candidates(intervals, config)
    candidates.extend(_visual_change_candidates(samples, config))
    keyframes = _merge_candidates(candidates, min_gap_s=config.min_keyframe_gap_s)
    keyframes = _apply_budget(keyframes, samples, intervals, config)
    report = {
        "stage": "qwen_keyframe_selector",
        "source_video": str(video_path),
        "video_simple_quality_json": str(video_simple_quality_json),
        "config": {
            "analysis_fps": config.analysis_fps,
            "base_interval_s": config.base_interval_s,
            "boundary_margin_s": config.boundary_margin_s,
            "min_keyframe_gap_s": config.min_keyframe_gap_s,
            "visual_change_quantile": config.visual_change_quantile,
            "max_keyframes_per_minute": config.max_keyframes_per_minute,
            "max_width": config.max_width,
        },
        "source_intervals": intervals,
        "keyframe_count": len(keyframes),
        "keyframes": keyframes,
        "samples": samples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def sample_video_metrics(
    *,
    video_path: Path,
    intervals: list[dict[str, Any]],
    config: KeyframeSelectorConfig,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / video_fps if video_fps > 0 else 0.0
    step_s = 1.0 / max(float(config.analysis_fps), 1e-6)
    samples: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    try:
        for interval in intervals:
            start_s = max(0.0, float(interval["start"]))
            end_s = min(duration_s, float(interval["end"]))
            for time_s in np.arange(start_s, end_s, step_s):
                frame = _read_frame_at(cap, time_s=time_s, video_fps=video_fps, frame_count=frame_count)
                if frame is None:
                    continue
                frame = _resize_max_width(frame, config.max_width)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = _hsv_hist(frame)
                gray_diff = float(np.mean(cv2.absdiff(gray, previous_gray))) if previous_gray is not None else 0.0
                hist_diff = _hist_diff(hist, previous_hist) if previous_hist is not None else 0.0
                samples.append(
                    {
                        "t": round(float(time_s), 6),
                        "gray_diff": gray_diff,
                        "hist_diff": hist_diff,
                        "visual_change_score": gray_diff + 50.0 * hist_diff,
                    }
                )
                previous_gray = gray
                previous_hist = hist
    finally:
        cap.release()
    return samples


def _read_frame_at(
    cap: cv2.VideoCapture,
    *,
    time_s: float,
    video_fps: float,
    frame_count: int,
) -> np.ndarray | None:
    frame_idx = int(round(time_s * video_fps)) if video_fps > 0 else 0
    frame_idx = max(0, min(frame_count - 1, frame_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def _hsv_hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist


def _hist_diff(hist: np.ndarray, previous_hist: np.ndarray | None) -> float:
    if previous_hist is None:
        return 0.0
    correlation = float(cv2.compareHist(hist, previous_hist, cv2.HISTCMP_CORREL))
    if not np.isfinite(correlation):
        return 0.0
    return float(max(0.0, min(2.0, 1.0 - correlation)))


def _resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def _base_candidates(intervals: list[dict[str, Any]], config: KeyframeSelectorConfig) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for interval in intervals:
        start_s = float(interval["start"])
        end_s = float(interval["end"])
        if end_s <= start_s:
            continue
        candidates.append({"t": round(start_s, 6), "reasons": ["interval_start"]})
        end_boundary = max(start_s, end_s - config.boundary_margin_s)
        if config.boundary_margin_s > 0.0 and end_boundary > start_s:
            candidates.append({"t": round(end_boundary, 6), "reasons": ["interval_end"]})
        for time_s in np.arange(start_s, end_s, max(config.base_interval_s, 1e-6)):
            candidates.append({"t": round(float(time_s), 6), "reasons": ["base_interval_sample"]})
    return candidates


def _visual_change_candidates(samples: list[dict[str, Any]], config: KeyframeSelectorConfig) -> list[dict[str, Any]]:
    if len(samples) < 2:
        return []
    if config.visual_change_quantile > 1.0:
        return []
    scores = np.asarray([float(sample["visual_change_score"]) for sample in samples], dtype=np.float64)
    threshold = float(np.quantile(scores, min(max(config.visual_change_quantile, 0.0), 1.0)))
    candidates: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        score = float(sample["visual_change_score"])
        if score <= 0.0 or score < threshold:
            continue
        previous_score = float(samples[idx - 1]["visual_change_score"]) if idx > 0 else -1.0
        next_score = float(samples[idx + 1]["visual_change_score"]) if idx + 1 < len(samples) else -1.0
        if score >= previous_score and score >= next_score:
            candidates.append(
                {
                    "t": float(sample["t"]),
                    "reasons": ["visual_change_peak"],
                    "metrics": {
                        "visual_change_score": score,
                        "gray_diff": float(sample["gray_diff"]),
                        "hist_diff": float(sample["hist_diff"]),
                    },
                }
            )
    return candidates


def _merge_candidates(candidates: list[dict[str, Any]], *, min_gap_s: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: float(item["t"])):
        if merged and float(candidate["t"]) - float(merged[-1]["t"]) < min_gap_s:
            for reason in candidate.get("reasons", []):
                if reason not in merged[-1]["reasons"]:
                    merged[-1]["reasons"].append(reason)
            if candidate.get("metrics") and not merged[-1].get("metrics"):
                merged[-1]["metrics"] = candidate["metrics"]
            continue
        item = {"t": float(candidate["t"]), "reasons": list(candidate.get("reasons", []))}
        if candidate.get("metrics"):
            item["metrics"] = candidate["metrics"]
        merged.append(item)
    return merged


def _apply_budget(
    keyframes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    config: KeyframeSelectorConfig,
) -> list[dict[str, Any]]:
    total_duration_s = sum(max(0.0, float(item["end"]) - float(item["start"])) for item in intervals)
    budget = int(np.ceil(max(total_duration_s, 1.0) / 60.0 * max(config.max_keyframes_per_minute, 1)))
    if len(keyframes) <= budget:
        return keyframes
    score_by_t = {float(sample["t"]): float(sample["visual_change_score"]) for sample in samples}

    def priority(item: dict[str, Any]) -> tuple[int, float]:
        reasons = set(item.get("reasons", []))
        fixed = int(bool(reasons & {"interval_start", "interval_end", "base_interval_sample"}))
        return fixed, float(score_by_t.get(float(item["t"]), item.get("metrics", {}).get("visual_change_score", 0.0)))

    selected = sorted(keyframes, key=priority, reverse=True)[:budget]
    return sorted(selected, key=lambda item: float(item["t"]))
