from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PreSemanticGateConfig:
    sample_every_s: float = 1.0
    dark_threshold: int = 10
    bright_threshold: int = 245
    max_black_ratio: float = 0.98
    max_overexposed_ratio: float = 0.98
    min_texture_std: float = 3.0
    freeze_absdiff_threshold: float = 1.5
    low_motion_absdiff_threshold: float = 4.0
    min_freeze_duration_s: float = 5.0
    min_bad_visual_duration_s: float = 3.0


DROP_LABELS = {
    "black_screen",
    "overexposed_screen",
    "camera_fully_blocked",
    "frozen_video",
}


def frame_gate_label(frame_bgr: np.ndarray, config: PreSemanticGateConfig) -> dict[str, Any]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))
    dark_ratio = float(np.mean(gray <= config.dark_threshold))
    bright_ratio = float(np.mean(gray >= config.bright_threshold))

    metrics = {
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }
    if dark_ratio >= config.max_black_ratio:
        return {
            "label": "black_screen",
            "route_hint": "drop_before_qwen",
            "confidence": 0.99,
            "metrics": metrics,
            "aux": {"low_motion": False},
        }
    if bright_ratio >= config.max_overexposed_ratio:
        return {
            "label": "overexposed_screen",
            "route_hint": "drop_before_qwen",
            "confidence": 0.99,
            "metrics": metrics,
            "aux": {"low_motion": False},
        }
    if gray_std < config.min_texture_std:
        return {
            "label": "camera_fully_blocked",
            "route_hint": "drop_before_qwen",
            "confidence": 0.95,
            "metrics": metrics,
            "aux": {"low_motion": False},
        }
    return {
        "label": "valid_visual_content",
        "route_hint": "pass_to_qwen",
        "confidence": 0.98,
        "metrics": metrics,
        "aux": {"low_motion": False},
    }


def analyze_sampled_frames(
    frames_bgr: list[np.ndarray],
    *,
    times_s: list[float],
    config: PreSemanticGateConfig,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    freeze_run_start: float | None = None

    for idx, (frame, time_s) in enumerate(zip(frames_bgr, times_s)):
        row = frame_gate_label(frame, config)
        row["sample_idx"] = idx
        row["time_s"] = float(time_s)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff_mean: float | None = None
        if previous_gray is not None:
            frame_diff_mean = float(np.mean(cv2.absdiff(gray, previous_gray)))
            row["metrics"]["frame_diff_mean"] = frame_diff_mean
            if frame_diff_mean <= config.freeze_absdiff_threshold:
                if freeze_run_start is None:
                    freeze_run_start = times_s[idx - 1]
                freeze_duration = float(time_s - freeze_run_start + config.sample_every_s)
                row["aux"]["low_motion"] = True
                if freeze_duration >= config.min_freeze_duration_s and row["label"] == "valid_visual_content":
                    row["label"] = "frozen_video"
                    row["route_hint"] = "drop_before_qwen"
                    row["confidence"] = 0.99
                    row["reason"] = f"near-identical frames for at least {freeze_duration:.1f} seconds"
                elif row["label"] == "valid_visual_content":
                    row["label"] = "low_motion_static"
                    row["route_hint"] = "pass_to_qwen_with_aux"
                    row["confidence"] = 0.9
            else:
                freeze_run_start = None
                if frame_diff_mean <= config.low_motion_absdiff_threshold and row["label"] == "valid_visual_content":
                    row["label"] = "low_motion_static"
                    row["route_hint"] = "pass_to_qwen_with_aux"
                    row["confidence"] = 0.9
                    row["aux"]["low_motion"] = True
        else:
            row["metrics"]["frame_diff_mean"] = None
        previous_gray = gray
        samples.append(row)

    return samples


def read_sampled_video_frames(video_path: Path, config: PreSemanticGateConfig) -> tuple[dict[str, Any], list[np.ndarray], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (
            {
                "decode_ok": False,
                "path": str(video_path),
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "frame_count": 0,
                "duration_s": 0.0,
            },
            [],
            [],
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    times = list(np.arange(0.0, duration_s, max(config.sample_every_s, 1e-6)))
    frames: list[np.ndarray] = []
    sampled_times: list[float] = []
    for time_s in times:
        frame_idx = int(round(time_s * fps)) if fps > 0 else 0
        frame_idx = max(0, min(frame_count - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
        sampled_times.append(float(time_s))
    cap.release()
    return (
        {
            "decode_ok": bool(frames),
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": duration_s,
        },
        frames,
        sampled_times,
    )


def build_intervals(samples: list[dict[str, Any]], config: PreSemanticGateConfig) -> list[dict[str, Any]]:
    if not samples:
        return []
    intervals: list[dict[str, Any]] = []
    start_idx = 0
    for idx in range(1, len(samples)):
        if _interval_key(samples[idx]) != _interval_key(samples[start_idx]):
            intervals.append(_interval_from_samples(samples[start_idx:idx], config))
            start_idx = idx
    intervals.append(_interval_from_samples(samples[start_idx:], config))
    return intervals


def _interval_key(sample: dict[str, Any]) -> tuple[str, str, bool]:
    return (str(sample["label"]), str(sample["route_hint"]), bool(sample.get("aux", {}).get("low_motion", False)))


def _interval_from_samples(samples: list[dict[str, Any]], config: PreSemanticGateConfig) -> dict[str, Any]:
    start = float(samples[0]["time_s"])
    end = float(samples[-1]["time_s"] + config.sample_every_s)
    labels = [str(sample["label"]) for sample in samples]
    route_hint = str(samples[0]["route_hint"])
    confidence = float(np.mean([float(sample.get("confidence", 0.0)) for sample in samples]))
    interval = {
        "start": start,
        "end": end,
        "label": labels[0],
        "route_hint": route_hint,
        "confidence": confidence,
        "aux": {"low_motion": any(bool(sample.get("aux", {}).get("low_motion", False)) for sample in samples)},
        "sample_count": len(samples),
    }
    if labels[0] in DROP_LABELS:
        interval["reason"] = _reason_for_label(labels[0])
    return interval


def _reason_for_label(label: str) -> str:
    return {
        "black_screen": "near-total black frames",
        "overexposed_screen": "near-total overexposed frames",
        "camera_fully_blocked": "single-color low-texture frames",
        "frozen_video": "near-identical frames over threshold duration",
    }.get(label, label)


def summarize_checks(samples: list[dict[str, Any]], video_info: dict[str, Any]) -> dict[str, Any]:
    sample_count = len(samples)

    def ratio_for(label: str) -> float:
        if sample_count == 0:
            return 0.0
        return float(sum(1 for sample in samples if sample["label"] == label) / sample_count)

    return {
        "decode_ok": bool(video_info.get("decode_ok")),
        "timestamp_ok": True,
        "missing_frames_ratio": 0.0,
        "sampled_frame_count": sample_count,
        "black_screen_ratio": ratio_for("black_screen"),
        "overexposed_ratio": ratio_for("overexposed_screen"),
        "frozen_ratio": ratio_for("frozen_video"),
        "blocked_ratio": ratio_for("camera_fully_blocked"),
        "low_motion_ratio": float(
            sum(1 for sample in samples if sample.get("aux", {}).get("low_motion", False)) / sample_count
        )
        if sample_count
        else 0.0,
    }


def run_pre_semantic_quality_gate(
    *,
    video_path: Path,
    output_json: Path,
    config: PreSemanticGateConfig = PreSemanticGateConfig(),
) -> dict[str, Any]:
    video_path = Path(video_path).expanduser()
    output_json = Path(output_json).expanduser()
    video_info, frames, times_s = read_sampled_video_frames(video_path, config)
    samples = analyze_sampled_frames(frames, times_s=times_s, config=config) if video_info["decode_ok"] else []
    intervals = build_intervals(samples, config)
    report = {
        "source_video": str(video_path),
        "stage": "video_simple_quality_gate",
        "config": {
            "sample_every_s": config.sample_every_s,
            "min_freeze_duration_s": config.min_freeze_duration_s,
            "min_bad_visual_duration_s": config.min_bad_visual_duration_s,
        },
        "video": video_info,
        "checks": summarize_checks(samples, video_info),
        "intervals": intervals,
        "samples": samples,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
