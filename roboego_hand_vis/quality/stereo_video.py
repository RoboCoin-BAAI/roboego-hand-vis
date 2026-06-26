from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoQualityThresholds:
    max_brightness_gap: float = 35.0
    min_sharpness: float = 50.0
    max_dark_ratio: float = 0.25
    max_bright_ratio: float = 0.25
    min_orb_matches: int = 30
    max_median_abs_dy: float = 2.0
    max_p90_abs_dy: float = 5.0


@dataclass(frozen=True)
class SummaryQualityConfig(StereoQualityThresholds):
    sample_count: int = 12
    dark_threshold: int = 10
    bright_threshold: int = 245
    orb_features: int = 1500


@dataclass(frozen=True)
class SegmentQualityConfig(StereoQualityThresholds):
    sample_every_s: float = 1.0
    orb_every_s: float = 5.0
    dark_threshold: int = 10
    bright_threshold: int = 245
    orb_features: int = 1500
    max_bad_contact_frames: int = 40


def load_manifest_video_paths(manifest_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    left_path = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
    right_path = Path(manifest["videos"]["right"]["path"]).expanduser().resolve()
    return left_path, right_path, manifest


def inspect_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    info = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": frame_count,
        "fps": fps,
        "duration_s": frame_count / fps if fps > 0 else 0.0,
    }
    cap.release()
    return info


def sample_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    sample_count = max(1, min(sample_count, frame_count))
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, sample_count)})


def sample_times(duration_s: float, sample_every_s: float) -> list[float]:
    if duration_s <= 0:
        return []
    sample_every_s = max(sample_every_s, 1e-6)
    times = list(np.arange(0.0, duration_s, sample_every_s))
    if not times or times[-1] < duration_s - 1e-6:
        times.append(duration_s)
    return times


def frame_index_from_time(time_s: float, fps: float, frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    idx = int(round(time_s * fps))
    return max(0, min(frame_count - 1, idx))


def read_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {path}")
    return frame


def frame_quality(frame: np.ndarray, dark_threshold: int, bright_threshold: int) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
        "sharp_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_ratio": float(np.mean(gray <= dark_threshold)),
        "bright_ratio": float(np.mean(gray >= bright_threshold)),
    }


def orb_vertical_error(left: np.ndarray, right: np.ndarray, max_features: int) -> dict[str, Any]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    left_kp, left_des = orb.detectAndCompute(left_gray, None)
    right_kp, right_des = orb.detectAndCompute(right_gray, None)
    if left_des is None or right_des is None:
        return empty_orb_result()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(left_des, right_des, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    dx_values = []
    dy_values = []
    for match in good_matches:
        x_left, y_left = left_kp[match.queryIdx].pt
        x_right, y_right = right_kp[match.trainIdx].pt
        dx_values.append(x_right - x_left)
        dy_values.append(y_right - y_left)

    if not dy_values:
        return empty_orb_result()

    dx = np.asarray(dx_values, dtype=np.float32)
    dy = np.asarray(dy_values, dtype=np.float32)
    return {
        "orb_matches": int(len(dy)),
        "median_dy": float(np.median(dy)),
        "median_abs_dy": float(np.median(np.abs(dy))),
        "p90_abs_dy": float(np.percentile(np.abs(dy), 90)),
        "median_dx": float(np.median(dx)),
    }


def empty_orb_result() -> dict[str, Any]:
    return {
        "orb_matches": 0,
        "median_dy": None,
        "median_abs_dy": None,
        "p90_abs_dy": None,
        "median_dx": None,
    }


def draw_pair(left: np.ndarray, right: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
    pair = np.concatenate([left, right], axis=1)
    height, width = left.shape[:2]
    for y in range(64, height, 64):
        cv2.line(pair, (0, y), (pair.shape[1] - 1, y), (0, 255, 255), 1)
    cv2.line(pair, (width, 0), (width, height - 1), (0, 255, 0), 2)
    time_s = frame_idx / fps if fps > 0 else 0.0
    cv2.putText(
        pair,
        f"frame {frame_idx}  t={time_s:.2f}s",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(pair, "LEFT", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pair, "RIGHT", (width + 12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return pair


def write_contact_sheet(pair_images: list[np.ndarray], output_path: Path) -> None:
    if not pair_images:
        return
    thumbs = [cv2.resize(img, (768, 384), interpolation=cv2.INTER_AREA) for img in pair_images]
    if len(thumbs) % 2 == 1:
        thumbs.append(np.zeros_like(thumbs[-1]))
    rows = [np.concatenate(thumbs[i : i + 2], axis=1) for i in range(0, len(thumbs), 2)]
    contact = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output_path), contact)


def write_bad_contact_sheet(bad_pairs: list[np.ndarray], output_path: Path) -> None:
    if bad_pairs:
        write_contact_sheet(bad_pairs, output_path)
        return

    canvas = np.full((384, 768, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        "No bad frames found",
        (80, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (40, 120, 40),
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), canvas)


def build_sample_row(
    frame_idx: int,
    fps: float,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
    dark_threshold: int,
    bright_threshold: int,
    orb_features: int | None = None,
) -> dict[str, Any]:
    left_quality = frame_quality(left_frame, dark_threshold, bright_threshold)
    right_quality = frame_quality(right_frame, dark_threshold, bright_threshold)
    row: dict[str, Any] = {
        "frame": frame_idx,
        "time_s": frame_idx / fps if fps > 0 else 0.0,
    }
    for key, value in left_quality.items():
        row[f"left_{key}"] = value
    for key, value in right_quality.items():
        row[f"right_{key}"] = value
    row["brightness_gap"] = abs(left_quality["gray_mean"] - right_quality["gray_mean"])
    if orb_features is None:
        row.update(empty_orb_result())
        row["orb_checked"] = False
    else:
        row.update(orb_vertical_error(left_frame, right_frame, orb_features))
        row["orb_checked"] = True
    return row


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def min_or_none(values: list[float]) -> float | None:
    return float(np.min(values)) if values else None


def summarize(rows: list[dict[str, Any]], config: StereoQualityThresholds) -> dict[str, Any]:
    summary = {
        "sample_count": len(rows),
        "left_mean_avg": mean_or_none(numeric_values(rows, "left_gray_mean")),
        "right_mean_avg": mean_or_none(numeric_values(rows, "right_gray_mean")),
        "left_sharp_avg": mean_or_none(numeric_values(rows, "left_sharp_laplacian_var")),
        "right_sharp_avg": mean_or_none(numeric_values(rows, "right_sharp_laplacian_var")),
        "left_dark_ratio_max": max_or_none(numeric_values(rows, "left_dark_ratio")),
        "right_dark_ratio_max": max_or_none(numeric_values(rows, "right_dark_ratio")),
        "left_bright_ratio_max": max_or_none(numeric_values(rows, "left_bright_ratio")),
        "right_bright_ratio_max": max_or_none(numeric_values(rows, "right_bright_ratio")),
        "brightness_gap_avg": mean_or_none(numeric_values(rows, "brightness_gap")),
        "orb_matches_min": min_or_none(numeric_values(rows, "orb_matches")),
        "median_abs_dy_avg": mean_or_none(numeric_values(rows, "median_abs_dy")),
        "p90_abs_dy_max": max_or_none(numeric_values(rows, "p90_abs_dy")),
    }

    checks = {
        "brightness_gap": compare_le(summary["brightness_gap_avg"], config.max_brightness_gap),
        "sharpness": compare_ge(
            min_non_none(summary["left_sharp_avg"], summary["right_sharp_avg"]),
            config.min_sharpness,
        ),
        "dark_ratio": compare_le(
            max_non_none(summary["left_dark_ratio_max"], summary["right_dark_ratio_max"]),
            config.max_dark_ratio,
        ),
        "bright_ratio": compare_le(
            max_non_none(summary["left_bright_ratio_max"], summary["right_bright_ratio_max"]),
            config.max_bright_ratio,
        ),
        "orb_matches": compare_ge(summary["orb_matches_min"], config.min_orb_matches),
        "median_abs_dy": compare_le(summary["median_abs_dy_avg"], config.max_median_abs_dy),
        "p90_abs_dy": compare_le(summary["p90_abs_dy_max"], config.max_p90_abs_dy),
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def compare_le(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def compare_ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def min_non_none(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def max_non_none(*values: float | None) -> float | None:
    present = [v for v in values if v is not None]
    return max(present) if present else None


def classify_frame(row: dict[str, Any], config: StereoQualityThresholds) -> list[str]:
    reasons = []
    if row["brightness_gap"] > config.max_brightness_gap:
        reasons.append("brightness_gap")
    if min(row["left_sharp_laplacian_var"], row["right_sharp_laplacian_var"]) < config.min_sharpness:
        reasons.append("sharpness_low")
    if max(row["left_dark_ratio"], row["right_dark_ratio"]) > config.max_dark_ratio:
        reasons.append("dark_ratio")
    if max(row["left_bright_ratio"], row["right_bright_ratio"]) > config.max_bright_ratio:
        reasons.append("bright_ratio")

    if row.get("orb_checked"):
        if row["orb_matches"] is None or row["orb_matches"] < config.min_orb_matches:
            reasons.append("orb_matches_low")
        if row["median_abs_dy"] is not None and row["median_abs_dy"] > config.max_median_abs_dy:
            reasons.append("median_abs_dy")
        if row["p90_abs_dy"] is not None and row["p90_abs_dy"] > config.max_p90_abs_dy:
            reasons.append("p90_abs_dy")
    return reasons


def should_run_orb(time_s: float, last_orb_time_s: float | None, orb_every_s: float) -> bool:
    if orb_every_s <= 0:
        return True
    if last_orb_time_s is None:
        return True
    return (time_s - last_orb_time_s) >= (orb_every_s - 1e-6)


def merge_bad_segments(bad_rows: list[dict[str, Any]], sample_every_s: float) -> list[dict[str, Any]]:
    if not bad_rows:
        return []

    segments = []
    current = {
        "start_s": bad_rows[0]["time_s"],
        "end_s": bad_rows[0]["time_s"] + sample_every_s,
        "start_frame": bad_rows[0]["frame"],
        "end_frame": bad_rows[0]["frame"],
        "reasons": set(bad_rows[0]["reasons"]),
        "bad_sample_count": 1,
    }

    for row in bad_rows[1:]:
        continuous = row["time_s"] <= current["end_s"] + sample_every_s * 0.5
        if continuous:
            current["end_s"] = row["time_s"] + sample_every_s
            current["end_frame"] = row["frame"]
            current["reasons"].update(row["reasons"])
            current["bad_sample_count"] += 1
        else:
            current["reasons"] = sorted(current["reasons"])
            segments.append(current)
            current = {
                "start_s": row["time_s"],
                "end_s": row["time_s"] + sample_every_s,
                "start_frame": row["frame"],
                "end_frame": row["frame"],
                "reasons": set(row["reasons"]),
                "bad_sample_count": 1,
            }

    current["reasons"] = sorted(current["reasons"])
    segments.append(current)
    return segments


def metric_explanations_zh() -> dict[str, str]:
    return {
        "sample_count": "本次从左右视频中均匀抽样的帧数。抽样越多，质检越稳定，但运行越慢。",
        "left_mean_avg/right_mean_avg": "左右相机抽样帧的平均灰度亮度，范围约为0-255。数值越大画面越亮。",
        "left_sharp_avg/right_sharp_avg": "左右相机抽样帧的平均清晰度，使用 Laplacian variance。数值越大通常越清晰；太低说明失焦、运动模糊或压缩损失严重。",
        "left_dark_ratio_max/right_dark_ratio_max": "抽样帧中黑暗像素比例的最大值。像素灰度 <= dark_threshold 会被认为偏黑。比例太高说明画面存在大面积死黑或欠曝。",
        "left_bright_ratio_max/right_bright_ratio_max": "抽样帧中过亮像素比例的最大值。像素灰度 >= bright_threshold 会被认为偏白。比例太高说明画面存在大面积过曝。",
        "brightness_gap_avg": "同一抽样帧中左右图平均亮度差的平均值。差异太大会影响双目匹配，因为左右图外观不一致。",
        "orb_matches_min": "每个抽样帧左右图 ORB 特征匹配数量的最小值。太少说明纹理少、模糊严重、曝光差异大，或者左右图不容易匹配。",
        "median_abs_dy_avg": "ORB 匹配点的左右图 y 坐标差的中位绝对值平均。双目图若已极线校正，对应点应该主要水平移动，所以这个值越接近0越好。",
        "p90_abs_dy_max": "ORB 匹配点 y 坐标差绝对值的90分位最大值。用于观察较差样本中的垂直误差尾部，越小越适合直接做双目深度。",
        "passed": "所有 checks 都为 true 时为 true，表示当前抽样质检通过，可以进入下一步。",
    }


def threshold_explanations_zh(config: StereoQualityThresholds) -> dict[str, str]:
    return {
        "max_brightness_gap": f"左右图平均亮度差的上限，当前阈值 {config.max_brightness_gap}。实际 brightness_gap_avg 必须 <= 该值。",
        "min_sharpness": f"左右图平均清晰度下限，当前阈值 {config.min_sharpness}。left_sharp_avg 和 right_sharp_avg 中较小者必须 >= 该值。",
        "max_dark_ratio": f"最大黑暗像素比例上限，当前阈值 {config.max_dark_ratio}。左右任一路 dark_ratio_max 都必须 <= 该值。",
        "max_bright_ratio": f"最大过曝像素比例上限，当前阈值 {config.max_bright_ratio}。左右任一路 bright_ratio_max 都必须 <= 该值。",
        "min_orb_matches": f"每个抽样帧最少 ORB 匹配数量，当前阈值 {config.min_orb_matches}。orb_matches_min 必须 >= 该值。",
        "max_median_abs_dy": f"平均中位垂直误差上限，当前阈值 {config.max_median_abs_dy} 像素。median_abs_dy_avg 必须 <= 该值。",
        "max_p90_abs_dy": f"90分位垂直误差最大值上限，当前阈值 {config.max_p90_abs_dy} 像素。p90_abs_dy_max 必须 <= 该值。",
    }


def next_step_gate_zh(summary: dict[str, Any]) -> dict[str, Any]:
    failed_checks = [name for name, ok in summary["checks"].items() if not ok]
    if summary["passed"]:
        recommendation = "当前抽样质检通过，可以进入下一步：读取 stereo 相机内参、baseline，并确认这些参数对应当前左右视频。"
    else:
        recommendation = "当前抽样质检未通过，不建议直接做 FoundationStereo 深度估计。请先查看 failed_checks 和 contact_sheet，定位曝光、模糊、极线校正或匹配不足的问题。"
    return {
        "进入下一步的标准": "summary.checks 中所有项目都为 true，也就是 summary.passed 为 true。",
        "下一步是什么": "确认 stereo 标定参数：K、baseline、是否已去畸变/极线校正、K 是否对应当前视频分辨率。",
        "当前建议": recommendation,
        "failed_checks": failed_checks,
    }


def segment_explanations_zh() -> dict[str, str]:
    return {
        "脚本目的": "密集采样视频，不用平均值掩盖局部问题，找出不适合做深度估计或手部姿态估计的时间点和时间段。",
        "sample_every_s": "每隔多少秒采样一帧做亮度、清晰度、过曝/欠曝、左右亮度差检查。",
        "orb_every_s": "每隔多少秒做一次 ORB 左右匹配和极线粗检。ORB 比亮度/清晰度慢，所以默认比普通采样稀疏。",
        "bad_frames.csv": "逐个异常采样点列表，包含异常帧号、时间、指标值和异常原因。",
        "bad_segments.json": "把连续异常采样点合并后的异常时间段，后续深度估计/手部估计可以跳过这些时间段或降低置信度。",
        "contact_sheet_bad.png": "异常采样点的左右并排图。没有异常时会生成一张 No bad frames found 的占位图。",
        "进入下一步": "bad_segments 为空时，可以认为这一步局部视频质量检查通过；如果不为空，应先人工查看异常片段。",
    }


def thresholds_to_dict(config: StereoQualityThresholds) -> dict[str, Any]:
    return {
        "max_brightness_gap": config.max_brightness_gap,
        "min_sharpness": config.min_sharpness,
        "max_dark_ratio": config.max_dark_ratio,
        "max_bright_ratio": config.max_bright_ratio,
        "min_orb_matches": config.min_orb_matches,
        "max_median_abs_dy": config.max_median_abs_dy,
        "max_p90_abs_dy": config.max_p90_abs_dy,
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if isinstance(csv_row.get("reasons"), list):
                csv_row["reasons"] = ";".join(csv_row["reasons"])
            writer.writerow(csv_row)


def run_summary_quality_check(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: SummaryQualityConfig,
) -> dict[str, Any]:
    left_path, right_path, manifest = load_manifest_video_paths(manifest_path)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_info = inspect_video(left_path)
    right_info = inspect_video(right_path)
    frame_count = min(left_info["frame_count"], right_info["frame_count"])
    sample_indices = sample_frame_indices(frame_count, config.sample_count)

    rows = []
    pair_images = []
    for frame_idx in sample_indices:
        left_frame = read_frame(left_path, frame_idx)
        right_frame = read_frame(right_path, frame_idx)
        rows.append(
            build_sample_row(
                frame_idx,
                left_info["fps"],
                left_frame,
                right_frame,
                config.dark_threshold,
                config.bright_threshold,
                config.orb_features,
            )
        )
        pair_images.append(draw_pair(left_frame, right_frame, frame_idx, left_info["fps"]))

    csv_path = output_dir / "stereo_quality_samples.csv"
    write_rows_csv(csv_path, rows)

    contact_sheet_path = output_dir / "contact_sheet.png"
    write_contact_sheet(pair_images, contact_sheet_path)

    report = {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "dataset_name": manifest.get("dataset_name"),
        "sequence_name": manifest.get("sequence_name"),
        "left_video": left_info,
        "right_video": right_info,
        "sample_indices": sample_indices,
        "thresholds": thresholds_to_dict(config),
        "summary": summarize(rows, config),
        "outputs": {
            "csv": str(csv_path),
            "contact_sheet": str(contact_sheet_path),
        },
    }
    report["metric_explanations_zh"] = metric_explanations_zh()
    report["threshold_explanations_zh"] = threshold_explanations_zh(config)
    report["next_step_gate_zh"] = next_step_gate_zh(report["summary"])

    json_path = output_dir / "stereo_quality_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_segment_quality_check(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: SegmentQualityConfig,
) -> dict[str, Any]:
    left_path, right_path, manifest = load_manifest_video_paths(manifest_path)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_info = inspect_video(left_path)
    right_info = inspect_video(right_path)
    frame_count = min(left_info["frame_count"], right_info["frame_count"])
    fps = left_info["fps"]
    duration_s = min(left_info["duration_s"], right_info["duration_s"])

    rows = []
    bad_rows = []
    bad_pairs = []
    last_orb_time_s = None
    seen_frames = set()

    for time_s in sample_times(duration_s, config.sample_every_s):
        frame_idx = frame_index_from_time(time_s, fps, frame_count)
        if frame_idx in seen_frames:
            continue
        seen_frames.add(frame_idx)

        left_frame = read_frame(left_path, frame_idx)
        right_frame = read_frame(right_path, frame_idx)
        run_orb = should_run_orb(frame_idx / fps if fps > 0 else time_s, last_orb_time_s, config.orb_every_s)
        row = build_sample_row(
            frame_idx,
            fps,
            left_frame,
            right_frame,
            config.dark_threshold,
            config.bright_threshold,
            config.orb_features if run_orb else None,
        )
        if run_orb:
            last_orb_time_s = row["time_s"]

        reasons = classify_frame(row, config)
        row["is_bad"] = bool(reasons)
        row["reasons"] = reasons
        rows.append(row)

        if reasons:
            bad_rows.append(row)
            if len(bad_pairs) < config.max_bad_contact_frames:
                bad_pairs.append(draw_pair(left_frame, right_frame, frame_idx, fps))

    bad_frames_csv = output_dir / "bad_frames.csv"
    write_rows_csv(bad_frames_csv, bad_rows)

    all_samples_csv = output_dir / "all_sampled_frames.csv"
    write_rows_csv(all_samples_csv, rows)

    bad_segments = merge_bad_segments(bad_rows, config.sample_every_s)
    bad_segments_path = output_dir / "bad_segments.json"
    report = {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "dataset_name": manifest.get("dataset_name"),
        "sequence_name": manifest.get("sequence_name"),
        "passed": len(bad_segments) == 0,
        "left_video": left_info,
        "right_video": right_info,
        "sample_every_s": config.sample_every_s,
        "orb_every_s": config.orb_every_s,
        "sampled_frame_count": len(rows),
        "bad_frame_count": len(bad_rows),
        "bad_segment_count": len(bad_segments),
        "bad_segments": bad_segments,
        "thresholds": thresholds_to_dict(config),
        "outputs": {
            "bad_frames_csv": str(bad_frames_csv),
            "all_sampled_frames_csv": str(all_samples_csv),
            "contact_sheet_bad": str(output_dir / "contact_sheet_bad.png"),
        },
        "explanations_zh": segment_explanations_zh(),
    }
    bad_segments_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    contact_sheet_bad = output_dir / "contact_sheet_bad.png"
    write_bad_contact_sheet(bad_pairs, contact_sheet_bad)
    return report
