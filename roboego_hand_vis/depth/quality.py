from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .foundation import load_foundation_inputs_from_manifest, manifest_meta


@dataclass(frozen=True)
class DepthQualityConfig:
    start: int = 0
    num_frames: int = 300
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    point_stride: int = 4
    max_preview_points: int = 90000
    overlay_alpha: float = 0.45
    warp_sample_stride: int = 2
    edge_percentile: float = 85.0


def depth_filename(depth_dir: Path, frame_idx: int) -> Path:
    preferred = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if preferred.exists():
        return preferred
    fallback = depth_dir / f"frame_{frame_idx:06d}_depth_m.npy"
    if fallback.exists():
        return fallback
    return preferred


def read_rgb_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def depth_to_vis(depth_m: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    gray = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, min_depth_m, max_depth_m)
        normalized = 1.0 - (clipped[valid] - min_depth_m) / max(max_depth_m - min_depth_m, 1e-6)
        gray[valid] = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def make_depth_overlay(rgb: np.ndarray, depth_m: np.ndarray, config: DepthQualityConfig) -> np.ndarray:
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)
    depth_vis_bgr = depth_to_vis(depth_m, config.min_depth_m, config.max_depth_m)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(rgb_bgr, 1.0 - config.overlay_alpha, depth_vis_bgr, config.overlay_alpha, 0)


def depth_stats(depth_m: np.ndarray, config: DepthQualityConfig) -> dict[str, Any]:
    valid = np.isfinite(depth_m) & (depth_m >= config.min_depth_m) & (depth_m <= config.max_depth_m)
    values = depth_m[valid]
    if values.size == 0:
        return {
            "valid_depth_ratio": 0.0,
            "depth_min_m": None,
            "depth_median_m": None,
            "depth_max_m": None,
        }
    return {
        "valid_depth_ratio": float(np.mean(valid)),
        "depth_min_m": float(np.min(values)),
        "depth_median_m": float(np.median(values)),
        "depth_max_m": float(np.max(values)),
    }


def edge_mask_from_rgb(rgb: np.ndarray, percentile: float) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mag = cv2.Laplacian(gray, cv2.CV_32F)
    abs_mag = np.abs(mag)
    threshold = np.percentile(abs_mag, percentile)
    return abs_mag >= threshold


def warp_right_to_left_error(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
    baseline_m: float,
    config: DepthQualityConfig,
) -> dict[str, Any]:
    if left_rgb.shape[:2] != depth_m.shape:
        left_rgb = cv2.resize(left_rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)
    if right_rgb.shape[:2] != depth_m.shape:
        right_rgb = cv2.resize(right_rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = depth_m.shape
    stride = max(1, config.warp_sample_stride)
    ys, xs = np.mgrid[0:h:stride, 0:w:stride].astype(np.float32)
    z = depth_m[ys.astype(np.int32), xs.astype(np.int32)]
    valid = np.isfinite(z) & (z >= config.min_depth_m) & (z <= config.max_depth_m)
    if not valid.any():
        return {"warp_valid_ratio": 0.0, "warp_l1_mean": None, "warp_l1_median": None, "edge_warp_l1_mean": None}

    disp = float(intrinsics["fx"]) * float(baseline_m) / np.maximum(z, 1e-6)
    x_right = xs - disp
    in_bounds = valid & (x_right >= 0) & (x_right <= w - 1)
    if not in_bounds.any():
        return {"warp_valid_ratio": 0.0, "warp_l1_mean": None, "warp_l1_median": None, "edge_warp_l1_mean": None}

    map_x = x_right.astype(np.float32)
    map_y = ys.astype(np.float32)
    warped_right = cv2.remap(
        right_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_left = left_rgb[ys.astype(np.int32), xs.astype(np.int32)]
    l1 = np.mean(np.abs(sampled_left.astype(np.float32) - warped_right.astype(np.float32)), axis=2)
    l1_values = l1[in_bounds]

    edge_mask = edge_mask_from_rgb(left_rgb, config.edge_percentile)[ys.astype(np.int32), xs.astype(np.int32)]
    edge_values = l1[in_bounds & edge_mask]
    return {
        "warp_valid_ratio": float(np.mean(in_bounds)),
        "warp_l1_mean": float(np.mean(l1_values)),
        "warp_l1_median": float(np.median(l1_values)),
        "edge_warp_l1_mean": float(np.mean(edge_values)) if edge_values.size else None,
    }


def depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict[str, float],
    config: DepthQualityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = depth_m.shape
    stride = max(1, config.point_stride)
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs].astype(np.float32)
    colors = rgb[ys, xs].reshape(-1, 3)
    valid = np.isfinite(z) & (z >= config.min_depth_m) & (z <= config.max_depth_m)
    xs = xs.astype(np.float32)[valid]
    ys = ys.astype(np.float32)[valid]
    z = z[valid]
    colors = colors[valid.reshape(-1)]

    x = (xs - float(intrinsics["cx"])) * z / float(intrinsics["fx"])
    y = (ys - float(intrinsics["cy"])) * z / float(intrinsics["fy"])
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    if config.max_preview_points > 0 and len(points) > config.max_preview_points:
        indices = np.linspace(0, len(points) - 1, config.max_preview_points).astype(np.int64)
        points = points[indices]
        colors = colors[indices]
    return points, colors.astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as f:
        f.write(header)
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def frame_warnings(frame_report: dict[str, Any]) -> list[str]:
    warnings = []
    if frame_report["valid_depth_ratio"] < 0.5:
        warnings.append("low_valid_depth_ratio")
    if frame_report["depth_median_m"] is None:
        warnings.append("missing_valid_depth")
    if frame_report.get("warp_l1_mean") is not None and frame_report["warp_l1_mean"] > 35.0:
        warnings.append("high_warp_error")
    if frame_report.get("edge_warp_l1_mean") is not None and frame_report["edge_warp_l1_mean"] > 45.0:
        warnings.append("high_edge_warp_error")
    return warnings


def load_stereo_quality_warnings(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    path = Path(path).expanduser()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("bad_segments", []))


def stereo_quality_warnings_for_frame(frame_idx: int, segments: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for segment in segments:
        start = int(segment.get("start_frame", -1))
        end = int(segment.get("end_frame", -1))
        if start <= frame_idx <= end:
            for reason in segment.get("reasons", []):
                warnings.append(f"stereo_quality_{reason}")
    return sorted(set(warnings))


def summary_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(f[key]) for f in frames if f.get(key) is not None]

    return {
        "frame_count": len(frames),
        "valid_depth_ratio_avg": float(np.mean(vals("valid_depth_ratio"))) if vals("valid_depth_ratio") else None,
        "depth_median_avg_m": float(np.mean(vals("depth_median_m"))) if vals("depth_median_m") else None,
        "warp_l1_mean_avg": float(np.mean(vals("warp_l1_mean"))) if vals("warp_l1_mean") else None,
        "warning_frame_count": sum(1 for f in frames if f.get("warnings")),
        "warnings": sorted({warning for f in frames for warning in f.get("warnings", [])}),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("warnings"), list):
                out["warnings"] = ";".join(out["warnings"])
            writer.writerow(out)


def run_depth_quality_check(
    *,
    manifest_path: Path,
    depth_dir: Path,
    output_dir: Path,
    config: DepthQualityConfig,
    stereo_quality_report: Path | None = None,
) -> dict[str, Any]:
    inputs = load_foundation_inputs_from_manifest(manifest_path)
    depth_dir = Path(depth_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    vis_dir = output_dir / "vis"
    pointcloud_dir = output_dir / "pointcloud_preview"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    stereo_quality_segments = load_stereo_quality_warnings(stereo_quality_report)

    frames = []
    for frame_idx in range(config.start, config.start + config.num_frames):
        path = depth_filename(depth_dir, frame_idx)
        if not path.exists():
            frames.append({"frame": frame_idx, "depth_path": str(path), "missing_depth": True, "warnings": ["missing_depth"]})
            continue

        depth_m = np.load(path).astype(np.float32)
        left_rgb = read_rgb_frame(inputs.left_video, frame_idx)
        right_rgb = read_rgb_frame(inputs.right_video, frame_idx)
        stats = depth_stats(depth_m, config)
        warp = warp_right_to_left_error(left_rgb, right_rgb, depth_m, inputs.intrinsics, inputs.baseline_m, config)

        overlay = make_depth_overlay(left_rgb, depth_m, config)
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}_depth_overlay.jpg"), overlay)
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}_depth_vis.jpg"), depth_to_vis(depth_m, config.min_depth_m, config.max_depth_m))

        points, colors = depth_to_pointcloud(depth_m, left_rgb, inputs.intrinsics, config)
        write_ply(pointcloud_dir / f"frame_{frame_idx:06d}_pointcloud_preview.ply", points, colors)

        frame_report = {
            "frame": frame_idx,
            "depth_path": str(path),
            "missing_depth": False,
            "hand_mask_depth_median": None,
            **stats,
            **warp,
            "pointcloud_preview_points": int(len(points)),
        }
        frame_report["warnings"] = sorted(
            set(frame_warnings(frame_report) + stereo_quality_warnings_for_frame(frame_idx, stereo_quality_segments))
        )
        frames.append(frame_report)

    report = {
        "manifest": manifest_meta(inputs),
        "depth_dir": str(depth_dir),
        "stereo_quality_report": str(stereo_quality_report) if stereo_quality_report else None,
        "config": {
            "start": config.start,
            "num_frames": config.num_frames,
            "min_depth_m": config.min_depth_m,
            "max_depth_m": config.max_depth_m,
            "point_stride": config.point_stride,
            "max_preview_points": config.max_preview_points,
            "warp_sample_stride": config.warp_sample_stride,
        },
        "summary": summary_from_frames(frames),
        "frames": frames,
        "outputs": {
            "csv": str(output_dir / "depth_quality_frames.csv"),
            "vis_dir": str(vis_dir),
            "pointcloud_preview_dir": str(pointcloud_dir),
        },
        "explanations_zh": {
            "valid_depth_ratio": "当前帧有效深度像素比例。过低说明深度缺失或超出范围严重。",
            "depth_min/median/max": "当前帧深度范围，单位米，用于检查尺度是否离谱。",
            "warp_l1_mean": "用 depth 把右图 warp 到左图后的平均颜色误差。越低说明 depth 和双目外观越一致；遮挡区域会自然偏高。",
            "edge_warp_l1_mean": "图像边缘区域的 warp 误差。手/物体边缘、遮挡和低纹理区域通常更高。",
            "hand_mask_depth_median": "预留字段。没有手部候选/segmentation 时为 null；接入 HaMeR 后可统计手部 mask 内深度。",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "depth_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "depth_quality_frames.csv", frames)
    return report
