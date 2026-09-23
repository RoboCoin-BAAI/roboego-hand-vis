from __future__ import annotations

from pathlib import Path
import argparse
import json

import cv2
import numpy as np


ROPEDIA_K = {
    "fx": 200.0,
    "fy": 200.0,
    "cx": 256.0,
    "cy": 256.0,
}


def read_rgb_frame(video_path: Path, frame_idx: int, resize_scale: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if abs(resize_scale - 1.0) > 1e-6:
        h, w = frame_rgb.shape[:2]
        frame_rgb = cv2.resize(
            frame_rgb,
            (int(round(w * resize_scale)), int(round(h * resize_scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame_rgb


def load_camera_intrinsics(calibration_path: Path, resize_scale: float = 1.0, camera: str = "left") -> dict:
    with calibration_path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    left = calib[camera]
    return {
        "fx": float(left["fx"]) * resize_scale,
        "fy": float(left["fy"]) * resize_scale,
        "cx": float(left["cx"]) * resize_scale,
        "cy": float(left["cy"]) * resize_scale,
    }


def load_wuxi_intrinsics(calibration_path: Path, resize_scale: float) -> dict:
    return load_camera_intrinsics(calibration_path, resize_scale=resize_scale, camera="left")


def depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[ys, xs].astype(np.float32)
    colors = rgb[ys, xs].reshape(-1, 3)

    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    xs = xs.astype(np.float32)[valid]
    ys = ys.astype(np.float32)[valid]
    z = z[valid]
    colors = colors[valid.reshape(-1)]

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
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


def dataset_defaults(args: argparse.Namespace) -> tuple[Path, Path, dict, float]:
    if args.dataset == "ropedia":
        depth_dir = Path(args.depth_dir or "tmp/ropedia_foundation_depth_continuous_000000_000029")
        depth_path = depth_dir / f"frame_{args.frame:06d}_foundation_depth_m.npy"
        video_path = Path(args.video or "data/ropedia_data/stereo_left.mp4")
        return depth_path, video_path, dict(ROPEDIA_K), 1.0

    depth_dir = Path(args.depth_dir or "tmp/wuxi_foundation_depth_continuous_000000_000029")
    depth_path = depth_dir / f"frame_{args.frame:06d}_depth_m.npy"
    video_path = Path(args.video or "data/wuxi/stereo_left.mp4")
    resize_scale = args.resize_scale
    intrinsics = load_wuxi_intrinsics(Path(args.calibration), resize_scale)
    return depth_path, video_path, intrinsics, resize_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a depth map as a colored point cloud PLY.")
    parser.add_argument("--dataset", choices=["ropedia", "wuxi"], required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--depth_dir", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--calibration", default="data/wuxi/camera_intrinsics.json")
    parser.add_argument("--resize_scale", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_path, video_path, intrinsics, resize_scale = dataset_defaults(args)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = depth_path.parent / f"frame_{args.frame:06d}_{args.dataset}_pointcloud.ply"

    depth_m = np.load(depth_path).astype(np.float32)
    rgb = read_rgb_frame(video_path, args.frame, resize_scale)
    points, colors = depth_to_pointcloud(
        depth_m=depth_m,
        rgb=rgb,
        intrinsics=intrinsics,
        stride=max(1, args.stride),
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        max_points=args.max_points,
    )
    write_ply(output_path, points, colors)

    print(f"Depth: {depth_path}")
    print(f"RGB: {video_path}")
    print(f"Intrinsics: {intrinsics}")
    print(f"Points: {len(points)}")
    print(f"PLY: {output_path}")


if __name__ == "__main__":
    main()
