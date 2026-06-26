#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth.foundation import load_foundation_inputs_from_manifest, manifest_meta


def load_intrinsics(path: Path, camera: str) -> tuple[dict, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cam = data[camera]
    baseline = float(data["baseline_m"])
    return {
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
    }, baseline


def read_frame(cap: cv2.VideoCapture, frame_idx: int, video_path: Path) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def disparity_to_depth_original_size(
    disp_scaled: np.ndarray,
    original_hw: tuple[int, int],
    fx: float,
    baseline_m: float,
    scale: float,
    min_disp: float,
    max_depth_m: float,
) -> np.ndarray:
    h0, w0 = original_hw
    if disp_scaled.shape != (h0, w0):
        disp = cv2.resize(disp_scaled, (w0, h0), interpolation=cv2.INTER_LINEAR)
    else:
        disp = disp_scaled
    valid = np.isfinite(disp) & (disp > min_disp)
    depth = np.full((h0, w0), np.nan, dtype=np.float32)
    depth[valid] = float(scale) * float(fx) * float(baseline_m) / disp[valid]
    depth[(depth <= 0) | (depth > max_depth_m)] = np.nan
    return depth.astype(np.float32)


def depth_to_vis(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m)
    out = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, 0.0, max_depth_m)
        out[valid] = (255.0 * (1.0 - clipped[valid] / max_depth_m)).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FoundationStereo on left/right videos and save depth in meters.")
    parser.add_argument("--manifest", default="", help="Optional dataset_manifest.json. Overrides video/intrinsics args.")
    parser.add_argument("--left_video", default="")
    parser.add_argument("--right_video", default="")
    parser.add_argument("--intrinsics_json", default="")
    parser.add_argument("--intrinsics_camera", default="left")
    parser.add_argument("--foundation_root", default="/home/liuyou/Documents/FoundationStereo")
    parser.add_argument(
        "--ckpt",
        default="/home/liuyou/Documents/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--valid_iters", type=int, default=32)
    parser.add_argument("--min_disp", type=float, default=1e-3)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, dict, float, dict]:
    if args.manifest:
        inputs = load_foundation_inputs_from_manifest(Path(args.manifest))
        return inputs.left_video, inputs.right_video, inputs.intrinsics, inputs.baseline_m, manifest_meta(inputs)
    if not args.left_video or not args.right_video or not args.intrinsics_json:
        raise ValueError("Either --manifest or all of --left_video, --right_video, --intrinsics_json are required.")
    intrinsics, baseline_m = load_intrinsics(Path(args.intrinsics_json), args.intrinsics_camera)
    return Path(args.left_video), Path(args.right_video), intrinsics, baseline_m, {
        "manifest_path": None,
        "left_video": str(Path(args.left_video).resolve()),
        "right_video": str(Path(args.right_video).resolve()),
        "intrinsics_json": str(Path(args.intrinsics_json).resolve()),
        "intrinsics_camera": args.intrinsics_camera,
        "intrinsics": intrinsics,
        "baseline_m": baseline_m,
        "reference_camera": "left",
    }


def main() -> None:
    args = parse_args()
    if not (0 < args.scale <= 1.0):
        raise ValueError("--scale must be in (0, 1].")

    foundation_root = Path(args.foundation_root).resolve()
    sys.path.insert(0, str(foundation_root))
    from core.foundation_stereo import FoundationStereo  # noqa: WPS433
    from core.utils.utils import InputPadder  # noqa: WPS433

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    left_path, right_path, intrinsics, baseline_m, input_meta = resolve_inputs(args)

    ckpt_path = Path(args.ckpt).resolve()
    cfg = OmegaConf.load(str(ckpt_path.parent / "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg.valid_iters = args.valid_iters

    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoundationStereo(cfg)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    left_cap = cv2.VideoCapture(str(left_path))
    right_cap = cv2.VideoCapture(str(right_path))
    if not left_cap.isOpened():
        raise RuntimeError(f"Cannot open left video: {left_path}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Cannot open right video: {right_path}")

    meta = {
        **input_meta,
        "scale": args.scale,
        "valid_iters": args.valid_iters,
        "hiera": args.hiera,
        "start": args.start,
        "num_frames": args.num_frames,
        "depth_filename_pattern": "frame_%06d_foundation_depth_m.npy",
    }
    (output_dir / "foundation_stereo_video_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    for frame_idx in range(args.start, args.start + args.num_frames):
        depth_path = output_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
        if depth_path.exists() and not args.overwrite:
            continue

        left_rgb = read_frame(left_cap, frame_idx, left_path)
        right_rgb = read_frame(right_cap, frame_idx, right_path)
        if left_rgb.shape[:2] != right_rgb.shape[:2]:
            right_rgb = cv2.resize(right_rgb, (left_rgb.shape[1], left_rgb.shape[0]), interpolation=cv2.INTER_AREA)

        h0, w0 = left_rgb.shape[:2]
        if abs(args.scale - 1.0) > 1e-6:
            left_in = cv2.resize(left_rgb, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            right_in = cv2.resize(right_rgb, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        else:
            left_in = left_rgb
            right_in = right_rgb

        img0 = torch.as_tensor(left_in, device=device).float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_in, device=device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if args.hiera:
                disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
            else:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
        disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(left_in.shape[:2])

        depth_m = disparity_to_depth_original_size(
            disp,
            original_hw=(h0, w0),
            fx=intrinsics["fx"],
            baseline_m=baseline_m,
            scale=args.scale,
            min_disp=args.min_disp,
            max_depth_m=args.max_depth_m,
        )
        np.save(depth_path, depth_m)

        if args.save_vis:
            vis = depth_to_vis(depth_m, args.max_depth_m)
            panel = np.concatenate([cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR), vis], axis=1)
            imageio.imwrite(vis_dir / f"frame_{frame_idx:06d}_depth_vis.jpg", panel)

        print(f"[{frame_idx}] {depth_path}")

    left_cap.release()
    right_cap.release()


if __name__ == "__main__":
    main()
