#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.depth.foundation import load_foundation_inputs_from_manifest, manifest_meta
from roboego_hand_vis.depth.io import depth_path_for_frame, depth_uint16_png_path_for_frame, find_depth_path_for_frame, save_depth_uint16_png


def read_frame(cap: cv2.VideoCapture, frame_idx: int, video_path: Path) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
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


def make_depth_preview_panel_rgb(left_rgb: np.ndarray, depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth_vis_rgb = cv2.cvtColor(depth_to_vis(depth_m, max_depth_m), cv2.COLOR_BGR2RGB)
    return np.concatenate([left_rgb, depth_vis_rgb], axis=1)


def to_jsonable(value):
    if OmegaConf.is_config(value):
        return to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fast-FoundationStereo on left/right videos and save depth in meters.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fast_foundation_root", default="/home/liuyou/Documents/Fast-FoundationStereo")
    parser.add_argument(
        "--ckpt",
        default="/home/liuyou/Documents/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--hiera", type=int, default=0)
    parser.add_argument("--valid_iters", type=int, default=4)
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--min_disp", type=float, default=1e-3)
    parser.add_argument("--max_depth_m", type=float, default=4.0)
    parser.add_argument("--save_depth_png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_npy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(*, root: Path, ckpt_path: Path, valid_iters: int, max_disp: int, device: torch.device):
    sys.path.insert(0, str(root))
    from core.utils.utils import InputPadder  # noqa: WPS433
    from Utils import AMP_DTYPE  # noqa: WPS433

    cfg_path = ckpt_path.parent / "cfg.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["valid_iters"] = int(valid_iters)
    cfg["max_disp"] = int(max_disp)
    args = OmegaConf.create(cfg)
    model = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = int(valid_iters)
    model.args.max_disp = int(max_disp)
    model.to(device).eval()
    return model, InputPadder, AMP_DTYPE, args


def main() -> None:
    args = parse_args()
    if not (0 < args.scale <= 1.0):
        raise ValueError("--scale must be in (0, 1].")
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be > 0.")
    if not args.save_depth_png and not args.save_npy:
        raise ValueError("At least one of --save_depth_png or --save_npy must be enabled.")

    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Fast-FoundationStereo requires CUDA for this script.")

    root = Path(args.fast_foundation_root).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    model, InputPadder, amp_dtype, model_args = load_model(
        root=root,
        ckpt_path=ckpt_path,
        valid_iters=args.valid_iters,
        max_disp=args.max_disp,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_foundation_inputs_from_manifest(Path(args.manifest))
    left_cap = cv2.VideoCapture(str(inputs.left_video))
    right_cap = cv2.VideoCapture(str(inputs.right_video))
    if not left_cap.isOpened():
        raise RuntimeError(f"Cannot open left video: {inputs.left_video}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Cannot open right video: {inputs.right_video}")

    meta = {
        **manifest_meta(inputs),
        "backend": "fast_foundation_stereo",
        "fast_foundation_root": str(root),
        "ckpt": str(ckpt_path),
        "scale": args.scale,
        "valid_iters": args.valid_iters,
        "max_disp": args.max_disp,
        "hiera": args.hiera,
        "start": args.start,
        "num_frames": args.num_frames,
        "model_args": to_jsonable(model_args),
        "depth_filename_pattern": "depth_uint16_png/frame_%06d_depth_mm.png",
        "depth_unit": "millimeter",
        "depth_encoding": "uint16_png",
        "invalid_value": 0,
        "save_depth_png": bool(args.save_depth_png),
        "save_npy": bool(args.save_npy),
    }
    (output_dir / "fast_foundation_stereo_video_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        for frame_idx in range(args.start, args.start + args.num_frames):
            depth_path = depth_path_for_frame(output_dir, frame_idx)
            depth_png_path = depth_uint16_png_path_for_frame(output_dir, frame_idx)
            existing_depth_path = find_depth_path_for_frame(output_dir, frame_idx)
            if existing_depth_path.exists() and not args.overwrite:
                continue

            t0 = time.perf_counter()
            left_rgb = read_frame(left_cap, frame_idx, inputs.left_video)
            right_rgb = read_frame(right_cap, frame_idx, inputs.right_video)
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

            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                if args.hiera:
                    disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
                else:
                    disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
            disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(left_in.shape[:2]).clip(0, None)

            depth_m = disparity_to_depth_original_size(
                disp,
                original_hw=(h0, w0),
                fx=inputs.intrinsics["fx"],
                baseline_m=inputs.baseline_m,
                scale=args.scale,
                min_disp=args.min_disp,
                max_depth_m=args.max_depth_m,
            )
            saved_paths: list[str] = []
            if args.save_npy:
                np.save(depth_path, depth_m)
                saved_paths.append(str(depth_path))
            if args.save_depth_png:
                if not save_depth_uint16_png(depth_png_path, depth_m, max_depth_m=args.max_depth_m):
                    raise RuntimeError(f"Failed to write depth PNG: {depth_png_path}")
                saved_paths.append(str(depth_png_path))

            if args.save_vis:
                panel = make_depth_preview_panel_rgb(left_rgb, depth_m, args.max_depth_m)
                imageio.imwrite(vis_dir / f"frame_{frame_idx:06d}_depth_vis.jpg", panel)
            print(f"[{frame_idx}] {','.join(saved_paths)} elapsed_s={time.perf_counter() - t0:.3f}", flush=True)
    finally:
        left_cap.release()
        right_cap.release()


if __name__ == "__main__":
    main()
