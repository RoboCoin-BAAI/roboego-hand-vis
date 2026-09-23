from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

try:
    from ..common.depth_io import (
        depth_path_for_frame,
        depth_uint16_png_path_for_frame,
        find_depth_path_for_frame,
        save_depth_uint16_png,
    )
    from ..common.stereo_manifest import load_foundation_inputs_from_manifest, manifest_meta
    from .visualization import make_depth_preview_panel_rgb
except ImportError:
    from common.depth_io import (
        depth_path_for_frame,
        depth_uint16_png_path_for_frame,
        find_depth_path_for_frame,
        save_depth_uint16_png,
    )
    from common.stereo_manifest import load_foundation_inputs_from_manifest, manifest_meta
    from visualization import make_depth_preview_panel_rgb


@dataclass(frozen=True)
class FastFoundationDepthConfig:
    manifest: Path
    output_dir: Path
    fast_foundation_root: Path = Path("../Fast-FoundationStereo")
    ckpt: Path = Path("../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth")
    start: int = 0
    num_frames: int | None = None
    scale: float = 0.5
    hiera: int = 0
    valid_iters: int = 4
    max_disp: int = 192
    min_disp: float = 1e-3
    max_depth_m: float = 4.0
    save_depth_png: bool = True
    save_npy: bool = False
    save_vis: bool = True
    preview_every_n_frames: int = 5
    progress_every_n_frames: int = 20
    verbose: bool = False
    overwrite: bool = False


class SequentialFrameReader:
    """Read a continuous frame range without seeking before every frame."""

    def __init__(self, cap: cv2.VideoCapture, start_frame: int, video_path: Path | str) -> None:
        self.cap = cap
        self.next_frame = int(start_frame)
        self.video_path = Path(video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.next_frame)

    def read_next(self) -> np.ndarray:
        frame_idx = self.next_frame
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_idx} from {self.video_path}")
        self.next_frame += 1
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def read_frame(cap: cv2.VideoCapture, frame_idx: int, video_path: Path) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def should_write_preview_frame(frame_idx: int, *, start_frame: int, every_n_frames: int) -> bool:
    if every_n_frames <= 0:
        raise ValueError("preview_every_n_frames must be > 0.")
    return (int(frame_idx) - int(start_frame)) % int(every_n_frames) == 0


def preview_video_fps(*, source_fps: float, every_n_frames: int) -> float:
    if every_n_frames <= 0:
        raise ValueError("preview_every_n_frames must be > 0.")
    if source_fps <= 0:
        return 6.0
    return float(source_fps) / float(every_n_frames)


def should_print_progress(
    frame_idx: int,
    *,
    start_frame: int,
    num_frames: int,
    every_n_frames: int,
    verbose: bool,
) -> bool:
    if verbose:
        return True
    if every_n_frames <= 0:
        return False
    offset = int(frame_idx) - int(start_frame)
    if offset == 0:
        return True
    if offset == int(num_frames) - 1:
        return True
    return (offset + 1) % int(every_n_frames) == 0


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


def to_jsonable(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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


def run_fast_foundation_depth(config: FastFoundationDepthConfig) -> dict[str, Any]:
    if not (0 < config.scale <= 1.0):
        raise ValueError("--scale must be in (0, 1].")
    if config.num_frames is not None and config.num_frames <= 0:
        raise ValueError("num_frames must be > 0 or None(auto).")
    if config.preview_every_n_frames <= 0:
        raise ValueError("--preview_every_n_frames must be > 0.")
    if config.progress_every_n_frames < 0:
        raise ValueError("--progress_every_n_frames must be >= 0.")
    if not config.save_depth_png and not config.save_npy:
        raise ValueError("At least one of --save_depth_png or --save_npy must be enabled.")

    torch.autograd.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Fast-FoundationStereo requires CUDA for this script.")

    root = config.fast_foundation_root.resolve()
    ckpt_path = config.ckpt.resolve()
    model, InputPadder, amp_dtype, model_args = load_model(
        root=root,
        ckpt_path=ckpt_path,
        valid_iters=config.valid_iters,
        max_disp=config.max_disp,
        device=device,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if config.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_foundation_inputs_from_manifest(config.manifest)
    left_cap = cv2.VideoCapture(str(inputs.left_video))
    right_cap = cv2.VideoCapture(str(inputs.right_video))
    if not left_cap.isOpened():
        raise RuntimeError(f"Cannot open left video: {inputs.left_video}")
    if not right_cap.isOpened():
        raise RuntimeError(f"Cannot open right video: {inputs.right_video}")

    # num_frames=auto 时，直接使用左目视频剩余帧数
    left_frame_count = int(
        left_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    num_frames = (
        left_frame_count - config.start
        if config.num_frames is None
        else config.num_frames
    )

    end_frame = config.start + num_frames

    preview_fps = preview_video_fps(
        source_fps=float(left_cap.get(cv2.CAP_PROP_FPS) or 0.0),
        every_n_frames=config.preview_every_n_frames,
    )

    meta = {
        **manifest_meta(inputs),
        "backend": "fast_foundation_stereo",
        "fast_foundation_root": str(root),
        "ckpt": str(ckpt_path),
        "scale": config.scale,
        "valid_iters": config.valid_iters,
        "max_disp": config.max_disp,
        "hiera": config.hiera,
        "start": config.start,
        "num_frames": config.num_frames,
        "model_args": to_jsonable(model_args),
        "depth_filename_pattern": "depth_uint16_png/frame_%06d_depth_mm.png",
        "depth_unit": "millimeter",
        "depth_encoding": "uint16_png",
        "invalid_value": 0,
        "save_depth_png": bool(config.save_depth_png),
        "save_npy": bool(config.save_npy),
        "save_vis": bool(config.save_vis),
        "preview_every_n_frames": int(config.preview_every_n_frames),
        "preview_fps": float(preview_fps),
        "preview_video": str(vis_dir / "depth_preview.mp4") if config.save_vis else None,
        "progress_every_n_frames": int(config.progress_every_n_frames),
        "verbose": bool(config.verbose),
    }
    meta_path = output_dir / "fast_foundation_stereo_video_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    processed_frames: list[int] = []
    preview_writer: cv2.VideoWriter | None = None
    preview_frame_count = 0
    left_reader = SequentialFrameReader(left_cap, config.start, inputs.left_video)
    right_reader = SequentialFrameReader(right_cap, config.start, inputs.right_video)
    try:
        for frame_idx in range(config.start, end_frame):
            depth_path = depth_path_for_frame(output_dir, frame_idx)
            depth_png_path = depth_uint16_png_path_for_frame(output_dir, frame_idx)
            existing_depth_path = find_depth_path_for_frame(output_dir, frame_idx)
            if existing_depth_path.exists() and not config.overwrite:
                continue

            t0 = time.perf_counter()
            left_rgb = left_reader.read_next()
            right_rgb = right_reader.read_next()
            if left_rgb.shape[:2] != right_rgb.shape[:2]:
                right_rgb = cv2.resize(right_rgb, (left_rgb.shape[1], left_rgb.shape[0]), interpolation=cv2.INTER_AREA)

            h0, w0 = left_rgb.shape[:2]
            if abs(config.scale - 1.0) > 1e-6:
                left_in = cv2.resize(left_rgb, None, fx=config.scale, fy=config.scale, interpolation=cv2.INTER_AREA)
                right_in = cv2.resize(right_rgb, None, fx=config.scale, fy=config.scale, interpolation=cv2.INTER_AREA)
            else:
                left_in = left_rgb
                right_in = right_rgb

            img0 = torch.as_tensor(left_in, device=device).float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right_in, device=device).float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0, img1 = padder.pad(img0, img1)

            with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                if config.hiera:
                    disp = model.run_hierachical(img0, img1, iters=config.valid_iters, test_mode=True, small_ratio=0.5)
                else:
                    disp = model.forward(img0, img1, iters=config.valid_iters, test_mode=True, optimize_build_volume="pytorch1")
            disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(left_in.shape[:2]).clip(0, None)

            depth_m = disparity_to_depth_original_size(
                disp,
                original_hw=(h0, w0),
                fx=inputs.intrinsics["fx"],
                baseline_m=inputs.baseline_m,
                scale=config.scale,
                min_disp=config.min_disp,
                max_depth_m=config.max_depth_m,
            )
            saved_paths: list[str] = []
            if config.save_npy:
                np.save(depth_path, depth_m)
                saved_paths.append(str(depth_path))
            if config.save_depth_png:
                if not save_depth_uint16_png(depth_png_path, depth_m, max_depth_m=config.max_depth_m):
                    raise RuntimeError(f"Failed to write depth PNG: {depth_png_path}")
                saved_paths.append(str(depth_png_path))

            if config.save_vis and should_write_preview_frame(
                frame_idx,
                start_frame=config.start,
                every_n_frames=config.preview_every_n_frames,
            ):
                panel = make_depth_preview_panel_rgb(left_rgb, depth_m, config.max_depth_m)
                if preview_writer is None:
                    height, width = panel.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    preview_writer = cv2.VideoWriter(
                        str(vis_dir / "depth_preview.mp4"),
                        fourcc,
                        preview_fps,
                        (int(width), int(height)),
                    )
                    if not preview_writer.isOpened():
                        raise RuntimeError(f"Cannot open preview video writer: {vis_dir / 'depth_preview.mp4'}")
                preview_writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
                preview_frame_count += 1
            processed_frames.append(frame_idx)
            if should_print_progress(
                frame_idx,
                start_frame=config.start,
                num_frames=num_frames,
                every_n_frames=config.progress_every_n_frames,
                verbose=config.verbose,
            ):
                done = frame_idx - config.start + 1
                print(
                    f"[{done}/{num_frames}] frame={frame_idx} elapsed_s={time.perf_counter() - t0:.3f} "
                    f"saved={len(saved_paths)}",
                    flush=True,
                )
    finally:
        if preview_writer is not None:
            preview_writer.release()
        left_cap.release()
        right_cap.release()

    return {
        "stage": "fast_foundation_depth",
        "status": "complete",
        "manifest": str(config.manifest),
        "output_dir": str(output_dir),
        "meta_path": str(meta_path),
        "processed_frame_count": len(processed_frames),
        "processed_frames": processed_frames,
        "preview_video": str(vis_dir / "depth_preview.mp4") if config.save_vis else None,
        "preview_frame_count": preview_frame_count,
    }
