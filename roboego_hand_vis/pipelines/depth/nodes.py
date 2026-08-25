"""Nodes for depth estimation pipelines."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _required_value(config: dict[str, Any], key: str, namespace: str) -> str:
    value = _optional_str(config.get(key))
    if not value:
        raise ValueError(f"{namespace}.{key} is required when enabled")
    return value


def _append_arg(args: list[str], name: str, value: Any) -> None:
    value_str = _optional_str(value)
    if value_str:
        args.extend([name, value_str])


def _default_depth_output_dir(manifest: Path) -> Path:
    data_root = manifest.parent.parent if manifest.parent.name == "dataset_manifest" else manifest.parent
    return data_root / "foundation_depth"


def _default_fast_depth_output_dir(manifest: Path) -> Path:
    data_root = manifest.parent.parent if manifest.parent.name == "dataset_manifest" else manifest.parent
    return data_root / "fast_foundation_depth"


def run_foundation_stereo_depth_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "foundation_stereo_depth", "status": "skipped", "reason": "disabled"}

    manifest = Path(_required_value(config, "manifest", "foundation_stereo_depth"))
    output_dir = Path(_optional_str(config.get("output_dir")) or _default_depth_output_dir(manifest))
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/run_foundation_stereo_video.py",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
    ]
    _append_arg(args, "--foundation_root", config.get("foundation_root"))
    _append_arg(args, "--ckpt", config.get("ckpt"))
    _append_arg(args, "--start", config.get("start", 0))
    _append_arg(args, "--num_frames", config.get("num_frames", 300))
    _append_arg(args, "--scale", config.get("scale", 0.5))
    _append_arg(args, "--valid_iters", config.get("valid_iters", 32))
    _append_arg(args, "--hiera", config.get("hiera"))
    _append_arg(args, "--min_disp", config.get("min_disp"))
    _append_arg(args, "--max_depth_m", config.get("max_depth_m", 4.0))
    if bool(config.get("save_vis", True)):
        args.append("--save_vis")
    else:
        args.append("--no-save_vis")
    if bool(config.get("overwrite", False)):
        args.append("--overwrite")

    subprocess.run(args, check=True)
    return {
        "stage": "foundation_stereo_depth",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "meta_path": str(output_dir / "foundation_stereo_video_meta.json"),
    }


def run_fast_foundation_stereo_depth_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "fast_foundation_stereo_depth", "status": "skipped", "reason": "disabled"}

    manifest = Path(_required_value(config, "manifest", "fast_foundation_stereo_depth"))
    output_dir = Path(_optional_str(config.get("output_dir")) or _default_fast_depth_output_dir(manifest))
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/run_fast_foundation_stereo_video.py",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
    ]
    _append_arg(args, "--fast_foundation_root", config.get("fast_foundation_root"))
    _append_arg(args, "--ckpt", config.get("ckpt"))
    _append_arg(args, "--start", config.get("start", 0))
    _append_arg(args, "--num_frames", config.get("num_frames", 300))
    _append_arg(args, "--scale", config.get("scale", 0.5))
    _append_arg(args, "--valid_iters", config.get("valid_iters", 4))
    _append_arg(args, "--max_disp", config.get("max_disp", 192))
    _append_arg(args, "--hiera", config.get("hiera"))
    _append_arg(args, "--min_disp", config.get("min_disp"))
    _append_arg(args, "--max_depth_m", config.get("max_depth_m", 4.0))
    if bool(config.get("save_vis", True)):
        args.append("--save_vis")
    else:
        args.append("--no-save_vis")
    if bool(config.get("overwrite", False)):
        args.append("--overwrite")

    subprocess.run(args, check=True)
    return {
        "stage": "fast_foundation_stereo_depth",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "meta_path": str(output_dir / "fast_foundation_stereo_video_meta.json"),
    }
