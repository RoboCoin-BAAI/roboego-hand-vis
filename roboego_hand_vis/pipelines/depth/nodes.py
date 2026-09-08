"""Nodes for depth estimation pipelines."""

from __future__ import annotations

import json
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
    return data_root / "depth" / "foundation"


def _default_fast_depth_output_dir(manifest: Path) -> Path:
    data_root = manifest.parent.parent if manifest.parent.name == "dataset_manifest" else manifest.parent
    return data_root / "depth" / "fast_foundation"


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
    stereo_quality = _load_stereo_depth_gate(config.get("stereo_quality_report"))
    if stereo_quality is not None and not stereo_quality["usable_for_depth"]:
        return {
            "stage": "fast_foundation_stereo_depth",
            "status": "blocked",
            "reason": "stereo_quality_not_usable_for_depth",
            "manifest": str(manifest),
            "output_dir": str(output_dir),
            **stereo_quality,
        }
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
    if bool(config.get("save_depth_png", True)):
        args.append("--save_depth_png")
    else:
        args.append("--no-save_depth_png")
    if bool(config.get("save_npy", False)):
        args.append("--save_npy")
    else:
        args.append("--no-save_npy")
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
        **(stereo_quality or {}),
    }


def _load_stereo_depth_gate(value: Any) -> dict[str, Any] | None:
    path_str = _optional_str(value)
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_file():
        return {
            "stereo_quality_report": str(path),
            "stereo_quality_passed": None,
            "usable_for_depth": False,
            "depth_risk_level": "blocked",
            "depth_risk_reasons": ["missing_stereo_quality_report"],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    strict_passed = bool(summary.get("strict_passed", summary.get("passed", False)))
    usable_for_depth = bool(summary.get("usable_for_depth", strict_passed))
    return {
        "stereo_quality_report": str(path),
        "stereo_quality_passed": strict_passed,
        "usable_for_depth": usable_for_depth,
        "depth_risk_level": str(summary.get("depth_risk_level") or ("low" if strict_passed else "blocked")),
        "depth_risk_reasons": summary.get("depth_risk_reasons", []),
    }


def _load_required_stage_status(state_path: Path, stage: str) -> str:
    if not state_path.is_file():
        return "missing_state"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    stages = payload.get("stages") if isinstance(payload, dict) else {}
    if not isinstance(stages, dict):
        return "missing_stage"
    stage_state = stages.get(stage)
    if not isinstance(stage_state, dict):
        return "missing_stage"
    return str(stage_state.get("status") or "missing_status")


def run_archive_depth_cache_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "archive_depth_cache", "status": "skipped", "reason": "disabled"}

    depth_dir = Path(_required_value(config, "depth_dir", "archive_depth_cache"))
    output_dir = Path(_required_value(config, "output_dir", "archive_depth_cache"))
    depth_stage_state = _optional_str(config.get("depth_stage_state"))
    required_depth_stage = _optional_str(config.get("required_depth_stage")) or "fast_foundation_depth"
    if depth_stage_state:
        state_path = Path(depth_stage_state).expanduser()
        depth_stage_status = _load_required_stage_status(state_path, required_depth_stage)
        if depth_stage_status != "success":
            return {
                "stage": "archive_depth_cache",
                "status": "blocked",
                "reason": "required_depth_stage_not_success",
                "required_depth_stage": required_depth_stage,
                "depth_stage_status": depth_stage_status,
                "depth_stage_state": str(state_path),
                "depth_dir": str(depth_dir),
                "output_dir": str(output_dir),
            }
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/depth/archive_depth_cache.py",
        "--depth_dir",
        str(depth_dir),
        "--output_dir",
        str(output_dir),
    ]
    _append_arg(args, "--max_depth_m", config.get("max_depth_m", 4.0))
    _append_arg(args, "--invalid_value", config.get("invalid_value", 0))
    _append_arg(args, "--fps", config.get("fps", 30.0))
    _append_arg(args, "--ffmpeg_bin", config.get("ffmpeg_bin"))
    if bool(config.get("write_png", True)):
        args.append("--write_png")
    else:
        args.append("--no-write_png")
    if bool(config.get("write_avi", True)):
        args.append("--write_avi")
    else:
        args.append("--no-write_avi")
    if bool(config.get("write_preview", True)):
        args.append("--write_preview")
    else:
        args.append("--no-write_preview")
    _append_arg(args, "--preview_mode", config.get("preview_mode", "near"))
    _append_arg(args, "--preview_near_min_m", config.get("preview_near_min_m", 0.15))
    _append_arg(args, "--preview_near_max_m", config.get("preview_near_max_m", 3.0))
    _append_arg(args, "--preview_contrast_percentile", config.get("preview_contrast_percentile", 2.0))
    _append_arg(args, "--preview_codec", config.get("preview_codec"))
    _append_arg(args, "--ffv1_container", config.get("ffv1_container", "avi"))
    if bool(config.get("dry_run_avi", False)):
        args.append("--dry_run_avi")
    if bool(config.get("delete_npy", False)):
        args.append("--delete_npy")
    if bool(config.get("delete_png", False)):
        args.append("--delete_png")
    if bool(config.get("write_hevc12", False)):
        args.append("--write_hevc12")
    else:
        args.append("--no-write_hevc12")
    _append_arg(args, "--hevc12_crf", config.get("hevc12_crf", 0))
    _append_arg(args, "--hevc12_preset", config.get("hevc12_preset", "slow"))
    if bool(config.get("hevc12_lossless", True)):
        args.append("--hevc12_lossless")
    else:
        args.append("--no-hevc12_lossless")
    if bool(config.get("write_valid_mask", False)):
        args.append("--write_valid_mask")
    else:
        args.append("--no-write_valid_mask")

    subprocess.run(args, check=True)
    return {
        "stage": "archive_depth_cache",
        "status": "complete",
        "depth_dir": str(depth_dir),
        "output_dir": str(output_dir),
        "metadata_json": str(output_dir / "depth_archive_meta.json"),
    }
