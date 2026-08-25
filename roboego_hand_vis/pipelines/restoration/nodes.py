"""Nodes for video restoration pipelines."""

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


def _default_deblur_output_dir(manifest: Path) -> Path:
    data_root = manifest.parent.parent if manifest.parent.name == "dataset_manifest" else manifest.parent
    return data_root / "deblur_realvdeblur"


def run_realvdeblur_stereo_node(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", False)):
        return {"stage": "realvdeblur_stereo", "status": "skipped", "reason": "disabled"}

    manifest = Path(_required_value(config, "manifest", "realvdeblur_stereo"))
    output_dir = Path(_optional_str(config.get("output_dir")) or _default_deblur_output_dir(manifest))
    args = [
        _optional_str(config.get("python")) or "python",
        _optional_str(config.get("script")) or "scripts/restoration/run_realvdeblur_stereo_manifest.py",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
    ]
    _append_arg(args, "--realvdeblur_python", config.get("realvdeblur_python"))
    _append_arg(args, "--realvdeblur_root", config.get("realvdeblur_root"))
    _append_arg(args, "--wan_model_dir", config.get("wan_model_dir"))
    _append_arg(args, "--checkpoint", config.get("checkpoint"))
    _append_arg(args, "--height", config.get("height"))
    _append_arg(args, "--width", config.get("width"))
    _append_arg(args, "--device", config.get("device"))
    _append_arg(args, "--dtype", config.get("dtype"))
    _append_arg(args, "--num_inference_steps", config.get("num_inference_steps"))
    if bool(config.get("enable_twm", False)):
        args.append("--enable_twm")
        _append_arg(args, "--temporal_window_size", config.get("temporal_window_size"))
    _append_arg(args, "--codec", config.get("codec"))
    if bool(config.get("overwrite", False)):
        args.append("--overwrite")

    subprocess.run(args, check=True)
    return {
        "stage": "realvdeblur_stereo",
        "status": "complete",
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "deblur_manifest": str(output_dir / "dataset_manifest" / "dataset_manifest.json"),
    }
