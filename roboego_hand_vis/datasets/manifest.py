from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import generic_stereo, ropedia, wuxi
from .common import ResolveConfig, as_abs, manifest_summary


RESOLVERS: dict[str, Callable[[ResolveConfig], dict[str, Any]]] = {
    "wuxi": wuxi.resolve,
    "ropedia": ropedia.resolve,
    "generic_stereo": generic_stereo.resolve,
}


def resolve_dataset_manifest(config: ResolveConfig) -> dict[str, Any]:
    dataset = config.dataset.lower()
    if dataset not in RESOLVERS:
        raise ValueError(f"Unsupported dataset '{config.dataset}'. Available: {sorted(RESOLVERS)}")
    manifest = RESOLVERS[dataset](config)
    manifest["outputs"] = {
        "manifest_path": str(as_abs(config.output_dir) / "dataset_manifest.json"),
        "summary_path": str(as_abs(config.output_dir) / "dataset_summary.txt"),
    }
    return manifest


def write_manifest(manifest: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir = as_abs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "dataset_manifest.json"
    summary_path = output_dir / "dataset_summary.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(manifest_summary(manifest), encoding="utf-8")
    return manifest_path, summary_path
