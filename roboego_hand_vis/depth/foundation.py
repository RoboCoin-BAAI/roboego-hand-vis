from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FoundationStereoInputs:
    manifest_path: Path
    left_video: Path
    right_video: Path
    intrinsics: dict[str, float]
    baseline_m: float
    dataset_name: str | None
    sequence_name: str | None
    reference_camera: str


def load_foundation_inputs_from_manifest(manifest_path: Path) -> FoundationStereoInputs:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_camera = manifest.get("reference_camera", "left")
    if reference_camera != "left":
        raise ValueError(f"Only left reference camera is currently supported, got: {reference_camera}")

    left_video = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
    right_video = Path(manifest["videos"]["right"]["path"]).expanduser().resolve()
    left_intrinsics = manifest["intrinsics"]["left"]
    baseline = manifest.get("extrinsics", {}).get("baseline_m")
    if baseline is None:
        left_to_right = manifest.get("extrinsics", {}).get("left_to_right") or {}
        baseline = left_to_right.get("baseline_m")
    if baseline is None:
        raise ValueError("dataset_manifest.json must contain extrinsics.baseline_m for stereo depth.")

    intrinsics = {
        "fx": float(left_intrinsics["fx"]),
        "fy": float(left_intrinsics["fy"]),
        "cx": float(left_intrinsics["cx"]),
        "cy": float(left_intrinsics["cy"]),
    }
    return FoundationStereoInputs(
        manifest_path=manifest_path,
        left_video=left_video,
        right_video=right_video,
        intrinsics=intrinsics,
        baseline_m=float(baseline),
        dataset_name=manifest.get("dataset_name"),
        sequence_name=manifest.get("sequence_name"),
        reference_camera=reference_camera,
    )


def manifest_meta(inputs: FoundationStereoInputs) -> dict[str, Any]:
    return {
        "manifest_path": str(inputs.manifest_path),
        "dataset_name": inputs.dataset_name,
        "sequence_name": inputs.sequence_name,
        "left_video": str(inputs.left_video),
        "right_video": str(inputs.right_video),
        "intrinsics": inputs.intrinsics,
        "baseline_m": inputs.baseline_m,
        "reference_camera": inputs.reference_camera,
    }
