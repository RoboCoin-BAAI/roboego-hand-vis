from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HamerFilterInputs:
    manifest_path: Path
    video_path: Path
    dataset_name: str | None
    sequence_name: str | None
    reference_camera: str
    video_info: dict[str, Any]


def load_hamer_filter_inputs_from_manifest(manifest_path: Path) -> HamerFilterInputs:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_camera = manifest.get("reference_camera", "left")
    if reference_camera != "left":
        raise ValueError(f"Only left reference camera is currently supported, got: {reference_camera}")
    left_video = manifest["videos"]["left"]
    return HamerFilterInputs(
        manifest_path=manifest_path,
        video_path=Path(left_video["path"]).expanduser().resolve(),
        dataset_name=manifest.get("dataset_name"),
        sequence_name=manifest.get("sequence_name"),
        reference_camera=reference_camera,
        video_info=dict(left_video),
    )


def load_depth_quality_warnings(path: Path | None) -> dict[int, list[str]]:
    if path is None:
        return {}
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    warnings: dict[int, list[str]] = {}
    for frame in data.get("frames", []):
        frame_idx = int(frame.get("frame", frame.get("frame_idx", -1)))
        if frame_idx < 0:
            continue
        warnings[frame_idx] = list(frame.get("warnings", []))
    return warnings


def depth_quality_warnings_for_frame(frame_idx: int, warnings_by_frame: dict[int, list[str]]) -> list[str]:
    return list(warnings_by_frame.get(int(frame_idx), []))


def build_report_metadata(
    *,
    inputs: HamerFilterInputs | None,
    depth_dir: Path | None,
    depth_quality_report: Path | None,
    output_dir: Path,
    frame_range: dict[str, Any],
    filter_config: dict[str, Any],
    hamer_config: dict[str, Any],
    fallback_video: Path | None = None,
) -> dict[str, Any]:
    video = inputs.video_path if inputs is not None else fallback_video
    return {
        "schema_version": "1.0",
        "manifest_path": str(inputs.manifest_path) if inputs is not None else None,
        "dataset_name": inputs.dataset_name if inputs is not None else None,
        "sequence_name": inputs.sequence_name if inputs is not None else None,
        "reference_camera": inputs.reference_camera if inputs is not None else "left",
        "video": str(video.resolve()) if video is not None else None,
        "video_info": inputs.video_info if inputs is not None else None,
        "depth_dir": str(depth_dir.resolve()) if depth_dir else None,
        "depth_quality_report": str(depth_quality_report.resolve()) if depth_quality_report else None,
        "output_dir": str(output_dir.resolve()),
        "frame_range": frame_range,
        "filter_config": filter_config,
        "hamer_config": hamer_config,
        "frames": [],
    }
