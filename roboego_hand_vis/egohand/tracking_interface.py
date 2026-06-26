from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATUS_LAYERS = {
    "detected": "normal",
    "normal": "normal",
    "recovered": "recovered",
    "interpolated": "smooth",
    "smooth": "smooth",
}


@dataclass(frozen=True)
class TrackingInputs:
    manifest_path: Path
    video_path: Path
    dataset_name: str | None
    sequence_name: str | None
    reference_camera: str
    video_info: dict[str, Any]


def load_tracking_inputs_from_manifest(manifest_path: Path) -> TrackingInputs:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_camera = manifest.get("reference_camera", "left")
    if reference_camera != "left":
        raise ValueError(f"Only left reference camera is currently supported, got: {reference_camera}")
    left_video = manifest["videos"]["left"]
    return TrackingInputs(
        manifest_path=manifest_path,
        video_path=Path(left_video["path"]).expanduser().resolve(),
        dataset_name=manifest.get("dataset_name"),
        sequence_name=manifest.get("sequence_name"),
        reference_camera=reference_camera,
        video_info=dict(left_video),
    )


def status_layer(obs: dict[str, Any] | None) -> str:
    if obs is None:
        return "missing"
    return STATUS_LAYERS.get(str(obs.get("status", "detected")), "normal")


def add_layers_to_tracks(
    tracks: dict[str, list[dict[str, Any] | None]],
) -> dict[str, list[dict[str, Any] | None]]:
    out: dict[str, list[dict[str, Any] | None]] = {}
    for side, track in tracks.items():
        out[side] = []
        for obs in track:
            if obs is None:
                out[side].append(None)
                continue
            enriched = dict(obs)
            enriched["layer"] = status_layer(obs)
            enriched["current"] = True
            out[side].append(enriched)
    return out


def build_track_report_payload(
    *,
    inputs: TrackingInputs | None,
    alignment_report: Path,
    depth_dir: Path,
    output_dir: Path,
    parameters: dict[str, Any],
    summary: dict[str, Any],
    tracks: dict[str, list[dict[str, Any] | None]],
    fallback_video: Path | None = None,
) -> dict[str, Any]:
    video = inputs.video_path if inputs is not None else fallback_video
    legacy_inputs = {
        "alignment_report": str(alignment_report.resolve()),
        "depth_dir": str(depth_dir.resolve()),
        "video": str(video.resolve()) if video is not None else None,
    }
    return {
        "schema_version": "1.0",
        "inputs": legacy_inputs,
        "manifest_path": str(inputs.manifest_path) if inputs is not None else None,
        "dataset_name": inputs.dataset_name if inputs is not None else None,
        "sequence_name": inputs.sequence_name if inputs is not None else None,
        "reference_camera": inputs.reference_camera if inputs is not None else "left",
        "video": str(video.resolve()) if video is not None else None,
        "video_info": inputs.video_info if inputs is not None else None,
        "alignment_report": str(alignment_report.resolve()),
        "depth_dir": str(depth_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "status_layers": {
            "detected": "normal",
            "recovered": "recovered",
            "interpolated": "smooth",
            "missing": "missing",
            "current": "per-frame unified best hand result consumed by downstream RRD/evaluation",
        },
        "parameters": parameters,
        "summary": summary,
        "tracks": add_layers_to_tracks(tracks),
    }
