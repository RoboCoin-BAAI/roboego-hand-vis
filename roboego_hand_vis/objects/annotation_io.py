from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .segmentation import ObjectObservation


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def observation_to_json(obs: ObjectObservation, output_dir: Path) -> Dict[str, Any]:
    return {
        "frame_idx": int(obs.frame_idx),
        "bbox_xywh": obs.bbox_xywh,
        "mask_path": _rel(obs.mask_path, output_dir),
        "crop_path": _rel(obs.crop_path, output_dir),
        "mask_area_px": int(obs.mask_area_px),
        "depth_median_m": obs.depth_median_m,
        "depth_valid_ratio": obs.depth_valid_ratio,
        "source": obs.source,
    }


def write_object_tracks_json(
    *,
    observations: Dict[str, List[ObjectObservation]],
    output_path: Path,
    output_dir: Path,
    backend: str,
    video_path: Path,
    depth_dir: Path,
    warnings: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "backend": backend,
        "video": str(video_path),
        "depth_dir": str(depth_dir),
        "warnings": warnings or [],
        "objects": {
            object_id: {"frames": [observation_to_json(obs, output_dir) for obs in frames]}
            for object_id, frames in observations.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_object_tracks_csv(observations: Dict[str, List[ObjectObservation]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "object_id",
                "bbox_xywh",
                "mask_area_px",
                "depth_median_m",
                "depth_valid_ratio",
                "source",
            ],
        )
        writer.writeheader()
        for object_id, frames in observations.items():
            for obs in frames:
                writer.writerow(
                    {
                        "frame_idx": obs.frame_idx,
                        "object_id": object_id,
                        "bbox_xywh": obs.bbox_xywh,
                        "mask_area_px": obs.mask_area_px,
                        "depth_median_m": obs.depth_median_m,
                        "depth_valid_ratio": obs.depth_valid_ratio,
                        "source": obs.source,
                    }
                )


def write_json_payload(payload: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
