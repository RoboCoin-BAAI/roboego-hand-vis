from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from roboego_hand_vis.hand_depth_alignment.depth_mesh import (
    hamer_rendered_obj_to_camera_opencv,
    parse_obj_mesh,
    translate_points,
)
from roboego_hand_vis.hand_rerun_visualization.temporal_tracking import build_recovered_depth_overrides_from_frames
from linshi.scripts.rerun_utils import resolve_existing_artifact_path


def resolve_path(path: str, base_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base_dir / out
    return out.resolve()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def selected_candidates(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    selected = []
    for candidate in frame["candidates"]:
        if candidate.get("status") in {
            "ego_left",
            "ego_right",
            "ego_left_recovered",
            "ego_right_recovered",
        }:
            selected.append(candidate)
    selected.sort(key=lambda c: c.get("status", "").replace("_recovered", ""))
    return selected


def canonical_hand_status(status: str) -> str:
    if status == "ego_left_recovered":
        return "ego_left"
    if status == "ego_right_recovered":
        return "ego_right"
    return status


def load_depth(depth_dir: Path, frame_idx: int) -> np.ndarray:
    path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


def load_tracks_by_frame(path: Path | None) -> Dict[int, Dict[str, Dict[str, Any]]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tracks = data.get("tracks", {})
    params = data.get("parameters", {})
    start = int(params.get("start", 0))
    out: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for side, side_track in tracks.items():
        for offset, obs in enumerate(side_track):
            if obs is None:
                continue
            frame_idx = int(obs.get("frame_idx", start + offset))
            out.setdefault(frame_idx, {})[side] = obs
    return out


def build_recovered_depth_overrides(path: Path | None) -> Dict[tuple[int, str], Dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    return build_recovered_depth_overrides_from_frames(report.get("frames", []))


def write_recovered_depth_overrides(output_dir: Path, overrides: Dict[tuple[int, str], Dict[str, Any]]) -> None:
    if not overrides:
        return
    rows = [overrides[key] for key in sorted(overrides)]
    json_path = output_dir / "recovered_depth_overrides.json"
    csv_path = output_dir / "recovered_depth_overrides.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "frame_idx",
        "side",
        "observed_target_z_m",
        "target_z_m",
        "delta_from_observed_m",
        "mode",
        "prev_frame_idx",
        "next_frame_idx",
        "ratio",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def tracked_mesh_from_observation(obs: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh_vertices_obj, mesh_faces, _ = parse_obj_mesh(resolve_existing_artifact_path(obs["mesh_obj"]))
    vertices_camera = hamer_rendered_obj_to_camera_opencv(mesh_vertices_obj)
    vertices_smoothed = translate_points(vertices_camera, obs["smoothed_translation"])
    return vertices_smoothed, mesh_faces
