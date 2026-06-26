from __future__ import annotations

from typing import Iterable

import numpy as np
import rerun as rr


def side_mesh_entity(side: str, layer: str) -> str:
    return f"camera_opencv/hands/{side}/{layer}"


def rgb_overlay_entity(layer: str, side: str, name: str) -> str:
    return f"camera_opencv/camera/left/rgb/overlay/{layer}/{side}/{name}"


def log_mesh3d(entity_path: str, vertices: np.ndarray, faces: np.ndarray, color: Iterable[int]) -> None:
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
            albedo_factor=list(color),
        ),
    )


def log_points2d(entity_path: str, points: np.ndarray, color: Iterable[int], radius: float) -> None:
    rr.log(
        entity_path,
        rr.Points2D(
            points,
            colors=[list(color)] * len(points),
            radii=float(radius),
        ),
    )
