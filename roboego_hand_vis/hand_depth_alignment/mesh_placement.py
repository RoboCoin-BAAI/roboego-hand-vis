from __future__ import annotations

from typing import Any

import numpy as np

from .depth_anchor import DepthAnchorSelection
from .depth_mesh import fit_mesh_to_pixel_depth_anchor


def place_shape_preserved_mesh(
    *,
    mesh_vertices_camera: np.ndarray,
    projected_pixels: np.ndarray,
    depth_anchor: DepthAnchorSelection,
    intrinsics: dict[str, float],
    recovered_depth_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_z_override_m = None
    if recovered_depth_override is not None:
        target_z_override_m = recovered_depth_override.get("target_z_m")
    vertices, transform = fit_mesh_to_pixel_depth_anchor(
        mesh_vertices_camera,
        projected_pixels,
        depth_anchor.points,
        intrinsics,
        target_z_override_m=target_z_override_m,
    )
    transform["target_depth_points_source"] = depth_anchor.source
    transform["target_depth_points_count"] = int(len(depth_anchor.points))
    if recovered_depth_override is not None:
        transform["recovered_depth_override"] = recovered_depth_override
    return vertices, transform
