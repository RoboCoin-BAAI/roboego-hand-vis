from __future__ import annotations

from typing import Any, List

import numpy as np

from .trajectory_smoothing import smooth_centers_weighted, smooth_vertices_rigid


def smooth_final_mesh_track(
    vertices_by_frame: List[np.ndarray],
    statuses: List[str],
    *,
    window: int = 5,
) -> List[dict[str, Any]]:
    if len(vertices_by_frame) != len(statuses):
        raise ValueError("vertices_by_frame and statuses must have the same length")
    if not vertices_by_frame:
        return []
    centers = [np.median(vertices, axis=0) for vertices in vertices_by_frame]
    center_results = smooth_centers_weighted(centers, statuses, window=window)
    rigid_results = smooth_vertices_rigid(vertices_by_frame, statuses, window=window)
    out: List[dict[str, Any]] = []
    for center_result, rigid_result in zip(center_results, rigid_results):
        smoothed_center = np.asarray(center_result["center"], dtype=np.float32).reshape(1, 3)
        vertices = rigid_result["vertices_centered"] + smoothed_center
        out.append(
            {
                "vertices": vertices.astype(np.float32),
                "center_smoothing": center_result,
                "orientation_smoothing": {
                    key: value for key, value in rigid_result.items() if key != "vertices_centered"
                },
            }
        )
    return out
