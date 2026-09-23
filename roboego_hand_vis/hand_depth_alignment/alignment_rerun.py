from __future__ import annotations

import numpy as np
import rerun as rr

from roboego_hand_vis.hand_depth_alignment.depth_mesh import project_points_to_image
from roboego_hand_vis.hand_depth_alignment.alignment_geometry import draw_points_rgb


LAYER_COLORS_RGBA = {
    "normal": [255, 180, 60, 185],
    "recovered": [255, 80, 220, 190],
    "smooth": [80, 180, 255, 195],
    "current": [60, 255, 120, 210],
}

LAYER_COLORS_RGB = {
    "normal": (255, 180, 60),
    "recovered": (255, 80, 220),
    "smooth": (80, 180, 255),
    "current": (60, 255, 120),
}


def log_mesh_layer(
    *,
    side: str,
    layer: str,
    vertices_camera: np.ndarray,
    vertices_display: np.ndarray,
    faces: np.ndarray,
    rgb_layer_images: dict[str, np.ndarray],
    width: int,
    height: int,
    point_radius: float,
    intrinsics: dict,
) -> None:
    color_rgba = LAYER_COLORS_RGBA[layer]
    color_rgb = LAYER_COLORS_RGB[layer]
    rr.log(
        f"camera_display/hands/{side}/layers/{layer}_mesh",
        rr.Mesh3D(vertex_positions=vertices_display, triangle_indices=faces, albedo_factor=color_rgba),
    )
    rr.log(
        f"camera_opencv/hands/{side}/layers/{layer}_mesh",
        rr.Mesh3D(vertex_positions=vertices_camera, triangle_indices=faces, albedo_factor=color_rgba),
    )
    projected_pixels, _ = project_points_to_image(vertices_camera, intrinsics, width, height)
    image = rgb_layer_images[layer]
    draw_points_rgb(image, projected_pixels, color_rgb, radius=max(1, int(round(point_radius * 0.6))))


def clear_hand_entities(status: str) -> None:
    rr.log(f"camera_display/hands/{status}", rr.Clear(recursive=True))
    rr.log(f"camera_opencv/hands/{status}", rr.Clear(recursive=True))
    rr.log(f"camera_display/camera/left/rgb/overlay/{status}", rr.Clear(recursive=True))
