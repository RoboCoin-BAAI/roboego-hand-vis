from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from roboego_hand_vis.hand_depth_alignment.depth_mesh import opencv_to_rerun_display, rotate_points_axis_angle, translate_points


def sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx]


def display_vertices(points_camera: np.ndarray, args: Any) -> np.ndarray:
    points = opencv_to_rerun_display(points_camera)
    if args.display_mesh_rotate_axis != "none" and abs(args.display_mesh_rotate_degrees) > 1e-6:
        if args.display_mesh_rotate_about == "centroid" and len(points) > 0:
            origin = np.median(points, axis=0)
        else:
            origin = np.zeros(3, dtype=np.float32)
        points = rotate_points_axis_angle(points, args.display_mesh_rotate_axis, args.display_mesh_rotate_degrees, origin=origin)
    return translate_points(points, args.display_mesh_translate)


def draw_points_rgb(image: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    if len(pixels) == 0:
        return
    h, w = image.shape[:2]
    bgr = (int(color[2]), int(color[1]), int(color[0]))
    for u, v in np.round(pixels).astype(np.int32):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image, (int(u), int(v)), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)


def draw_mask_overlay_rgb(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.28,
) -> None:
    if mask.size == 0 or not mask.any():
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    overlay = image.copy()
    overlay[mask] = color
    image[:] = np.where(mask[:, :, None], image * (1.0 - alpha) + overlay * alpha, image).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness=2, lineType=cv2.LINE_AA)
