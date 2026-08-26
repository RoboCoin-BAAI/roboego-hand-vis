"""Geometry helpers for a streaming camera/trajectory preview."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopDownProjection:
    center_xy: np.ndarray
    pixels_per_meter: float
    center_pixel: tuple[float, float]

    def project(self, positions_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(positions_xy, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError(
                f"Expected positions ending in 2 coordinates, got {points.shape}"
            )
        pixels = np.empty_like(points)
        pixels[..., 0] = (
            self.center_pixel[0]
            + (points[..., 0] - self.center_xy[0]) * self.pixels_per_meter
        )
        pixels[..., 1] = (
            self.center_pixel[1]
            - (points[..., 1] - self.center_xy[1]) * self.pixels_per_meter
        )
        return np.rint(pixels).astype(np.int32)


def fit_topdown_projection(
    positions_xy: np.ndarray,
    *,
    width: int,
    height: int,
    margin: int = 36,
) -> TopDownProjection:
    points = np.asarray(positions_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("positions_xy must have shape (N, 2) with at least one point")
    if not np.isfinite(points).all():
        raise ValueError("positions_xy contains non-finite values")
    if width <= 2 * margin or height <= 2 * margin:
        raise ValueError("Preview panel is too small for the requested margin")
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) * 0.5
    extent = np.maximum(maximum - minimum, 0.5)
    scale = min((width - 2 * margin) / extent[0], (height - 2 * margin) / extent[1])
    scale *= 0.92
    return TopDownProjection(
        center_xy=center,
        pixels_per_meter=float(scale),
        center_pixel=(width * 0.5, height * 0.5),
    )
