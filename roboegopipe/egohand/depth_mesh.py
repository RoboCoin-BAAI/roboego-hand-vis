from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


def hamer_overlay_path_for_frame(
    candidates_json: Path,
    frame_idx: int,
    hamer_overlay_dir: Optional[Path] = None,
) -> Path:
    base_dir = Path(hamer_overlay_dir) if hamer_overlay_dir is not None else Path(candidates_json).parent / "hamer_overlay"
    return base_dir / f"frame_{frame_idx:06d}_hamer_overlay.jpg"


def opencv_to_rerun_display(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    x_camera = points[:, 0]
    y_camera = points[:, 1]
    z_camera = points[:, 2]
    return np.stack([z_camera, -x_camera, -y_camera], axis=1).astype(np.float32)


def hamer_rendered_obj_to_camera_opencv(vertices: np.ndarray) -> np.ndarray:
    """Undo HaMeR renderer's OBJ export flip to recover OpenCV camera coordinates.

    HaMeR's demo exports OBJ meshes after calling ``vertices_to_trimesh``. That
    helper first stores ``pred_vertices + pred_cam_t_full`` and then applies a
    180 degree rotation around X for pyrender display, so OBJ vertices are
    ``[x, -y, -z]`` relative to HaMeR full-frame/OpenCV camera coordinates.
    """
    if len(vertices) == 0:
        return vertices.astype(np.float32)
    verts = vertices.astype(np.float32)
    return np.stack([verts[:, 0], -verts[:, 1], -verts[:, 2]], axis=1).astype(np.float32)


def rotate_points_axis_angle(
    points: np.ndarray,
    axis: str,
    degrees: float,
    center: Optional[Iterable[float]] = None,
) -> np.ndarray:
    if len(points) == 0 or axis == "none" or abs(degrees) < 1e-9:
        return points.astype(np.float32)
    axis_vectors = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    if axis not in axis_vectors:
        raise ValueError(f"Unsupported rotation axis: {axis}")

    theta = np.deg2rad(float(degrees))
    unit_axis = axis_vectors[axis]
    cross = np.array(
        [
            [0.0, -unit_axis[2], unit_axis[1]],
            [unit_axis[2], 0.0, -unit_axis[0]],
            [-unit_axis[1], unit_axis[0], 0.0],
        ],
        dtype=np.float32,
    )
    rotation = (
        np.eye(3, dtype=np.float32) * np.cos(theta)
        + (1.0 - np.cos(theta)) * np.outer(unit_axis, unit_axis).astype(np.float32)
        + np.sin(theta) * cross
    )
    center_arr = (
        np.asarray(center, dtype=np.float32).reshape(1, 3)
        if center is not None
        else np.zeros((1, 3), dtype=np.float32)
    )
    centered = points.astype(np.float32) - center_arr
    return (centered @ rotation.T + center_arr).astype(np.float32)


def translate_points(points: np.ndarray, xyz: Iterable[float]) -> np.ndarray:
    offset = np.asarray(list(xyz), dtype=np.float32).reshape(1, 3)
    return (points.astype(np.float32) + offset).astype(np.float32)


def smooth_translation(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    current_arr = np.asarray(current, dtype=np.float32).reshape(3)
    if previous is None:
        return current_arr
    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    previous_arr = np.asarray(previous, dtype=np.float32).reshape(3)
    return (previous_arr * (1.0 - alpha_clamped) + current_arr * alpha_clamped).astype(np.float32)


def project_points_to_image(
    points: np.ndarray,
    intrinsics: Dict[str, float],
    width: int,
    height: int,
    min_z: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    z = points[:, 2].astype(np.float32)
    valid_z = np.isfinite(z) & (z > min_z)
    u = np.empty_like(z, dtype=np.float32)
    v = np.empty_like(z, dtype=np.float32)
    u[:] = np.nan
    v[:] = np.nan
    u[valid_z] = float(intrinsics["fx"]) * points[valid_z, 0] / z[valid_z] + float(intrinsics["cx"])
    v[valid_z] = float(intrinsics["fy"]) * points[valid_z, 1] / z[valid_z] + float(intrinsics["cy"])
    in_image = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixels = np.stack([u[in_image], v[in_image]], axis=1).astype(np.float32)
    return pixels, in_image


def hamer_renderer_obj_to_image(
    vertices: np.ndarray,
    focal_length: float,
    width: int,
    height: int,
    min_depth: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    verts = vertices.astype(np.float32)
    z = -verts[:, 2]
    valid_z = np.isfinite(z) & (z > min_depth)
    u = np.empty_like(z, dtype=np.float32)
    v = np.empty_like(z, dtype=np.float32)
    u[:] = np.nan
    v[:] = np.nan
    u[valid_z] = float(focal_length) * verts[valid_z, 0] / z[valid_z] + width / 2.0
    v[valid_z] = -float(focal_length) * verts[valid_z, 1] / z[valid_z] + height / 2.0
    in_image = valid_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixels = np.stack([u[in_image], v[in_image]], axis=1).astype(np.float32)
    return pixels, in_image


def lift_pixels_to_depth_points(
    pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.5,
    window_radius_px: int = 2,
    fallback_depth_m: Optional[float] = None,
    reference_depth_m: Optional[float] = None,
    max_depth_delta_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(pixels) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=bool),
            {"input_count": 0, "depth_sample_count": 0, "fallback_count": 0},
        )

    h, w = depth_m.shape[:2]
    radius = max(0, int(window_radius_px))
    points = np.empty((len(pixels), 3), dtype=np.float32)
    valid = np.zeros((len(pixels),), dtype=bool)
    sampled_depths: list[float] = []

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    fallback = None if fallback_depth_m is None else float(fallback_depth_m)
    fallback_count = 0
    outlier_count = 0
    reference_depth = None if reference_depth_m is None else float(reference_depth_m)
    max_delta = None if max_depth_delta_m is None or max_depth_delta_m <= 0 else float(max_depth_delta_m)
    for i, (u_float, v_float) in enumerate(pixels.astype(np.float32)):
        u = int(round(float(u_float)))
        v = int(round(float(v_float)))
        z_value: Optional[float] = None
        if 0 <= u < w and 0 <= v < h:
            x1 = max(0, u - radius)
            x2 = min(w, u + radius + 1)
            y1 = max(0, v - radius)
            y2 = min(h, v + radius + 1)
            window = depth_m[y1:y2, x1:x2].astype(np.float32).reshape(-1)
            good = window[np.isfinite(window) & (window >= min_depth_m) & (window <= max_depth_m)]
            if len(good) > 0:
                z_value = float(np.median(good))
                if (
                    reference_depth is not None
                    and np.isfinite(reference_depth)
                    and max_delta is not None
                    and abs(z_value - reference_depth) > max_delta
                ):
                    z_value = None
                    outlier_count += 1
                else:
                    valid[i] = True
                    sampled_depths.append(z_value)

        if z_value is None:
            if fallback is None or not np.isfinite(fallback):
                z_value = 0.0
            else:
                z_value = fallback
            fallback_count += 1

        x = (float(u_float) - cx) * z_value / fx
        y = (float(v_float) - cy) * z_value / fy
        points[i] = [x, y, z_value]

    stats = {
        "input_count": int(len(pixels)),
        "depth_sample_count": int(valid.sum()),
        "fallback_count": int(fallback_count),
        "outlier_count": int(outlier_count),
        "valid_depth_ratio": float(valid.sum() / max(1, len(pixels))),
        "fallback_depth_m": fallback,
        "reference_depth_m": reference_depth,
        "max_depth_delta_m": max_delta,
        "sampled_depth_median": float(np.median(sampled_depths)) if sampled_depths else None,
        "window_radius_px": radius,
    }
    return points, valid, stats


def _clip_bbox(
    bbox_xyxy: Iterable[float],
    width: int,
    height: int,
    margin_px: int = 0,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = int(np.floor(x1 - margin_px))
    y1 = int(np.floor(y1 - margin_px))
    x2 = int(np.ceil(x2 + margin_px))
    y2 = int(np.ceil(y2 + margin_px))
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def crop_depth_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    bbox_xyxy: Iterable[float],
    intrinsics: Dict[str, float],
    stride: int = 1,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.5,
    margin_px: int = 12,
    max_points: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, w, h, margin_px)
    if x2 <= x1 or y2 <= y1:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        return empty_points, empty_colors, {"roi_pixel_count": 0, "valid_depth_ratio": 0.0}

    stride = max(1, int(stride))
    ys, xs = np.mgrid[y1:y2:stride, x1:x2:stride]
    z = depth_m[ys, xs].astype(np.float32)
    colors = rgb[ys, xs].reshape(-1, 3)
    z_flat = z.reshape(-1)
    xs_flat = xs.astype(np.float32).reshape(-1)
    ys_flat = ys.astype(np.float32).reshape(-1)

    valid = np.isfinite(z_flat) & (z_flat >= min_depth_m) & (z_flat <= max_depth_m)
    valid_count = int(valid.sum())
    roi_pixel_count = int(z_flat.size)
    if valid_count == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_colors = np.empty((0, 3), dtype=np.uint8)
        return (
            empty_points,
            empty_colors,
            {"roi_pixel_count": roi_pixel_count, "valid_depth_ratio": 0.0},
        )

    xs_valid = xs_flat[valid]
    ys_valid = ys_flat[valid]
    z_valid = z_flat[valid]
    colors_valid = colors[valid]

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (xs_valid - cx) * z_valid / fx
    y = (ys_valid - cy) * z_valid / fy
    points = np.stack([x, y, z_valid], axis=1).astype(np.float32)

    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
        colors_valid = colors_valid[idx]

    stats = {
        "roi_pixel_count": roi_pixel_count,
        "valid_depth_count": valid_count,
        "valid_depth_ratio": valid_count / max(1, roi_pixel_count),
        "median_depth_m": float(np.median(z_valid)),
        "p10_depth_m": float(np.percentile(z_valid, 10)),
        "p90_depth_m": float(np.percentile(z_valid, 90)),
    }
    return points, colors_valid.astype(np.uint8), stats


def mask_from_projected_points(
    pixels: np.ndarray,
    width: int,
    height: int,
    padding_px: int = 8,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if len(pixels) < 3:
        return mask

    valid = np.isfinite(pixels).all(axis=1)
    pts = np.round(pixels[valid]).astype(np.int32)
    if len(pts) < 3:
        return mask
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask_u8, hull, 255)
    if padding_px > 0:
        kernel_size = int(padding_px) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return mask_u8.astype(bool)


def mask_from_projected_triangles(
    pixels: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    padding_px: int = 4,
) -> np.ndarray:
    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    if len(pixels) == 0 or len(faces) == 0:
        return mask_u8.astype(bool)

    px = pixels.astype(np.float32)
    finite_vertices = np.isfinite(px).all(axis=1)
    for face in faces.astype(np.int64):
        if len(face) != 3:
            continue
        if np.any(face < 0) or np.any(face >= len(px)):
            continue
        if not finite_vertices[face].all():
            continue
        tri = np.round(px[face]).astype(np.int32)
        if (
            tri[:, 0].max() < 0
            or tri[:, 0].min() >= width
            or tri[:, 1].max() < 0
            or tri[:, 1].min() >= height
        ):
            continue
        tri[:, 0] = np.clip(tri[:, 0], 0, width - 1)
        tri[:, 1] = np.clip(tri[:, 1], 0, height - 1)
        cv2.fillConvexPoly(mask_u8, tri, 255)

    if padding_px > 0 and mask_u8.any():
        kernel_size = int(padding_px) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return mask_u8.astype(bool)


def mask_from_overlay_difference(
    rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    bbox_xyxy: Iterable[float],
    diff_threshold: float = 8.0,
    search_margin_px: int = 24,
    padding_px: int = 2,
) -> np.ndarray:
    if rgb.shape[:2] != overlay_rgb.shape[:2]:
        overlay_rgb = cv2.resize(overlay_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.mean(
        np.abs(overlay_rgb.astype(np.int16) - rgb.astype(np.int16)),
        axis=2,
    )
    diff_mask = diff > float(diff_threshold)
    height, width = diff_mask.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, width, height, margin_px=search_margin_px)
    search = np.zeros_like(diff_mask, dtype=np.uint8)
    search[y1:y2, x1:x2] = diff_mask[y1:y2, x1:x2].astype(np.uint8)
    if search.sum() == 0:
        return search.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(search, connectivity=8)
    if num_labels <= 1:
        return search.astype(bool)

    bbox_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    best_label = None
    best_score = -np.inf
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        lx = stats[label, cv2.CC_STAT_LEFT]
        ly = stats[label, cv2.CC_STAT_TOP]
        lw = stats[label, cv2.CC_STAT_WIDTH]
        lh = stats[label, cv2.CC_STAT_HEIGHT]
        component_center = np.array([lx + lw * 0.5, ly + lh * 0.5], dtype=np.float32)
        distance = float(np.linalg.norm(component_center - bbox_center))
        score = area - 0.25 * distance
        if score > best_score:
            best_label = label
            best_score = score

    out = (labels == best_label).astype(np.uint8) if best_label is not None else search
    if padding_px > 0 and out.any():
        kernel_size = int(padding_px) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)
    return out.astype(bool)


def core_mask_from_mask(
    mask: np.ndarray,
    keep_x_ratio: float = 0.6,
    keep_y_ratio: float = 0.6,
    min_pixels: int = 20,
) -> np.ndarray:
    """Keep the central part of a 2D mask for a more stable depth anchor.

    The full rendered-hand mask is useful for visualization, but its edges can
    include object, sleeve, or forearm pixels. For depth z anchoring we prefer a
    conservative core and fall back to the original mask if the core becomes too
    small.
    """
    mask_bool = mask.astype(bool)
    ys, xs = np.where(mask_bool)
    if len(xs) < int(min_pixels):
        return mask_bool

    keep_x = float(np.clip(keep_x_ratio, 0.01, 1.0))
    keep_y = float(np.clip(keep_y_ratio, 0.01, 1.0))
    x_margin_percent = (1.0 - keep_x) * 50.0
    y_margin_percent = (1.0 - keep_y) * 50.0
    x_lo, x_hi = np.percentile(xs, [x_margin_percent, 100.0 - x_margin_percent])
    y_lo, y_hi = np.percentile(ys, [y_margin_percent, 100.0 - y_margin_percent])

    yy, xx = np.indices(mask_bool.shape)
    core = mask_bool & (xx >= x_lo) & (xx <= x_hi) & (yy >= y_lo) & (yy <= y_hi)
    if int(core.sum()) < int(min_pixels):
        return mask_bool
    return core.astype(bool)


def mask_upper_region(
    mask: np.ndarray,
    keep_top_ratio: float = 0.65,
    min_pixels: int = 20,
) -> np.ndarray:
    """Keep the upper/finger-side portion of a mask in image coordinates.

    For the current ego-view Ropedia clips, the wrist/forearm tends to sit near
    the lower side of the hand mask. This region is therefore useful as a
    conservative finger/palm anchor while avoiding wrist-heavy depth points.
    """
    mask_bool = mask.astype(bool)
    ys, _ = np.where(mask_bool)
    if len(ys) < int(min_pixels):
        return mask_bool

    keep = float(np.clip(keep_top_ratio, 0.01, 1.0))
    cutoff_y = np.percentile(ys, keep * 100.0)
    yy, _ = np.indices(mask_bool.shape)
    upper = mask_bool & (yy <= cutoff_y)
    if int(upper.sum()) < int(min_pixels):
        return mask_bool
    return upper.astype(bool)


def mask_overlap_stats(mask_a: np.ndarray, mask_b: np.ndarray) -> Dict[str, Any]:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    intersection = a & b
    union = a | b

    def centroid_xy(mask: np.ndarray) -> Optional[list[float]]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [float(np.mean(xs)), float(np.mean(ys))]

    ca = centroid_xy(a)
    cb = centroid_xy(b)
    if ca is None or cb is None:
        centroid_offset = None
    else:
        centroid_offset = float(np.linalg.norm(np.asarray(ca, dtype=np.float32) - np.asarray(cb, dtype=np.float32)))

    union_count = int(union.sum())
    intersection_count = int(intersection.sum())
    return {
        "mask_a_count": int(a.sum()),
        "mask_b_count": int(b.sum()),
        "intersection_count": intersection_count,
        "union_count": union_count,
        "iou": float(intersection_count / union_count) if union_count else None,
        "centroid_a_xy": ca,
        "centroid_b_xy": cb,
        "centroid_offset_px": centroid_offset,
    }


def depth_mask_to_pointcloud(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    intrinsics: Dict[str, float],
    stride: int = 1,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.5,
    max_points: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if rgb.shape[:2] != depth_m.shape:
        rgb = cv2.resize(rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)
    if mask.shape[:2] != depth_m.shape:
        mask = cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)

    stride = max(1, int(stride))
    h, w = depth_m.shape[:2]
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    mask_sample = mask[ys, xs].reshape(-1)
    z_flat = depth_m[ys, xs].astype(np.float32).reshape(-1)
    xs_flat = xs.astype(np.float32).reshape(-1)
    ys_flat = ys.astype(np.float32).reshape(-1)
    colors = rgb[ys, xs].reshape(-1, 3)

    valid = mask_sample & np.isfinite(z_flat) & (z_flat >= min_depth_m) & (z_flat <= max_depth_m)
    mask_pixel_count = int(mask_sample.sum())
    valid_count = int(valid.sum())
    if valid_count == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"mask_pixel_count": mask_pixel_count, "valid_depth_count": 0, "valid_depth_ratio": 0.0},
        )

    xs_valid = xs_flat[valid]
    ys_valid = ys_flat[valid]
    z_valid = z_flat[valid]
    colors_valid = colors[valid]

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (xs_valid - cx) * z_valid / fx
    y = (ys_valid - cy) * z_valid / fy
    points = np.stack([x, y, z_valid], axis=1).astype(np.float32)

    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
        colors_valid = colors_valid[idx]

    stats = {
        "mask_pixel_count": mask_pixel_count,
        "valid_depth_count": valid_count,
        "valid_depth_ratio": valid_count / max(1, mask_pixel_count),
        "median_depth_m": float(np.median(z_valid)),
        "p10_depth_m": float(np.percentile(z_valid, 10)),
        "p90_depth_m": float(np.percentile(z_valid, 90)),
    }
    return points, colors_valid.astype(np.uint8), stats


def parse_obj_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    vertices = []
    colors = []
    faces = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    color = [float(parts[4]), float(parts[5]), float(parts[6])]
                    colors.append([int(np.clip(round(c * 255.0), 0, 255)) for c in color])
            elif line.startswith("f "):
                face = []
                for token in line.split()[1:4]:
                    face.append(int(token.split("/")[0]) - 1)
                faces.append(face)

    vertex_arr = np.asarray(vertices, dtype=np.float32)
    face_arr = np.asarray(faces, dtype=np.int32)
    color_arr = np.asarray(colors, dtype=np.uint8) if len(colors) == len(vertices) else None
    return vertex_arr, face_arr, color_arr


def robust_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    lo = np.percentile(points, 5, axis=0)
    hi = np.percentile(points, 95, axis=0)
    return float(np.linalg.norm(hi - lo))


def fit_mesh_to_points_similarity(
    mesh_vertices: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(mesh_vertices) == 0 or len(target_points) == 0:
        return mesh_vertices.astype(np.float32), {
            "scale": 1.0,
            "translation": [0.0, 0.0, 0.0],
            "source_centroid": [0.0, 0.0, 0.0],
            "target_centroid": [0.0, 0.0, 0.0],
        }

    source_centroid = np.median(mesh_vertices, axis=0)
    target_centroid = np.median(target_points, axis=0)
    source_centered = mesh_vertices - source_centroid.reshape(1, 3)
    source_extent = robust_extent(source_centered)
    target_extent = robust_extent(target_points - target_centroid.reshape(1, 3))
    scale = target_extent / source_extent if source_extent > 1e-9 else 1.0
    fitted = source_centered * scale + target_centroid.reshape(1, 3)
    translation = target_centroid - source_centroid * scale
    transform = {
        "scale": float(scale),
        "translation": translation.astype(float).tolist(),
        "source_centroid": source_centroid.astype(float).tolist(),
        "target_centroid": target_centroid.astype(float).tolist(),
        "source_extent": float(source_extent),
        "target_extent": float(target_extent),
    }
    return fitted.astype(np.float32), transform


def fit_mesh_to_points_translation(
    mesh_vertices: np.ndarray,
    target_points: np.ndarray,
    scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(mesh_vertices) == 0 or len(target_points) == 0:
        return mesh_vertices.astype(np.float32), {
            "scale": float(scale),
            "translation": [0.0, 0.0, 0.0],
            "source_centroid": [0.0, 0.0, 0.0],
            "target_centroid": [0.0, 0.0, 0.0],
        }

    source_centroid = np.median(mesh_vertices, axis=0)
    target_centroid = np.median(target_points, axis=0)
    fitted = (mesh_vertices - source_centroid.reshape(1, 3)) * scale
    fitted = fitted + target_centroid.reshape(1, 3)
    translation = target_centroid - source_centroid * scale
    transform = {
        "scale": float(scale),
        "translation": translation.astype(float).tolist(),
        "source_centroid": source_centroid.astype(float).tolist(),
        "target_centroid": target_centroid.astype(float).tolist(),
        "source_extent": robust_extent(mesh_vertices - source_centroid.reshape(1, 3)),
        "target_extent": robust_extent(target_points - target_centroid.reshape(1, 3)),
    }
    return fitted.astype(np.float32), transform


def fit_mesh_z_to_points(
    mesh_vertices: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(mesh_vertices) == 0 or len(target_points) == 0:
        return mesh_vertices.astype(np.float32), {
            "translation": [0.0, 0.0, 0.0],
            "source_z_median": None,
            "target_z_median": None,
        }

    source_z = float(np.median(mesh_vertices[:, 2]))
    target_z = float(np.median(target_points[:, 2]))
    dz = target_z - source_z
    fitted = mesh_vertices.astype(np.float32).copy()
    fitted[:, 2] += np.float32(dz)
    return fitted, {
        "translation": [0.0, 0.0, float(dz)],
        "source_z_median": source_z,
        "target_z_median": target_z,
    }


def fit_mesh_to_pixel_depth_anchor(
    mesh_vertices: np.ndarray,
    projected_pixels: np.ndarray,
    target_points: np.ndarray,
    intrinsics: Dict[str, float],
    target_z_override_m: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(mesh_vertices) == 0 or len(projected_pixels) == 0 or len(target_points) == 0:
        return mesh_vertices.astype(np.float32), {
            "translation": [0.0, 0.0, 0.0],
            "source_anchor": None,
            "target_anchor": None,
            "target_pixel_xy": None,
            "target_z_median": None,
            "valid_projected_vertex_count": 0,
        }

    pixels = np.asarray(projected_pixels, dtype=np.float32)
    valid_count = min(len(mesh_vertices), len(pixels))
    pixels_for_fit = pixels[:valid_count]
    mesh_for_fit = mesh_vertices[:valid_count]
    valid_pixels = np.isfinite(pixels_for_fit).all(axis=1) & np.isfinite(mesh_for_fit).all(axis=1)
    if not valid_pixels.any():
        return mesh_vertices.astype(np.float32), {
            "translation": [0.0, 0.0, 0.0],
            "source_anchor": None,
            "target_anchor": None,
            "target_pixel_xy": None,
            "target_z_median": None,
            "valid_projected_vertex_count": 0,
        }

    observed_target_z = float(np.median(target_points[:, 2]))
    if target_z_override_m is None:
        target_z = observed_target_z
        target_z_source = "observed"
    else:
        target_z = float(target_z_override_m)
        target_z_source = "override"
    target_uv = np.median(pixels_for_fit[valid_pixels], axis=0)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    target_anchor = np.array(
        [
            (float(target_uv[0]) - cx) * target_z / fx,
            (float(target_uv[1]) - cy) * target_z / fy,
            target_z,
        ],
        dtype=np.float32,
    )
    source_anchor = np.median(mesh_for_fit[valid_pixels], axis=0).astype(np.float32)
    translation = target_anchor - source_anchor
    fitted = translate_points(mesh_vertices, translation)
    return fitted.astype(np.float32), {
        "translation": translation.astype(float).tolist(),
        "source_anchor": source_anchor.astype(float).tolist(),
        "target_anchor": target_anchor.astype(float).tolist(),
        "target_pixel_xy": target_uv.astype(float).tolist(),
        "target_z_median": target_z,
        "target_z_observed_median": observed_target_z,
        "target_z_source": target_z_source,
        "valid_projected_vertex_count": int(valid_pixels.sum()),
    }


def filter_points_by_depth_band(
    points: np.ndarray,
    colors: np.ndarray,
    center_depth_m: Optional[float],
    half_width_m: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(points) == 0 or center_depth_m is None or half_width_m <= 0:
        return points, colors, {
            "input_count": int(len(points)),
            "kept_count": int(len(points)),
            "kept_ratio": 1.0 if len(points) else 0.0,
            "center_depth_m": center_depth_m,
            "half_width_m": float(half_width_m),
        }

    z = points[:, 2]
    keep = np.isfinite(z) & (z >= center_depth_m - half_width_m) & (z <= center_depth_m + half_width_m)
    filtered_points = points[keep]
    filtered_colors = colors[keep]
    return filtered_points, filtered_colors, {
        "input_count": int(len(points)),
        "kept_count": int(len(filtered_points)),
        "kept_ratio": int(len(filtered_points)) / max(1, int(len(points))),
        "center_depth_m": float(center_depth_m),
        "half_width_m": float(half_width_m),
    }


def nearest_neighbor_error(source_points: np.ndarray, target_points: np.ndarray) -> Dict[str, Any]:
    if len(source_points) == 0 or len(target_points) == 0:
        return {
            "count": 0,
            "median_m": None,
            "mean_m": None,
            "p90_m": None,
            "max_m": None,
        }
    tree = cKDTree(target_points.astype(np.float32))
    distances, _ = tree.query(source_points.astype(np.float32), k=1)
    return {
        "count": int(len(distances)),
        "median_m": float(np.median(distances)),
        "mean_m": float(np.mean(distances)),
        "p90_m": float(np.percentile(distances, 90)),
        "max_m": float(np.max(distances)),
    }
