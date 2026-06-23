#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboegopipe.egohand.depth_mesh import (  # noqa: E402
    hamer_overlay_path_for_frame,
    hamer_rendered_obj_to_camera_opencv,
    hamer_renderer_obj_to_image,
    mask_from_overlay_difference,
    mask_from_projected_triangles,
    mask_overlap_stats,
    parse_obj_mesh,
    project_points_to_image,
)
from scripts.export_depth_pointcloud import ROPEDIA_K, read_rgb_frame  # noqa: E402


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
    out = [
        candidate
        for candidate in frame.get("candidates", [])
        if candidate.get("status")
        in {"ego_left", "ego_right", "ego_left_recovered", "ego_right_recovered"}
    ]
    out.sort(key=lambda c: c.get("status", "").replace("_recovered", ""))
    return out


def mask_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_mask_contours(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    label: str,
    label_xy: tuple[int, int],
) -> None:
    contours = mask_contours(mask)
    if contours:
        cv2.drawContours(image_rgb, contours, -1, color_rgb, thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        image_rgb,
        label,
        label_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color_rgb,
        1,
        cv2.LINE_AA,
    )


def draw_points(image_rgb: np.ndarray, pixels: np.ndarray, color_rgb: tuple[int, int, int], max_points: int) -> None:
    if len(pixels) == 0:
        return
    points = pixels
    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
    h, w = image_rgb.shape[:2]
    for u, v in np.round(points).astype(np.int32):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(image_rgb, (int(u), int(v)), 1, color_rgb, thickness=-1, lineType=cv2.LINE_AA)


def make_contact_sheet(image_paths: List[Path], output_path: Path, columns: int = 4, thumb_width: int = 360) -> None:
    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb = cv2.resize(
            image,
            (thumb_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / columns))
    thumb_height = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        row = i // columns
        col = i % columns
        y = row * thumb_height
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HaMeR official overlay mask with OBJ projections under different camera models."
    )
    parser.add_argument(
        "--candidates_json",
        default="tmp/ropedia_hamer_ego_filter_000000_000029_recovered/ego_hand_candidates_recovered.json",
    )
    parser.add_argument("--video", default="/home/liuyou/Documents/ropedia_data/stereo_left.mp4")
    parser.add_argument("--hamer_overlay_dir", default="")
    parser.add_argument("--output_dir", default="tmp/ropedia_hamer_renderer_projection_alignment")
    parser.add_argument("--hamer_base_focal_length", type=float, default=5000.0)
    parser.add_argument("--hamer_model_image_size", type=float, default=256.0)
    parser.add_argument("--overlay_diff_threshold", type=float, default=8.0)
    parser.add_argument("--overlay_diff_search_margin_px", type=int, default=24)
    parser.add_argument("--mesh_mask_padding_px", type=int, default=2)
    parser.add_argument("--max_projected_points", type=int, default=778)
    parser.add_argument("--contact_sheet_columns", type=int, default=4)
    parser.add_argument("--contact_sheet_thumb_width", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    original_cwd = Path.cwd()
    args = parse_args()
    candidates_json = resolve_path(args.candidates_json, original_cwd)
    video_path = resolve_path(args.video, original_cwd)
    output_dir = resolve_path(args.output_dir, original_cwd)
    overlay_dir = resolve_path(args.hamer_overlay_dir, original_cwd) if args.hamer_overlay_dir else None
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    with candidates_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    report: Dict[str, Any] = {
        "candidates_json": str(candidates_json),
        "video": str(video_path),
        "output_dir": str(output_dir),
        "legend": {
            "green": "HaMeR official overlay diff mask",
            "red": "OBJ projected with HaMeR renderer full-frame focal/camera center",
            "blue": "OBJ converted to camera_opencv and projected with Ropedia K",
        },
        "frames": [],
    }
    debug_paths: List[Path] = []
    all_renderer_iou: List[float] = []
    all_ropedia_iou: List[float] = []
    all_renderer_offset: List[float] = []
    all_ropedia_offset: List[float] = []

    for frame in data["frames"]:
        frame_idx = int(frame["frame_idx"])
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        height, width = rgb.shape[:2]
        hamer_focal = args.hamer_base_focal_length / args.hamer_model_image_size * max(width, height)

        overlay_path = hamer_overlay_path_for_frame(candidates_json, frame_idx, overlay_dir)
        overlay_rgb: Optional[np.ndarray] = None
        if overlay_path.exists():
            overlay_bgr = cv2.imread(str(overlay_path))
            if overlay_bgr is not None:
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        frame_report: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "hamer_overlay": str(overlay_path),
            "hamer_focal_length": float(hamer_focal),
            "hands": [],
        }
        debug = rgb.copy()

        for candidate in selected_candidates(frame):
            mesh_path = Path(candidate["mesh_obj"])
            if not mesh_path.exists() or overlay_rgb is None:
                frame_report["hands"].append(
                    {
                        "status": candidate.get("status"),
                        "candidate_id": candidate.get("candidate_id"),
                        "error": "missing mesh or overlay",
                    }
                )
                continue

            vertices_obj, faces, _ = parse_obj_mesh(mesh_path)
            overlay_mask = mask_from_overlay_difference(
                rgb,
                overlay_rgb,
                bbox_xyxy=candidate["bbox_xyxy"],
                diff_threshold=args.overlay_diff_threshold,
                search_margin_px=args.overlay_diff_search_margin_px,
                padding_px=args.mesh_mask_padding_px,
            )

            hamer_pixels_all = np.full((len(vertices_obj), 2), np.nan, dtype=np.float32)
            hamer_pixels, hamer_valid = hamer_renderer_obj_to_image(
                vertices_obj,
                focal_length=hamer_focal,
                width=width,
                height=height,
            )
            hamer_pixels_all[hamer_valid] = hamer_pixels
            hamer_mask = mask_from_projected_triangles(
                hamer_pixels_all,
                faces,
                width=width,
                height=height,
                padding_px=args.mesh_mask_padding_px,
            )

            vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
            ropedia_pixels_all = np.full((len(vertices_camera), 2), np.nan, dtype=np.float32)
            ropedia_pixels, ropedia_valid = project_points_to_image(vertices_camera, ROPEDIA_K, width=width, height=height)
            ropedia_pixels_all[ropedia_valid] = ropedia_pixels
            ropedia_mask = mask_from_projected_triangles(
                ropedia_pixels_all,
                faces,
                width=width,
                height=height,
                padding_px=args.mesh_mask_padding_px,
            )

            renderer_stats = mask_overlap_stats(overlay_mask, hamer_mask)
            ropedia_stats = mask_overlap_stats(overlay_mask, ropedia_mask)
            if renderer_stats["iou"] is not None:
                all_renderer_iou.append(float(renderer_stats["iou"]))
            if ropedia_stats["iou"] is not None:
                all_ropedia_iou.append(float(ropedia_stats["iou"]))
            if renderer_stats["centroid_offset_px"] is not None:
                all_renderer_offset.append(float(renderer_stats["centroid_offset_px"]))
            if ropedia_stats["centroid_offset_px"] is not None:
                all_ropedia_offset.append(float(ropedia_stats["centroid_offset_px"]))

            x1, y1, x2, y2 = [int(round(v)) for v in candidate["bbox_xyxy"]]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
            label_xy = (max(0, x1), max(14, y1 - 8))
            draw_mask_contours(debug, overlay_mask, (0, 255, 0), candidate["status"], label_xy)
            draw_mask_contours(debug, hamer_mask, (255, 0, 0), "hamerK", (label_xy[0], label_xy[1] + 14))
            draw_mask_contours(debug, ropedia_mask, (0, 120, 255), "ropediaK", (label_xy[0], label_xy[1] + 28))
            draw_points(debug, hamer_pixels, (255, 0, 0), args.max_projected_points)
            draw_points(debug, ropedia_pixels, (0, 120, 255), args.max_projected_points)

            frame_report["hands"].append(
                {
                    "status": candidate.get("status"),
                    "candidate_id": candidate.get("candidate_id"),
                    "handedness": candidate.get("handedness"),
                    "bbox_xyxy": candidate.get("bbox_xyxy"),
                    "pred_cam_t_full": candidate.get("pred_cam_t_full"),
                    "mesh_obj": str(mesh_path),
                    "hamer_renderer_projection": renderer_stats,
                    "ropedia_k_projection": ropedia_stats,
                    "hamer_renderer_visible_vertices": int(len(hamer_pixels)),
                    "ropedia_k_visible_vertices": int(len(ropedia_pixels)),
                }
            )

        debug_path = debug_dir / f"frame_{frame_idx:06d}_hamer_projection_alignment.jpg"
        cv2.imwrite(str(debug_path), cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
        debug_paths.append(debug_path)
        frame_report["debug_image"] = str(debug_path)
        report["frames"].append(frame_report)

    def average(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    report["summary"] = {
        "hamer_renderer_iou_avg": average(all_renderer_iou),
        "ropedia_k_iou_avg": average(all_ropedia_iou),
        "hamer_renderer_centroid_offset_px_avg": average(all_renderer_offset),
        "ropedia_k_centroid_offset_px_avg": average(all_ropedia_offset),
        "hand_count": int(len(all_renderer_iou)),
    }

    report_path = output_dir / "hamer_renderer_projection_alignment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)

    contact_sheet_path = output_dir / "hamer_renderer_projection_alignment_contact_sheet.jpg"
    make_contact_sheet(
        debug_paths,
        contact_sheet_path,
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )

    print(f"Report: {report_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
