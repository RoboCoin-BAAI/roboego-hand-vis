#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.refined_temporal import build_temporal_tracks_from_refined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal audit and short-gap smoothing for strict refined HaMeR hands.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--refined_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="refined_temporal_gate_tracks.json")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--max_gap", type=int, default=10)
    parser.add_argument("--max_bbox_jump_px", type=float, default=220.0)
    parser.add_argument("--max_bbox_area_change_ratio", type=float, default=3.0)
    parser.add_argument("--max_cam_t_jump", type=float, default=0.0, help="Disable camera-translation jump gate when <= 0.")
    parser.add_argument("--min_cam_t_z", type=float, default=0.0)
    parser.add_argument("--max_cam_t_z", type=float, default=200.0)
    parser.add_argument("--max_mano_pose_jump", type=float, default=0.0, help="Disable MANO pose jump gate when <= 0.")
    parser.add_argument("--max_mesh_centroid_jump", type=float, default=0.0, help="Disable mesh centroid jump gate when <= 0.")
    parser.add_argument("--max_mesh_vertex_rmse", type=float, default=0.0, help="Disable mesh vertex RMSE gate when <= 0.")
    parser.add_argument("--local_window_radius", type=int, default=2)
    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact_sheet_columns", type=int, default=4)
    parser.add_argument("--contact_sheet_thumb_width", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_json = Path(args.refined_json).expanduser()
    manifest = Path(args.manifest).expanduser()
    refined_payload = json.loads(refined_json.read_text(encoding="utf-8"))
    result = build_temporal_tracks_from_refined(
        refined_payload,
        start=args.start,
        num_frames=args.num_frames,
        max_gap=args.max_gap,
        max_bbox_jump_px=args.max_bbox_jump_px,
        max_bbox_area_change_ratio=args.max_bbox_area_change_ratio,
        max_cam_t_jump=args.max_cam_t_jump,
        min_cam_t_z=args.min_cam_t_z,
        max_cam_t_z=args.max_cam_t_z,
        max_mano_pose_jump=args.max_mano_pose_jump,
        max_mesh_centroid_jump=args.max_mesh_centroid_jump,
        max_mesh_vertex_rmse=args.max_mesh_vertex_rmse,
        local_window_radius=args.local_window_radius,
    )
    result["source_refined_json"] = str(refined_json)
    result["source_manifest"] = str(manifest)
    result_path = output_dir / args.output_name
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_vis:
        _write_temporal_visualization(
            manifest=manifest,
            tracks_payload=result,
            output_dir=output_dir / "temporal_annotated",
            contact_sheet_path=output_dir / "temporal_contact_sheet.jpg",
            columns=args.contact_sheet_columns,
            thumb_width=args.contact_sheet_thumb_width,
        )
    print(json.dumps({"tracks_json": str(result_path), "summary": result["summary"]}, indent=2, ensure_ascii=False))


def _write_temporal_visualization(
    *,
    manifest: Path,
    tracks_payload: dict,
    output_dir: Path,
    contact_sheet_path: Path,
    columns: int,
    thumb_width: int,
) -> None:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    video_path = Path(manifest_payload["videos"]["left"]["path"])
    if not video_path.is_absolute():
        video_path = manifest.parent / video_path
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("frame_*_temporal.jpg", "frame_*_temporal.png"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    if contact_sheet_path.exists():
        contact_sheet_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    sheet_images: list[np.ndarray] = []
    try:
        for frame in tracks_payload.get("frames", []):
            frame_idx = int(frame["frame_idx"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                continue
            annotated = _draw_temporal_frame(image, frame)
            out_path = output_dir / f"frame_{frame_idx:06d}_temporal.jpg"
            cv2.imwrite(str(out_path), annotated)
            if len(sheet_images) < 24:
                sheet_images.append(_resize_for_sheet(annotated, width=thumb_width))
    finally:
        cap.release()
    if sheet_images:
        cv2.imwrite(str(contact_sheet_path), _make_contact_sheet(sheet_images, cols=columns))


def _draw_temporal_frame(image: np.ndarray, frame: dict) -> np.ndarray:
    out = image.copy()
    for side, obs in (frame.get("hands") or {}).items():
        if not isinstance(obs, dict):
            continue
        bbox = obs.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
        color = _status_color(obs)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 4)
        flags = ",".join(obs.get("anomaly_flags", [])[:2])
        label = f"{side} {obs.get('status')} {obs.get('candidate_id')} {flags}".strip()
        cv2.putText(out, label, (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"refined temporal: {frame.get('visibility_state', '')}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    return out


def _status_color(obs: dict) -> tuple[int, int, int]:
    flags = set(obs.get("anomaly_flags", []))
    if "large_bbox_center_jump" in flags:
        return (0, 220, 255)
    if obs.get("status") == "interpolated":
        return (255, 180, 0)
    return (40, 255, 40)


def _resize_for_sheet(image: np.ndarray, *, width: int) -> np.ndarray:
    scale = width / max(1, image.shape[1])
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def _make_contact_sheet(images: list[np.ndarray], *, cols: int) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = int(np.ceil(len(images) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        sheet[row * h : row * h + img.shape[0], col * w : col * w + img.shape[1]] = img
    return sheet


if __name__ == "__main__":
    main()
