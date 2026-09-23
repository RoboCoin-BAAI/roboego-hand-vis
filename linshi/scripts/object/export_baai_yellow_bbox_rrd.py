#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rerun as rr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.hand_depth_alignment.orientation import load_mano_regressors  # noqa: E402
from roboego_hand_vis.hand_rerun_visualization.track_rerun import build_final_smoothed_meshes, log_rerun  # noqa: E402
from linshi.scripts.export_depth_pointcloud import read_rgb_frame  # noqa: E402


ROOT = Path(".")
TRACK_DIR = ROOT / "tmp/baai_ego_hand_tracking_900_identity_gate"
TRACKS_JSON = TRACK_DIR / "ego_hand_tracks.json"
ALIGNMENT_REPORT = ROOT / "tmp/baai_hand_depth_alignment_900/hand_depth_mesh_alignment_report.json"
OUT_RRD = TRACK_DIR / "baai_ego_hands_with_yellow_object_bbox.rrd"
OUT_BOXES = TRACK_DIR / "baai_yellow_object_bboxes.json"
OVERLAY_ENTITY = "camera_opencv/camera/left/rgb_object_bbox_overlay"


def detect_yellow_bboxes(video_path: Path, num_frames: int) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    raw: list[dict | None] = []
    last_bbox: list[int] | None = None
    for frame_idx in range(num_frames):
        ok, bgr = cap.read()
        if not ok:
            raw.append(None)
            continue
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(h * 0.38):h, int(w * 0.20):int(w * 0.80)] = 255
        mask = cv2.inRange(hsv, np.array([16, 55, 70]), np.array([45, 255, 255]))
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for label in range(1, num):
            x, y, bw, bh, area = [int(v) for v in stats[label]]
            cx, cy = [float(v) for v in centroids[label]]
            if area < 900 or cy < h * 0.42:
                continue
            if last_bbox is not None:
                lx, ly, lw, lh = last_bbox
                lcx, lcy = lx + lw * 0.5, ly + lh * 0.5
                jump = float(np.hypot(cx - lcx, cy - lcy))
                if jump > 260 and area < 5000:
                    continue
            components.append((area, [x, y, bw, bh], [cx, cy]))

        if not components:
            raw.append(None)
            continue
        components.sort(reverse=True)
        area, bbox, centroid = components[0]
        raw.append(
            {
                "frame_idx": frame_idx,
                "bbox_xywh": bbox,
                "centroid_xy": centroid,
                "area_px": int(area),
                "source": "yellow_hsv_component",
            }
        )
        last_bbox = bbox
    cap.release()

    # Fill short gaps so the box does not flicker during hand occlusion.
    filled: list[dict] = []
    valid_indices = [i for i, item in enumerate(raw) if item is not None]
    for frame_idx, item in enumerate(raw):
        if item is not None:
            filled.append(item)
            continue
        prev = max((i for i in valid_indices if i < frame_idx), default=None)
        nxt = min((i for i in valid_indices if i > frame_idx), default=None)
        if prev is not None and nxt is not None and nxt - prev <= 15:
            ratio = (frame_idx - prev) / (nxt - prev)
            a = np.asarray(raw[prev]["bbox_xywh"], dtype=np.float32)
            b = np.asarray(raw[nxt]["bbox_xywh"], dtype=np.float32)
            bbox = np.round(a * (1.0 - ratio) + b * ratio).astype(int).tolist()
            filled.append(
                {
                    "frame_idx": frame_idx,
                    "bbox_xywh": bbox,
                    "centroid_xy": [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5],
                    "area_px": int(bbox[2] * bbox[3]),
                    "source": "interpolated_yellow_bbox",
                }
            )
        elif prev is not None:
            bbox = raw[prev]["bbox_xywh"]
            filled.append(
                {
                    "frame_idx": frame_idx,
                    "bbox_xywh": bbox,
                    "centroid_xy": [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5],
                    "area_px": int(bbox[2] * bbox[3]),
                    "source": "held_previous_yellow_bbox",
                }
            )
        else:
            filled.append({"frame_idx": frame_idx, "bbox_xywh": None, "source": "missing"})
    return filled


def main() -> None:
    tracks_payload = json.loads(TRACKS_JSON.read_text())
    report = json.loads(ALIGNMENT_REPORT.read_text())
    tracks = tracks_payload["tracks"]
    params = tracks_payload["parameters"]
    intrinsics = report["intrinsics"]
    orientation_config = report.get("orientation_aware_compensation") or {"enabled": False}
    regressor_by_side = {}
    if orientation_config.get("enabled") and orientation_config.get("mano_model_dir"):
        regressor_by_side = load_mano_regressors(Path(orientation_config["mano_model_dir"]))

    num_frames = int(params.get("num_frames", 900))
    video_path = Path(tracks_payload["video"])
    bboxes = detect_yellow_bboxes(video_path, num_frames)
    OUT_BOXES.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "entity_path": OVERLAY_ENTITY,
                "frames": bboxes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    args = SimpleNamespace(
        start=int(params.get("start", 0)),
        num_frames=num_frames,
        rrd_start=0,
        rrd_num_frames=num_frames,
        min_depth_m=0.15,
        max_depth_m=2.5,
        log_full_scene=True,
        scene_point_stride=6,
        max_scene_points=50000,
        point_radius=0.004,
        rgb_overlay_point_radius=1.0,
        log_layered_meshes=True,
    )
    final_tracks = build_final_smoothed_meshes(
        tracks,
        window=int(params.get("final_smoothing_window", 5)),
        regressor_by_side=regressor_by_side,
        orientation_config=orientation_config,
    )
    log_rerun(
        tracks,
        final_tracks,
        video_path=video_path,
        depth_dir=Path(tracks_payload["depth_dir"]),
        save_rrd=OUT_RRD,
        args=args,
        intrinsics=intrinsics,
        orientation_config=orientation_config,
        regressor_by_side=regressor_by_side,
        candidates_json=Path(report["candidates_json"]) if report.get("candidates_json") else None,
        hamer_overlay_dir=Path(report["hamer_overlay_dir"]) if report.get("hamer_overlay_dir") else None,
    )

    for item in bboxes:
        rr.set_time("frame", sequence=int(item["frame_idx"]))
        rgb = read_rgb_frame(video_path, int(item["frame_idx"]), resize_scale=1.0)
        overlay = rgb.copy()
        bbox = item.get("bbox_xywh")
        if bbox is None:
            rr.log(OVERLAY_ENTITY, rr.Image(overlay))
            continue
        x, y, w, h = [int(round(float(v))) for v in bbox]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 40, 20), 4)
        cv2.putText(
            overlay,
            "yellow_object",
            (x, max(28, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 40, 20),
            2,
            cv2.LINE_AA,
        )
        rr.log(OVERLAY_ENTITY, rr.Image(overlay))

    print(f"Rerun: {OUT_RRD}")
    print(f"BBoxes: {OUT_BOXES}")
    print({"frames": num_frames, "detected_or_filled": sum(1 for x in bboxes if x.get("bbox_xywh") is not None)})


if __name__ == "__main__":
    main()
