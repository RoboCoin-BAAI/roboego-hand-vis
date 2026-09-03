from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: list[float]
    score: float = 1.0


@dataclass(frozen=True)
class FaceBlurConfig:
    detect_every_n_frames: int = 3
    face_score_threshold: float = 0.4
    nms_iou_threshold: float = 0.5
    scale_factor_detections: float = 1.35
    max_gap_frames: int = 12
    smooth_window: int = 5
    match_iou_threshold: float = 0.05
    match_center_dist_ratio: float = 0.12
    crf: int = 18
    preset: str = "medium"


def expand_detection_schedule(*, frame_count: int, detect_every_n_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    step = max(1, int(detect_every_n_frames))
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def build_temporal_tracks(
    detections_by_frame: dict[int, list[Detection]],
    *,
    frame_count: int,
    config: FaceBlurConfig = FaceBlurConfig(),
) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    next_track_id = 0
    image_diag = 1.0
    for frame_idx in sorted(detections_by_frame):
        frame_detections = detections_by_frame[frame_idx]
        assigned_tracks: set[int] = set()
        for detection in frame_detections:
            best_track = _best_matching_track(
                tracks,
                detection,
                frame_idx=frame_idx,
                image_diag=image_diag,
                config=config,
                assigned_tracks=assigned_tracks,
            )
            observation = {"frame_idx": frame_idx, "bbox": list(detection.bbox), "score": float(detection.score)}
            if best_track is None:
                tracks.append({"track_id": next_track_id, "observations": [observation]})
                next_track_id += 1
            else:
                best_track["observations"].append(observation)
                assigned_tracks.add(int(best_track["track_id"]))
    return interpolate_tracks(
        tracks,
        max_gap_frames=config.max_gap_frames,
        smooth_window=config.smooth_window,
        frame_count=frame_count,
    )


def interpolate_tracks(
    tracks: list[dict[str, Any]],
    *,
    max_gap_frames: int,
    smooth_window: int,
    frame_count: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for track in tracks:
        observations = sorted(track.get("observations", []), key=lambda item: int(item["frame_idx"]))
        filled: list[dict[str, Any]] = []
        for idx, obs in enumerate(observations):
            if idx:
                prev = observations[idx - 1]
                gap = int(obs["frame_idx"]) - int(prev["frame_idx"])
                if 1 < gap <= max_gap_frames + 1:
                    for frame_idx in range(int(prev["frame_idx"]) + 1, int(obs["frame_idx"])):
                        alpha = (frame_idx - int(prev["frame_idx"])) / gap
                        bbox = _lerp_box(prev["bbox"], obs["bbox"], alpha)
                        filled.append({"frame_idx": frame_idx, "bbox": bbox, "score": min(prev["score"], obs["score"]), "interpolated": True})
            filled.append({**obs, "interpolated": bool(obs.get("interpolated", False))})
        smoothed = _smooth_observations(filled, smooth_window=smooth_window)
        if frame_count is not None:
            smoothed = [obs for obs in smoothed if 0 <= int(obs["frame_idx"]) < frame_count]
        out.append({"track_id": track["track_id"], "observations": smoothed})
    return out


def detections_to_frame_boxes(tracks: list[dict[str, Any]], *, frame_count: int) -> dict[int, list[dict[str, Any]]]:
    frame_boxes: dict[int, list[dict[str, Any]]] = {}
    for track in tracks:
        track_id = int(track["track_id"])
        for obs in track.get("observations", []):
            frame_idx = int(obs["frame_idx"])
            if not 0 <= frame_idx < frame_count:
                continue
            frame_boxes.setdefault(frame_idx, []).append(
                {
                    "track_id": track_id,
                    "bbox": [float(v) for v in obs["bbox"]],
                    "score": float(obs.get("score", 0.0)),
                    "interpolated": bool(obs.get("interpolated", False)),
                }
            )
    return frame_boxes


def run_egoblur_temporal_face_blur(
    *,
    input_video: Path,
    output_video: Path,
    face_model_path: Path,
    egoblur_root: Path,
    output_json: Path,
    config: FaceBlurConfig = FaceBlurConfig(),
    camera_name: str = "camera-rgb",
) -> dict[str, Any]:
    input_video = Path(input_video).expanduser()
    output_video = Path(output_video).expanduser()
    output_json = Path(output_json).expanduser()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    _ensure_egoblur_importable(egoblur_root)
    from gen2.script.constants import FACE_THRESHOLDS_GEN2, RESIZE_MAX_GEN2, RESIZE_MIN_GEN2
    from gen2.script.detectron2.export.torchscript_patch import patch_instances
    from gen2.script.predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS
    from gen2.script.utils import _get_threshold, get_device, get_image_tensor, visualize

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {input_video}")

    threshold = _get_threshold(camera_name, config.face_score_threshold, FACE_THRESHOLDS_GEN2)
    detector = EgoblurDetector(
        model_path=str(face_model_path),
        device=get_device(),
        detection_class=ClassID.FACE,
        score_threshold=threshold,
        nms_iou_threshold=config.nms_iou_threshold,
        resize_aug={"min_size_test": RESIZE_MIN_GEN2, "max_size_test": RESIZE_MAX_GEN2},
    )

    started = time.time()
    schedule = expand_detection_schedule(frame_count=frame_count, detect_every_n_frames=config.detect_every_n_frames)
    detections_by_frame: dict[int, list[Detection]] = {}
    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        for frame_idx in schedule:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            image_tensor = get_image_tensor(frame)
            boxes = detector.run(image_tensor)[0]
            detections_by_frame[frame_idx] = [Detection(bbox=[float(v) for v in box], score=threshold) for box in boxes]
    cap.release()

    tracks = build_temporal_tracks(detections_by_frame, frame_count=frame_count, config=config)
    frame_boxes = detections_to_frame_boxes(tracks, frame_count=frame_count)
    _write_blurred_video(
        input_video=input_video,
        output_video=output_video,
        fps=fps,
        size=(width, height),
        frame_boxes=frame_boxes,
        scale_factor_detections=config.scale_factor_detections,
        visualize=visualize,
        crf=config.crf,
        preset=config.preset,
    )

    report = {
        "stage": "egoblur_temporal_face_blur",
        "status": "complete",
        "input_video": str(input_video),
        "output_video": str(output_video),
        "face_model_path": str(face_model_path),
        "egoblur_root": str(egoblur_root),
        "camera_name": camera_name,
        "video": {"fps": fps, "frame_count": frame_count, "width": width, "height": height},
        "config": config.__dict__,
        "detected_frame_count": len(detections_by_frame),
        "tracked_frame_count": len(frame_boxes),
        "track_count": len(tracks),
        "tracks": tracks,
        "elapsed_s": time.time() - started,
    }
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _write_blurred_video(
    *,
    input_video: Path,
    output_video: Path,
    fps: float,
    size: tuple[int, int],
    frame_boxes: dict[int, list[dict[str, Any]]],
    scale_factor_detections: float,
    visualize: Any,
    crf: int,
    preset: str,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {output_video}")
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boxes = [item["bbox"] for item in frame_boxes.get(frame_idx, [])]
            if boxes:
                frame = visualize(frame, boxes, scale_factor_detections)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    _reencode_h264_if_available(output_video, fps=fps, crf=crf, preset=preset)


def _reencode_h264_if_available(output_video: Path, *, fps: float, crf: int, preset: str) -> None:
    tmp_path = output_video.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(output_video),
        "-r",
        f"{fps:.6f}",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(tmp_path),
    ]
    try:
        import subprocess

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        return
    tmp_path.replace(output_video)


def _best_matching_track(
    tracks: list[dict[str, Any]],
    detection: Detection,
    *,
    frame_idx: int,
    image_diag: float,
    config: FaceBlurConfig,
    assigned_tracks: set[int],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for track in tracks:
        if int(track["track_id"]) in assigned_tracks:
            continue
        last = track["observations"][-1]
        gap = frame_idx - int(last["frame_idx"])
        if gap <= 0 or gap > config.max_gap_frames + config.detect_every_n_frames:
            continue
        iou = _box_iou(last["bbox"], detection.bbox)
        center_dist = _center_distance(last["bbox"], detection.bbox)
        center_ok = center_dist <= max(image_diag * config.match_center_dist_ratio, _box_diag(last["bbox"]) * 3.0)
        if iou < config.match_iou_threshold and not center_ok:
            continue
        score = iou - 0.001 * center_dist
        if best is None or score > best[0]:
            best = (score, track)
    return best[1] if best else None


def _smooth_observations(observations: list[dict[str, Any]], *, smooth_window: int) -> list[dict[str, Any]]:
    if smooth_window <= 1 or len(observations) <= 2:
        return observations
    radius = max(1, smooth_window // 2)
    out: list[dict[str, Any]] = []
    for idx, obs in enumerate(observations):
        left = max(0, idx - radius)
        right = min(len(observations), idx + radius + 1)
        boxes = np.asarray([item["bbox"] for item in observations[left:right]], dtype=np.float32)
        out.append({**obs, "bbox": boxes.mean(axis=0).astype(float).tolist()})
    return out


def _lerp_box(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [float(x + (y - x) * alpha) for x, y in zip(a, b)]


def _box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bx, by = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float(np.hypot(ax - bx, ay - by))


def _box_diag(box: list[float]) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _ensure_egoblur_importable(egoblur_root: Path) -> None:
    root = str(Path(egoblur_root).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
