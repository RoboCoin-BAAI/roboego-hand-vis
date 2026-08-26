#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.final_hand_trajectory_polish import polish_depth_anchored_joints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polish final hand 21-joint trajectories with wrist-first continuity.")
    parser.add_argument("--joints_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--preview_video", default="")
    parser.add_argument("--max_interp_gap", type=int, default=10)
    parser.add_argument("--wrist_jump_m", type=float, default=0.0)
    parser.add_argument("--max_local_wrist_deviation_m", type=float, default=0.0)
    parser.add_argument("--local_window_radius", type=int, default=5)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--smooth_relative_fingers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_preview", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joints_json = Path(args.joints_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(joints_json.read_text(encoding="utf-8"))
    result = polish_depth_anchored_joints(
        payload,
        max_interp_gap=args.max_interp_gap,
        wrist_jump_m=args.wrist_jump_m,
        max_local_wrist_deviation_m=args.max_local_wrist_deviation_m,
        local_window_radius=args.local_window_radius,
        smooth_window=args.smooth_window,
        smooth_relative_fingers=args.smooth_relative_fingers,
    )
    outputs = _write_outputs(result, output_dir, source_joints_json=joints_json)
    if args.write_preview:
        preview = _write_preview_video(
            result,
            manifest=Path(args.manifest).expanduser() if args.manifest else None,
            output_path=Path(args.preview_video).expanduser() if args.preview_video else output_dir / "hand_estimation_preview_smoothed.mp4",
        )
        if preview:
            outputs["preview_video"] = str(preview)
    print(json.dumps({"report": result["report"], "outputs": outputs}, indent=2, ensure_ascii=False))


def _write_outputs(result: dict[str, Any], output_dir: Path, *, source_joints_json: Path) -> dict[str, str]:
    json_path = output_dir / "smoothed_joints21.json"
    report_path = output_dir / "trajectory_polish_report.json"
    npz_path = output_dir / "smoothed_joints21.npz"
    serializable = {**result, "source_joints_json": str(source_joints_json)}
    json_path.write_text(json.dumps(_jsonable(serializable), indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(_jsonable(result["report"]), indent=2, ensure_ascii=False), encoding="utf-8")
    arrays = _compact_arrays(result)
    np.savez_compressed(npz_path, **arrays)
    return {
        "smoothed_joints21_json": str(json_path),
        "smoothed_joints21_npz": str(npz_path),
        "trajectory_polish_report_json": str(report_path),
    }


def _compact_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    frames = result.get("frames") or []
    out: dict[str, np.ndarray] = {}
    for side, prefix in (("ego_left", "left"), ("ego_right", "right")):
        joints = np.full((len(frames), 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros((len(frames),), dtype=bool)
        interpolated = np.zeros((len(frames),), dtype=bool)
        for idx, frame in enumerate(frames):
            hand = (frame.get("hands") or {}).get(side) or {}
            if not hand.get("valid"):
                continue
            arr = np.asarray(hand.get("joints21_camera_opencv"), dtype=np.float32)
            if arr.shape == (21, 3):
                joints[idx] = arr
                valid[idx] = True
                interpolated[idx] = hand.get("status") == "interpolated"
        out[f"{prefix}_joints21_camera_opencv"] = joints
        out[f"{prefix}_valid"] = valid
        out[f"{prefix}_interpolated"] = interpolated
    return out


def _write_preview_video(result: dict[str, Any], *, manifest: Path | None, output_path: Path) -> Path | None:
    if manifest is None or not manifest.is_file():
        return None
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    video_path = Path(manifest_payload["videos"]["left"]["path"]).expanduser()
    if not video_path.is_file():
        return None
    intr = manifest_payload.get("intrinsics", {}).get("left") or result.get("intrinsics") or {}
    if not all(key in intr for key in ("fx", "fy", "cx", "cy")):
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(manifest_payload["videos"]["left"].get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None
    frames_by_idx = {int(frame["frame_idx"]): frame for frame in result.get("frames", [])}
    bones = result.get("bones") or []
    fx, fy, cx, cy = [float(intr[key]) for key in ("fx", "fy", "cx", "cy")]
    colors = {"ego_left": (60, 255, 120), "ego_right": (80, 190, 255)}
    idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frame = frames_by_idx.get(idx, {"hands": {}})
        labels = []
        for side in ("ego_left", "ego_right"):
            hand = (frame.get("hands") or {}).get(side) or {}
            if not hand.get("valid"):
                continue
            pts, valid = _project(np.asarray(hand["joints21_camera_opencv"], dtype=np.float32), fx, fy, cx, cy, width, height)
            color = colors[side]
            for start, end in bones:
                if start < len(valid) and end < len(valid) and valid[start] and valid[end]:
                    cv2.line(image, tuple(np.round(pts[start]).astype(int)), tuple(np.round(pts[end]).astype(int)), color, 3, cv2.LINE_AA)
            for point, is_valid in zip(np.round(pts).astype(int), valid):
                if bool(is_valid):
                    cv2.circle(image, tuple(point), 5, color, -1, cv2.LINE_AA)
            labels.append(f"{side}:{hand.get('status', 'smoothed')}")
        _draw_label(image, f"frame {idx:06d}  " + (", ".join(labels) if labels else "no valid hand"))
        writer.write(image)
        idx += 1
    cap.release()
    writer.release()
    return output_path


def _project(joints: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    z = joints[:, 2]
    valid = np.isfinite(joints).all(axis=1) & (z > 1e-6)
    pts = np.zeros((len(joints), 2), dtype=np.float32)
    pts[:, 0] = fx * joints[:, 0] / np.maximum(z, 1e-6) + cx
    pts[:, 1] = fy * joints[:, 1] / np.maximum(z, 1e-6) + cy
    valid &= (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    return pts, valid


def _draw_label(image: np.ndarray, text: str) -> None:
    cv2.putText(image, text, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
