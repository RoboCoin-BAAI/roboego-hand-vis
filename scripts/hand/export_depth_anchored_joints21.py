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

from roboego_hand_vis.egohand.depth_anchored_joints21 import (  # noqa: E402
    anchor_joints_by_projected_xy_and_depth,
    fit_joints_xy_to_bbox,
    project_joints_to_image,
    sample_depth_anchor,
)
from roboego_hand_vis.egohand.depth_mesh import hamer_rendered_obj_to_camera_opencv, parse_obj_mesh  # noqa: E402
from roboego_hand_vis.egohand.joints21 import HAND21_BONES, HAND21_NAMES  # noqa: E402
from roboego_hand_vis.egohand.orientation import load_mano_regressors, vertices_to_hamer21_joints  # noqa: E402
from roboego_hand_vis.egohand.temporal_tracking import SIDES  # noqa: E402
from scripts.export_depth_pointcloud import read_rgb_frame  # noqa: E402
from scripts.rerun_utils import resolve_existing_artifact_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HaMeR-trusted 21 joints using stereo depth only as a weak wrist/palm anchor."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates_json", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=240)
    parser.add_argument("--mano_model_dir", default="/home/liuyou/Documents/hamer/_DATA/data/mano")
    parser.add_argument("--anchor_mode", choices=["wrist", "palm", "wrist_palm"], default="wrist_palm")
    parser.add_argument("--anchor_joint_index", type=int, default=0)
    parser.add_argument("--depth_window_radius_px", type=int, default=4)
    parser.add_argument("--min_depth_m", type=float, default=0.15)
    parser.add_argument("--max_depth_m", type=float, default=2.5)
    parser.add_argument("--overlay_frames", default="auto", help="'auto', empty string, or comma-separated frame indices.")
    parser.add_argument("--overlay_thumb_width", type=int, default=480)
    parser.add_argument("--overlay_columns", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    candidates_json = Path(args.candidates_json).expanduser().resolve()
    depth_dir = Path(args.depth_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_path = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
    intrinsics = _manifest_intrinsics(manifest)
    width = int(manifest["videos"]["left"].get("width") or intrinsics.get("width") or 0)
    height = int(manifest["videos"]["left"].get("height") or intrinsics.get("height") or 0)
    candidates = json.loads(candidates_json.read_text(encoding="utf-8"))
    candidates_by_frame = {int(frame.get("frame_idx", -1)): frame.get("candidates", []) for frame in candidates.get("frames", [])}
    regressor_by_side = load_mano_regressors(Path(args.mano_model_dir).expanduser())

    joints_by_side = {side: np.full((args.num_frames, 21, 3), np.nan, dtype=np.float32) for side in SIDES}
    raw_joints_by_side = {side: np.full((args.num_frames, 21, 3), np.nan, dtype=np.float32) for side in SIDES}
    valid_by_side = {side: np.zeros((args.num_frames,), dtype=bool) for side in SIDES}
    result_frames: list[dict[str, Any]] = []
    summary = {
        side: {"valid": 0, "missing": 0, "invalid_depth_anchor": 0, "missing_mesh": 0, "total": int(args.num_frames)}
        for side in SIDES
    }

    for offset, frame_idx in enumerate(range(args.start, args.start + args.num_frames)):
        depth_path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
        depth_m = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
        hands_out: dict[str, Any] = {}
        by_side = _select_candidates_by_side(candidates_by_frame.get(frame_idx, []))
        for side in SIDES:
            candidate = by_side.get(side)
            if candidate is None:
                summary[side]["missing"] += 1
                hands_out[side] = {"valid": False, "reason": "missing_candidate"}
                continue
            mesh_path = resolve_existing_artifact_path(candidate["mesh_obj"])
            if not mesh_path.exists():
                summary[side]["missing_mesh"] += 1
                hands_out[side] = {"valid": False, "reason": f"missing_mesh:{mesh_path}"}
                continue
            vertices_obj, _faces, _colors = parse_obj_mesh(mesh_path)
            vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
            raw_joints = vertices_to_hamer21_joints(vertices_camera, regressor_by_side[side])
            raw_joints_by_side[side][offset] = raw_joints
            pixels, pixel_valid = fit_joints_xy_to_bbox(
                raw_joints,
                bbox_xyxy=candidate["bbox_xyxy"],
                width=width,
                height=height,
            )
            if depth_m is None:
                anchor = None
            else:
                anchor = sample_depth_anchor(
                    depth_m,
                    pixels,
                    pixel_valid,
                    mode=args.anchor_mode,
                    window_radius_px=args.depth_window_radius_px,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                )
            if anchor is None or not anchor.valid or not bool(pixel_valid[args.anchor_joint_index]):
                summary[side]["invalid_depth_anchor"] += 1
                hands_out[side] = {
                    "valid": False,
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": "invalid_depth_anchor",
                    "depth_anchor": None if anchor is None else anchor.to_json(),
                }
                continue
            anchored_joints, anchoring = anchor_joints_by_projected_xy_and_depth(
                raw_joints,
                target_depth_m=float(anchor.depth_m),
                pixels_xy=pixels,
                valid_pixels=pixel_valid,
                intrinsics=intrinsics,
                source_joint_index=args.anchor_joint_index,
                min_depth_m=args.min_depth_m,
            )
            if not anchoring.get("valid"):
                summary[side]["invalid_depth_anchor"] += 1
                hands_out[side] = {
                    "valid": False,
                    "candidate_id": candidate.get("candidate_id"),
                    "reason": anchoring.get("reason", "invalid_depth_anchor"),
                    "depth_anchor": anchor.to_json(),
                    "anchoring": anchoring,
                }
                continue
            joints_by_side[side][offset] = anchored_joints
            valid_by_side[side][offset] = True
            summary[side]["valid"] += 1
            hands_out[side] = {
                "valid": True,
                "candidate_id": candidate.get("candidate_id"),
                "mesh_obj": candidate.get("mesh_obj"),
                "depth_anchor": anchor.to_json(),
                "anchoring": anchoring,
                "joints21_image_xy": pixels.astype(float).tolist(),
                "joints21_camera_opencv": anchored_joints.astype(float).tolist(),
                "raw_joints21_camera_opencv": raw_joints.astype(float).tolist(),
            }
        result_frames.append({"frame_idx": int(frame_idx), "hands": hands_out})

    _write_outputs(
        output_dir,
        args=args,
        manifest_path=manifest_path,
        candidates_json=candidates_json,
        depth_dir=depth_dir,
        video_path=video_path,
        intrinsics=intrinsics,
        frames=result_frames,
        joints_by_side=joints_by_side,
        raw_joints_by_side=raw_joints_by_side,
        valid_by_side=valid_by_side,
        summary=summary,
    )
    overlay_paths = _write_overlays(
        output_dir,
        args=args,
        video_path=video_path,
        intrinsics=intrinsics,
        frames=result_frames,
        width=width,
        height=height,
    )
    summary["outputs"] = {
        "json": str(output_dir / "joints21_depth_anchored.json"),
        "npz": str(output_dir / "joints21_depth_anchored.npz"),
        "summary": str(output_dir / "joints21_depth_anchored_summary.json"),
        "overlay_contact_sheet": str(overlay_paths.get("contact_sheet")) if overlay_paths.get("contact_sheet") else None,
    }
    (output_dir / "joints21_depth_anchored_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _manifest_intrinsics(manifest: dict[str, Any]) -> dict[str, float]:
    intr = manifest.get("intrinsics", {}).get("left")
    if not intr:
        raise ValueError("dataset manifest is missing intrinsics.left")
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "width": int(intr.get("width", 0) or 0),
        "height": int(intr.get("height", 0) or 0),
    }


def _select_candidates_by_side(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        side = candidate.get("side") or candidate.get("status")
        if side in SIDES and side not in out:
            out[side] = candidate
    return out


def _write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    candidates_json: Path,
    depth_dir: Path,
    video_path: Path,
    intrinsics: dict[str, float],
    frames: list[dict[str, Any]],
    joints_by_side: dict[str, np.ndarray],
    raw_joints_by_side: dict[str, np.ndarray],
    valid_by_side: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> None:
    payload = {
        "stage": "joints21_depth_anchored_v1",
        "coordinate_frame": "camera_opencv",
        "unit": "meter",
        "method": (
            "HaMeR local 21-joint shape is fitted to the selected hand bbox; stereo depth is sampled "
            "only around fitted wrist/palm pixels and used as the wrist Z anchor."
        ),
        "inputs": {
            "manifest": str(manifest_path),
            "candidates_json": str(candidates_json),
            "depth_dir": str(depth_dir),
            "video": str(video_path),
        },
        "parameters": {
            "start": int(args.start),
            "num_frames": int(args.num_frames),
            "anchor_mode": args.anchor_mode,
            "anchor_joint_index": int(args.anchor_joint_index),
            "depth_window_radius_px": int(args.depth_window_radius_px),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
        },
        "intrinsics": intrinsics,
        "joint_names": HAND21_NAMES,
        "bones": HAND21_BONES,
        "summary": summary,
        "frames": frames,
    }
    (output_dir / "joints21_depth_anchored.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "joints21_depth_anchored.npz",
        left_joints21_camera_opencv=joints_by_side["ego_left"],
        right_joints21_camera_opencv=joints_by_side["ego_right"],
        left_raw_joints21_camera_opencv=raw_joints_by_side["ego_left"],
        right_raw_joints21_camera_opencv=raw_joints_by_side["ego_right"],
        left_valid=valid_by_side["ego_left"],
        right_valid=valid_by_side["ego_right"],
        joint_names=np.asarray(HAND21_NAMES),
        bones=np.asarray(HAND21_BONES, dtype=np.int32),
    )


def _write_overlays(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    video_path: Path,
    intrinsics: dict[str, float],
    frames: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Path]:
    selected = _overlay_frame_indices(args.overlay_frames, args.start, args.num_frames, frames)
    if not selected:
        return {}
    overlay_dir = output_dir / "rgb_skeleton_check"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_overlays(overlay_dir)
    frame_by_idx = {int(frame["frame_idx"]): frame for frame in frames}
    paths: list[Path] = []
    for frame_idx in selected:
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        image = rgb.copy()
        frame = frame_by_idx.get(frame_idx, {"hands": {}})
        labels = [f"frame {frame_idx:06d}"]
        for side in SIDES:
            hand = frame.get("hands", {}).get(side, {})
            if not hand.get("valid"):
                labels.append(f"{side}: {hand.get('reason', 'missing')}")
                continue
            joints = np.asarray(hand["joints21_camera_opencv"], dtype=np.float32)
            image = _draw_skeleton(image, joints, intrinsics, _side_color(side), width=width, height=height)
            labels.append(f"{side}: valid z={hand['depth_anchor']['depth_m']:.3f}m")
        _draw_labels(image, labels)
        path = overlay_dir / f"frame_{frame_idx:06d}_depth_anchored_joints21.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        paths.append(path)
    contact_sheet = overlay_dir / "contact_sheet_depth_anchored_joints21.jpg"
    _write_contact_sheet(paths, contact_sheet, thumb_width=args.overlay_thumb_width, columns=args.overlay_columns)
    return {"contact_sheet": contact_sheet}


def _clear_previous_overlays(overlay_dir: Path) -> None:
    for path in overlay_dir.glob("frame_*_depth_anchored_joints21.jpg"):
        path.unlink()
    contact_sheet = overlay_dir / "contact_sheet_depth_anchored_joints21.jpg"
    if contact_sheet.exists():
        contact_sheet.unlink()


def _overlay_frame_indices(spec: str, start: int, num_frames: int, frames: list[dict[str, Any]]) -> list[int]:
    if spec.strip() == "":
        return []
    if spec != "auto":
        return [int(value) for value in spec.split(",") if value.strip()]
    valid_frames = [
        int(frame["frame_idx"])
        for frame in frames
        if any((hand or {}).get("valid") for hand in frame.get("hands", {}).values())
    ]
    if len(valid_frames) <= 12:
        return valid_frames
    idx = np.linspace(0, len(valid_frames) - 1, 12).astype(np.int64)
    return [valid_frames[int(i)] for i in idx]


def _draw_skeleton(
    image_rgb: np.ndarray,
    joints: np.ndarray,
    intrinsics: dict[str, float],
    color_rgb: tuple[int, int, int],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    out = image_rgb.copy()
    pixels, valid = project_joints_to_image(joints, intrinsics, width=width, height=height)
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for start, end in HAND21_BONES:
        if bool(valid[start]) and bool(valid[end]):
            p0 = tuple(np.round(pixels[start]).astype(np.int32).tolist())
            p1 = tuple(np.round(pixels[end]).astype(np.int32).tolist())
            cv2.line(out, p0, p1, bgr, thickness=3, lineType=cv2.LINE_AA)
    for idx, (u, v) in enumerate(np.round(pixels).astype(np.int32)):
        if bool(valid[idx]):
            cv2.circle(out, (int(u), int(v)), 5, bgr, thickness=-1, lineType=cv2.LINE_AA)
    return out


def _draw_labels(image_rgb: np.ndarray, labels: list[str]) -> None:
    y = 32
    for label in labels:
        cv2.putText(image_rgb, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_rgb, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32


def _write_contact_sheet(paths: list[Path], output_path: Path, *, thumb_width: int, columns: int) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        thumbs.append(cv2.resize(image, (thumb_width, int(round(h * thumb_width / w))), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = max(1, int(columns))
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * cell_h, cols * thumb_width, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet[row * cell_h : row * cell_h + thumb.shape[0], col * thumb_width : col * thumb_width + thumb.shape[1]] = thumb
    cv2.imwrite(str(output_path), sheet)


def _side_color(side: str) -> tuple[int, int, int]:
    return (60, 255, 120) if side == "ego_left" else (80, 190, 255)


if __name__ == "__main__":
    main()
