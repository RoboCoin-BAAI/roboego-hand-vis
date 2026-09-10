from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.depth_mesh import (  # noqa: E402
    hamer_rendered_obj_to_camera_opencv,
    parse_obj_mesh,
    translate_points,
)
from scripts.export_depth_pointcloud import ROPEDIA_K, read_rgb_frame  # noqa: E402
from scripts.rerun_utils import resolve_existing_artifact_path  # noqa: E402


SIDES = ("ego_left", "ego_right")
OFFICIAL_SIDE_KEY = {"ego_left": "left", "ego_right": "right"}

# Matches HaMeR's MANO wrapper:
# wrist, thumb1..4, index1..4, middle1..4, ring1..4, pinky1..4.
MANO_TO_21 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO_TIP_VERTEX_IDS = [744, 320, 443, 554, 671]  # thumb, index, middle, ring, pinky.
FINGERTIP_IDS = [4, 8, 12, 16, 20]
PALM_IDS = [0, 5, 9, 13, 17]


def install_chumpy_pickle_stubs() -> None:
    """Provide enough chumpy module names to unpickle MANO model files.

    We only need the sparse J_regressor from MANO_LEFT/RIGHT.pkl. Some other
    entries reference chumpy classes during pickle import, but we do not use
    those objects here.
    """
    if "chumpy" in sys.modules:
        return
    chumpy = types.ModuleType("chumpy")
    chumpy.__path__ = []
    reordering = types.ModuleType("chumpy.reordering")
    ch = types.ModuleType("chumpy.ch")

    class Select:  # noqa: D401 - pickle placeholder
        pass

    class Ch:  # noqa: D401 - pickle placeholder
        pass

    reordering.Select = Select
    ch.Ch = Ch
    chumpy.Ch = Ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.reordering"] = reordering
    sys.modules["chumpy.ch"] = ch


def load_mano_joint_regressor(model_path: Path) -> np.ndarray:
    install_chumpy_pickle_stubs()
    with model_path.open("rb") as f:
        model = pickle.load(f, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.shape != (16, 778):
        raise ValueError(f"Unexpected MANO J_regressor shape in {model_path}: {regressor.shape}")
    return regressor


def vertices_to_hamer21_joints(vertices: np.ndarray, regressor_16: np.ndarray) -> np.ndarray:
    if vertices.shape != (778, 3):
        raise ValueError(f"Expected MANO mesh vertices shape (778, 3), got {vertices.shape}")
    mano16 = regressor_16 @ vertices.astype(np.float32)
    tips = vertices[np.asarray(MANO_TIP_VERTEX_IDS, dtype=np.int64)]
    joints21 = np.concatenate([mano16, tips], axis=0)[np.asarray(MANO_TO_21, dtype=np.int64)]
    return joints21.astype(np.float32)


def load_official_joints(annotation_path: Path, start: int, num_frames: int) -> Dict[str, np.ndarray]:
    end = start + num_frames
    with h5py.File(annotation_path, "r") as f:
        left = np.asarray(f["hand_mocap/left_joints_3d"][start:end], dtype=np.float32)
        right = np.asarray(f["hand_mocap/right_joints_3d"][start:end], dtype=np.float32)
    return {"ego_left": left, "ego_right": right}


def current_vertices_camera_from_hand(hand: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    mesh_path = resolve_existing_artifact_path(hand["mesh_obj"])
    vertices_obj, _, _ = parse_obj_mesh(mesh_path)
    vertices_camera = hamer_rendered_obj_to_camera_opencv(vertices_obj)
    current_source = hand.get("current_best_mesh_source")
    tracking = hand.get("tracking_smoothed") or {}
    if current_source == "tracking_smoothed_shape_preserved_mano" or hand.get("current_layer") == "smooth":
        translation = tracking.get("smoothed_translation")
        if translation is None:
            raise ValueError(f"Smooth hand has no smoothed_translation: frame hand={hand}")
        return translate_points(vertices_camera, translation), "smooth"

    transform = hand.get("shape_preserved_fit_transform") or {}
    translation = transform.get("translation")
    if translation is None:
        raise ValueError(f"Observation hand has no shape_preserved translation: frame hand={hand}")
    layer = str(hand.get("current_layer") or hand.get("layer") or "normal")
    return translate_points(vertices_camera, translation), layer


def load_our_joints(
    alignment_report: Path,
    regressor_by_side: Dict[str, np.ndarray],
    start: int,
    num_frames: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, list]]:
    with alignment_report.open("r", encoding="utf-8") as f:
        report = json.load(f)

    joints = {side: np.full((num_frames, 21, 3), np.nan, dtype=np.float32) for side in SIDES}
    vertices = {side: np.full((num_frames, 778, 3), np.nan, dtype=np.float32) for side in SIDES}
    meta = {side: [None for _ in range(num_frames)] for side in SIDES}

    for frame in report["frames"]:
        frame_idx = int(frame["frame_idx"])
        offset = frame_idx - start
        if offset < 0 or offset >= num_frames:
            continue
        for hand in frame.get("hands", []):
            side = hand.get("canonical_status") or hand.get("status")
            if side not in SIDES:
                continue
            verts, layer = current_vertices_camera_from_hand(hand)
            vertices[side][offset] = verts
            joints[side][offset] = vertices_to_hamer21_joints(verts, regressor_by_side[side])
            meta[side][offset] = {
                "frame_idx": frame_idx,
                "side": side,
                "layer": layer,
                "status": hand.get("status"),
                "current_layer": hand.get("current_layer"),
                "current_best_mesh_source": hand.get("current_best_mesh_source"),
                "mesh_obj": hand.get("mesh_obj"),
            }
    return joints, vertices, meta


def valid_joints(arr: np.ndarray) -> bool:
    return bool(np.isfinite(arr).all() and not np.allclose(arr, 0.0, atol=1e-7))


def point_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=1)


def project_points(points: np.ndarray, intrinsics: dict, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    u = np.full(len(points), np.nan, dtype=np.float32)
    v = np.full(len(points), np.nan, dtype=np.float32)
    u[valid] = intrinsics["fx"] * points[valid, 0] / z[valid] + intrinsics["cx"]
    v[valid] = intrinsics["fy"] * points[valid, 1] / z[valid] + intrinsics["cy"]
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u, v], axis=1), valid


def temporal_jitter(joints: np.ndarray, valid: np.ndarray) -> Dict[str, Optional[float]]:
    diffs = []
    for i in range(1, len(joints)):
        if valid[i] and valid[i - 1]:
            diffs.append(float(np.mean(np.linalg.norm(joints[i] - joints[i - 1], axis=1))))
    if not diffs:
        return {"mean_m": None, "median_m": None, "p90_m": None}
    arr = np.asarray(diffs, dtype=np.float32)
    return {
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
    }


def summarize(values: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if len(arr) == 0:
        return {"mean_m": None, "median_m": None, "p90_m": None, "count": 0}
    return {
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p90_m": float(np.percentile(arr, 90)),
        "count": int(len(arr)),
    }


def draw_joints(image: np.ndarray, pixels: np.ndarray, valid: np.ndarray, color: Tuple[int, int, int]) -> None:
    for idx, (pt, ok) in enumerate(zip(pixels, valid)):
        if not ok:
            continue
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        cv2.circle(image, (x, y), 3 if idx in [0, *FINGERTIP_IDS] else 2, color, -1, lineType=cv2.LINE_AA)


def make_projection_contact_sheet(
    video_path: Path,
    official: Dict[str, np.ndarray],
    ours: Dict[str, np.ndarray],
    output_path: Path,
    start: int,
    num_frames: int,
    max_frames: int,
) -> None:
    if max_frames <= 0:
        return
    frame_ids = np.linspace(0, num_frames - 1, min(max_frames, num_frames)).astype(int)
    panels = []
    for offset in frame_ids:
        frame_idx = start + int(offset)
        rgb = read_rgb_frame(video_path, frame_idx, resize_scale=1.0)
        panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = panel.shape[:2]
        for side in SIDES:
            off = official[side][offset]
            our = ours[side][offset]
            if valid_joints(off):
                pix, val = project_points(off, ROPEDIA_K, w, h)
                draw_joints(panel, pix, val, (0, 255, 255))
            if valid_joints(our):
                pix, val = project_points(our, ROPEDIA_K, w, h)
                draw_joints(panel, pix, val, (0, 255, 0))
        cv2.putText(
            panel,
            f"frame {frame_idx}  official=yellow ours=green",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumb_w = 420
        scale = thumb_w / w
        panels.append(cv2.resize(panel, (thumb_w, int(round(h * scale))), interpolation=cv2.INTER_AREA))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    ph, pw = panels[0].shape[:2]
    sheet = np.zeros((rows * ph, cols * pw, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        sheet[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = panel
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=Path("data/ropedia_data/annotation.hdf5"))
    parser.add_argument(
        "--alignment_report",
        type=Path,
        default=Path(
            "tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_layered/hand_depth_mesh_alignment_report.json"
        ),
    )
    parser.add_argument("--video", type=Path, default=Path("data/ropedia_data/stereo_left.mp4"))
    parser.add_argument("--mano_model_dir", type=Path, default=Path("../hamer/_DATA/data/mano"))
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/ropedia_official_hand_eval_000000_000299"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--contact_sheet_frames", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regressor_by_side = {
        "ego_left": load_mano_joint_regressor(args.mano_model_dir / "MANO_LEFT.pkl"),
        "ego_right": load_mano_joint_regressor(args.mano_model_dir / "MANO_RIGHT.pkl"),
    }
    official = load_official_joints(args.annotation, args.start, args.num_frames)
    ours, vertices, meta = load_our_joints(args.alignment_report, regressor_by_side, args.start, args.num_frames)

    np.savez_compressed(
        args.output_dir / "our_hand_joints_camera_opencv.npz",
        left_joints_3d_camera_opencv=ours["ego_left"],
        right_joints_3d_camera_opencv=ours["ego_right"],
        left_mesh_vertices_camera_opencv=vertices["ego_left"],
        right_mesh_vertices_camera_opencv=vertices["ego_right"],
        official_left_joints_3d_camera=official["ego_left"],
        official_right_joints_3d_camera=official["ego_right"],
        frame_indices=np.arange(args.start, args.start + args.num_frames, dtype=np.int32),
    )
    (args.output_dir / "our_hand_joints_meta.json").write_text(
        json.dumps({"ego_left": meta["ego_left"], "ego_right": meta["ego_right"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    summary: Dict[str, Any] = {
        "inputs": {
            "annotation": str(args.annotation),
            "alignment_report": str(args.alignment_report),
            "mano_model_dir": str(args.mano_model_dir),
            "start": args.start,
            "num_frames": args.num_frames,
        },
        "coordinate_frame": "camera_opencv / Ropedia camera frame, x right, y down, z forward, meters",
        "joint_order_assumption": "HaMeR MANO 21 order: wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4",
        "sides": {},
    }

    for side in SIDES:
        mpjpe_values = []
        wrist_values = []
        fingertip_values = []
        palm_values = []
        reproj_values = []
        valid_ours = np.zeros(args.num_frames, dtype=bool)
        valid_official = np.zeros(args.num_frames, dtype=bool)
        valid_compare = np.zeros(args.num_frames, dtype=bool)
        layer_counts: Dict[str, int] = {}

        for offset in range(args.num_frames):
            frame_idx = args.start + offset
            pred = ours[side][offset]
            gt = official[side][offset]
            our_ok = valid_joints(pred)
            off_ok = valid_joints(gt)
            valid_ours[offset] = our_ok
            valid_official[offset] = off_ok
            layer = (meta[side][offset] or {}).get("layer", "missing")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

            row = {
                "frame_idx": frame_idx,
                "side": side,
                "our_valid": int(our_ok),
                "official_valid": int(off_ok),
                "layer": layer,
                "mpjpe_m": "",
                "wrist_error_m": "",
                "fingertip_error_m": "",
                "palm_error_m": "",
                "reprojection_error_px": "",
            }
            if our_ok and off_ok:
                valid_compare[offset] = True
                err = point_errors(pred, gt)
                mpjpe = float(np.mean(err))
                wrist = float(err[0])
                fingertip = float(np.mean(err[FINGERTIP_IDS]))
                palm = float(
                    np.linalg.norm(np.mean(pred[PALM_IDS], axis=0) - np.mean(gt[PALM_IDS], axis=0))
                )
                pred_px, pred_valid_px = project_points(pred, ROPEDIA_K, width=512, height=512)
                gt_px, gt_valid_px = project_points(gt, ROPEDIA_K, width=512, height=512)
                reproj_valid = pred_valid_px & gt_valid_px
                reproj = (
                    float(np.mean(np.linalg.norm(pred_px[reproj_valid] - gt_px[reproj_valid], axis=1)))
                    if np.any(reproj_valid)
                    else np.nan
                )
                mpjpe_values.append(mpjpe)
                wrist_values.append(wrist)
                fingertip_values.append(fingertip)
                palm_values.append(palm)
                reproj_values.append(reproj)
                row.update(
                    {
                        "mpjpe_m": mpjpe,
                        "wrist_error_m": wrist,
                        "fingertip_error_m": fingertip,
                        "palm_error_m": palm,
                        "reprojection_error_px": reproj,
                    }
                )
            rows.append(row)

        summary["sides"][side] = {
            "our_valid_frames": int(np.sum(valid_ours)),
            "official_valid_frames": int(np.sum(valid_official)),
            "compared_frames": int(np.sum(valid_compare)),
            "our_missing_rate": float(1.0 - np.mean(valid_ours)),
            "official_missing_rate": float(1.0 - np.mean(valid_official)),
            "layer_counts": layer_counts,
            "mpjpe": summarize(mpjpe_values),
            "wrist_error": summarize(wrist_values),
            "fingertip_error": summarize(fingertip_values),
            "palm_error": summarize(palm_values),
            "reprojection_error_px": summarize(reproj_values),
            "our_temporal_jitter": temporal_jitter(ours[side], valid_ours),
            "official_temporal_jitter": temporal_jitter(official[side], valid_official),
        }

    with (args.output_dir / "official_vs_ours_per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "official_vs_ours_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    make_projection_contact_sheet(
        args.video,
        official,
        ours,
        args.output_dir / "contact_sheet_projection_check.png",
        args.start,
        args.num_frames,
        args.contact_sheet_frames,
    )

    print(json.dumps(summary["sides"], indent=2, ensure_ascii=False))
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
