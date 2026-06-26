#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from roboego_hand_vis.egohand.surface_compensation import orientation_aware_compensate_points
from scripts.export_and_evaluate_ropedia_hand_joints import (
    FINGERTIP_IDS,
    PALM_IDS,
    SIDES,
    load_official_joints,
    load_our_joints,
    load_mano_joint_regressor,
    point_errors,
    project_points,
    summarize,
    temporal_jitter,
    valid_joints,
    vertices_to_hamer21_joints,
)
from scripts.export_depth_pointcloud import ROPEDIA_K


def evaluate(pred_all: np.ndarray, official_all: np.ndarray) -> Dict:
    mpjpe = []
    wrist = []
    palm = []
    fingertip = []
    non_wrist = []
    reproj = []
    valid_pred = []
    for pred, gt in zip(pred_all, official_all):
        pred_ok = valid_joints(pred)
        valid_pred.append(pred_ok)
        if not pred_ok or not valid_joints(gt):
            continue
        err = point_errors(pred, gt)
        mpjpe.append(float(np.mean(err)))
        wrist.append(float(err[0]))
        palm.append(float(np.mean(err[PALM_IDS])))
        fingertip.append(float(np.mean(err[FINGERTIP_IDS])))
        non_wrist.append(float(np.mean(err[1:])))
        pred_px, pred_valid = project_points(pred, ROPEDIA_K, width=512, height=512)
        gt_px, gt_valid = project_points(gt, ROPEDIA_K, width=512, height=512)
        ok = pred_valid & gt_valid
        reproj.append(float(np.mean(np.linalg.norm(pred_px[ok] - gt_px[ok], axis=1))) if ok.any() else np.nan)
    return {
        "compared_frames": int(len(mpjpe)),
        "mpjpe": summarize(mpjpe),
        "wrist_error": summarize(wrist),
        "palm_error": summarize(palm),
        "fingertip_error": summarize(fingertip),
        "non_wrist_mpjpe": summarize(non_wrist),
        "reprojection_error_px": summarize(reproj),
        "temporal_jitter": temporal_jitter(pred_all, np.asarray(valid_pred, dtype=bool)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate orientation-aware surface compensation against Ropedia official joints.")
    parser.add_argument("--annotation", type=Path, default=Path("/home/liuyou/Documents/ropedia_data/annotation.hdf5"))
    parser.add_argument(
        "--alignment_report",
        type=Path,
        default=Path("tmp/ropedia_hamer_new_method_shape_preserved_000000_000299_area0007_recovered_z_interp_layered/hand_depth_mesh_alignment_report.json"),
    )
    parser.add_argument("--mano_model_dir", type=Path, default=Path("/home/liuyou/Documents/hamer/_DATA/data/mano"))
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/ropedia_orientation_compensation_eval"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--depth_bias_m", type=float, default=0.035)
    parser.add_argument("--surface_min_m", type=float, default=0.005)
    parser.add_argument("--surface_max_m", type=float, default=0.020)
    parser.add_argument("--ray_blend", type=float, default=0.3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    regressors = {
        "ego_left": load_mano_joint_regressor(args.mano_model_dir / "MANO_LEFT.pkl"),
        "ego_right": load_mano_joint_regressor(args.mano_model_dir / "MANO_RIGHT.pkl"),
    }
    official = load_official_joints(args.annotation, args.start, args.num_frames)
    current_joints, current_vertices, _ = load_our_joints(
        args.alignment_report,
        regressors,
        args.start,
        args.num_frames,
    )

    compensated = {
        side: np.full_like(current_joints[side], np.nan, dtype=np.float32)
        for side in SIDES
    }
    compensated_vertices = {
        side: np.full_like(current_vertices[side], np.nan, dtype=np.float32)
        for side in SIDES
    }
    for side in SIDES:
        for idx in range(args.num_frames):
            joints = current_joints[side][idx]
            vertices = current_vertices[side][idx]
            if not valid_joints(joints) or not np.isfinite(vertices).all():
                continue
            new_vertices, _ = orientation_aware_compensate_points(
                vertices,
                joints,
                depth_bias_m=args.depth_bias_m,
                surface_min_m=args.surface_min_m,
                surface_max_m=args.surface_max_m,
                ray_blend=args.ray_blend,
            )
            compensated_vertices[side][idx] = new_vertices
            compensated[side][idx] = vertices_to_hamer21_joints(new_vertices, regressors[side])

    summary = {
        "parameters": {
            "depth_bias_m": args.depth_bias_m,
            "surface_min_m": args.surface_min_m,
            "surface_max_m": args.surface_max_m,
            "ray_blend": args.ray_blend,
        },
        "current": {side: evaluate(current_joints[side], official[side]) for side in SIDES},
        "orientation_compensated": {side: evaluate(compensated[side], official[side]) for side in SIDES},
    }
    (args.output_dir / "orientation_compensation_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "orientation_compensated_joints_vertices.npz",
        left_joints_3d_camera_opencv=compensated["ego_left"],
        right_joints_3d_camera_opencv=compensated["ego_right"],
        left_mesh_vertices_camera_opencv=compensated_vertices["ego_left"],
        right_mesh_vertices_camera_opencv=compensated_vertices["ego_right"],
        frame_indices=np.arange(args.start, args.start + args.num_frames, dtype=np.int32),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
