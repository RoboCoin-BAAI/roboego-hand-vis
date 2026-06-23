#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboegopipe.egohand.filtering import EgoHandFilterConfig, select_ego_hands


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


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


def iter_video_frames(
    video_path: Path,
    start: int,
    num_frames: int,
    stride: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        for offset in range(num_frames):
            frame_idx = start + offset * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield frame_idx, frame_bgr
    finally:
        cap.release()


def load_depth(depth_dir: Optional[Path], frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if depth_dir is None:
        return None, None
    path = depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy"
    if not path.exists():
        return None, None
    return np.load(path), str(path)


def status_color(status: str) -> Tuple[int, int, int]:
    if status in {"ego_left", "ego_right"}:
        return (40, 220, 40)
    if status in {"ego_left_recovered", "ego_right_recovered"}:
        return (0, 220, 255)
    if status == "other_hands":
        return (160, 160, 160)
    return (0, 0, 255)


def draw_candidates(image_bgr: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for idx, candidate in enumerate(candidates):
        x1, y1, x2, y2 = [int(round(v)) for v in candidate["bbox_xyxy"]]
        color = status_color(candidate["status"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        depth = candidate.get("median_depth_m")
        depth_text = "d=na" if depth is None else f"d={depth:.2f}"
        label = (
            f"{idx}:{candidate['status']} "
            f"{candidate.get('handedness', '?')} "
            f"c={candidate.get('confidence', 0.0):.2f} "
            f"s={candidate.get('ego_score', 0.0):.2f} "
            f"{depth_text}"
        )
        text_org = (max(0, x1), max(16, y1 - 6))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            out,
            (text_org[0], text_org[1] - th - baseline - 2),
            (text_org[0] + tw + 4, text_org[1] + baseline),
            color,
            thickness=-1,
        )
        cv2.putText(
            out,
            label,
            (text_org[0] + 2, text_org[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def make_contact_sheet(
    image_paths: List[Path],
    output_path: Path,
    columns: int = 4,
    thumb_width: int = 360,
) -> None:
    if not image_paths:
        return

    thumbs = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = thumb_width / image.shape[1]
        thumb_height = int(round(image.shape[0] * scale))
        thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
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


class HamerHandEstimator:
    def __init__(self, args: argparse.Namespace) -> None:
        hamer_root = args.hamer_root
        if str(hamer_root) not in sys.path:
            sys.path.insert(0, str(hamer_root))

        os.chdir(str(hamer_root))

        from hamer.configs import CACHE_DIR_HAMER
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
        from hamer.utils.renderer import Renderer, cam_crop_to_full
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        self.ViTDetDataset = ViTDetDataset
        self.recursive_to = recursive_to
        self.cam_crop_to_full = cam_crop_to_full

        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = Path(CACHE_DIR_HAMER) / "hamer_ckpts/checkpoints/hamer.ckpt"
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"HaMeR checkpoint not found: {checkpoint}")

        self.model, self.model_cfg = load_hamer(str(checkpoint))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        if args.body_detector == "vitdet":
            from detectron2.config import LazyConfig
            import hamer

            cfg_path = Path(hamer.__file__).parent / "configs/cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = (
                    args.person_score_threshold
                )
        else:
            from detectron2 import model_zoo

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
                trained=True,
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = (
                args.person_score_threshold
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = args.person_nms_threshold

        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.cpm = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self.keypoint_threshold = args.keypoint_threshold
        self.rescale_factor = args.rescale_factor
        self.batch_size = args.batch_size
        self.save_mesh = args.save_mesh

    def detect_hand_candidates(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        det_out = self.detector(image_bgr)
        image_rgb = image_bgr[:, :, ::-1]

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.0)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        if len(pred_bboxes) == 0:
            return []

        vitposes_out = self.cpm.predict_pose(
            image_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        candidates: List[Dict[str, Any]] = []
        for person_idx, vitposes in enumerate(vitposes_out):
            person_score = float(pred_scores[person_idx]) if person_idx < len(pred_scores) else None
            for handedness, is_right, keypoints in (
                ("left", 0, vitposes["keypoints"][-42:-21]),
                ("right", 1, vitposes["keypoints"][-21:]),
            ):
                valid = keypoints[:, 2] > self.keypoint_threshold
                if int(valid.sum()) <= 3:
                    continue

                hand_keypoints = keypoints[valid]
                keypoint_mean_conf = float(hand_keypoints[:, 2].mean())
                if person_score is None:
                    confidence = keypoint_mean_conf
                else:
                    confidence = 0.5 * keypoint_mean_conf + 0.5 * person_score
                bbox = [
                    float(hand_keypoints[:, 0].min()),
                    float(hand_keypoints[:, 1].min()),
                    float(hand_keypoints[:, 0].max()),
                    float(hand_keypoints[:, 1].max()),
                ]
                candidates.append(
                    {
                        "candidate_id": len(candidates),
                        "person_index": int(person_idx),
                        "handedness": handedness,
                        "is_right": int(is_right),
                        "bbox_xyxy": bbox,
                        "confidence": float(confidence),
                        "detector_score": person_score,
                        "keypoint_mean_conf": keypoint_mean_conf,
                        "valid_keypoints": int(valid.sum()),
                    }
                )
        return candidates

    def reconstruct_meshes(
        self,
        image_bgr: np.ndarray,
        candidates: List[Dict[str, Any]],
        mesh_dir: Path,
        frame_idx: int,
    ) -> None:
        if not candidates:
            return
        boxes = np.asarray([candidate["bbox_xyxy"] for candidate in candidates], dtype=np.float32)
        right = np.asarray([candidate["is_right"] for candidate in candidates], dtype=np.float32)
        dataset = self.ViTDetDataset(
            self.model_cfg,
            image_bgr,
            boxes,
            right,
            rescale_factor=self.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        for batch in dataloader:
            batch = self.recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * img_size.max()
            )
            pred_cam_t_full = self.cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()

            for n in range(batch["img"].shape[0]):
                person_id = int(batch["personid"][n])
                candidate = candidates[person_id]
                verts = out["pred_vertices"][n].detach().cpu().numpy()
                is_right = float(batch["right"][n].detach().cpu().numpy())
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                candidate["pred_cam_t_full"] = cam_t.tolist()
                candidate["_verts"] = verts
                candidate["_is_right_float"] = is_right
                candidate["_scaled_focal_length"] = float(scaled_focal_length)

                if self.save_mesh:
                    mesh_dir.mkdir(parents=True, exist_ok=True)
                    mesh_path = mesh_dir / f"frame_{frame_idx:06d}_candidate_{person_id:02d}.obj"
                    tmesh = self.renderer.vertices_to_trimesh(
                        verts,
                        cam_t.copy(),
                        LIGHT_BLUE,
                        is_right=is_right,
                    )
                    tmesh.export(str(mesh_path))
                    candidate["mesh_obj"] = str(mesh_path)

    def render_full_frame_overlay(
        self,
        image_bgr: np.ndarray,
        candidates: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        selected = [
            candidate
            for candidate in candidates
            if candidate.get("status")
            in {"ego_left", "ego_right", "ego_left_recovered", "ego_right_recovered"}
            and "_verts" in candidate
        ]
        if not selected:
            return None

        vertices = [candidate["_verts"] for candidate in selected]
        cam_t = [np.asarray(candidate["pred_cam_t_full"], dtype=np.float32) for candidate in selected]
        is_right = [candidate["_is_right_float"] for candidate in selected]
        focal_length = float(selected[0].get("_scaled_focal_length", self.renderer.focal_length))

        cam_view = self.renderer.render_rgba_multiple(
            vertices,
            cam_t=cam_t,
            render_res=[image_bgr.shape[1], image_bgr.shape[0]],
            is_right=is_right,
            focal_length=focal_length,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        input_rgb = image_bgr.astype(np.float32)[:, :, ::-1] / 255.0
        input_rgba = np.concatenate([input_rgb, np.ones_like(input_rgb[:, :, :1])], axis=2)
        overlay_rgb = input_rgba[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        return np.clip(255.0 * overlay_rgb[:, :, ::-1], 0, 255).astype(np.uint8)


def build_filter_config(args: argparse.Namespace) -> EgoHandFilterConfig:
    return EgoHandFilterConfig(
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        preferred_area_ratio=args.preferred_area_ratio,
        min_center_y_ratio=args.min_center_y_ratio,
        bottom_start_y_ratio=args.bottom_start_y_ratio,
        min_confidence=args.min_confidence,
        near_depth_m=args.near_depth_m,
        far_depth_m=args.far_depth_m,
        max_median_depth_m=args.max_median_depth_m,
        recovery_min_area_ratio=args.recovery_min_area_ratio,
        recovery_min_center_y_ratio=args.recovery_min_center_y_ratio,
        recovery_max_median_depth_m=args.recovery_max_median_depth_m,
    )


def strip_private_candidate_fields(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HaMeR on Ropedia frames and filter ego-hand candidates."
    )
    parser.add_argument("--video", default="/home/liuyou/Documents/ropedia_data/stereo_left.mp4")
    parser.add_argument("--depth_dir", default=None)
    parser.add_argument("--output_dir", default="tmp/ropedia_hamer_ego_filter")
    parser.add_argument("--hamer_root", default="/home/liuyou/Documents/hamer")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--body_detector", choices=["regnety", "vitdet"], default="regnety")
    parser.add_argument("--person_score_threshold", type=float, default=0.5)
    parser.add_argument("--person_nms_threshold", type=float, default=0.4)
    parser.add_argument("--keypoint_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument("--save_mesh", action="store_true")
    parser.add_argument("--min_area_ratio", type=float, default=0.002)
    parser.add_argument("--max_area_ratio", type=float, default=0.35)
    parser.add_argument("--preferred_area_ratio", type=float, default=0.035)
    parser.add_argument("--min_center_y_ratio", type=float, default=0.12)
    parser.add_argument("--bottom_start_y_ratio", type=float, default=0.35)
    parser.add_argument("--min_confidence", type=float, default=0.35)
    parser.add_argument("--near_depth_m", type=float, default=0.25)
    parser.add_argument("--far_depth_m", type=float, default=1.8)
    parser.add_argument("--max_median_depth_m", type=float, default=3.0)
    parser.add_argument("--recovery_min_area_ratio", type=float, default=0.001)
    parser.add_argument("--recovery_min_center_y_ratio", type=float, default=0.35)
    parser.add_argument("--recovery_max_median_depth_m", type=float, default=1.2)
    parser.add_argument("--contact_sheet_columns", type=int, default=4)
    parser.add_argument("--contact_sheet_thumb_width", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    original_cwd = Path.cwd()
    args = parse_args()
    args.video = resolve_path(args.video, original_cwd)
    args.output_dir = resolve_path(args.output_dir, original_cwd)
    args.hamer_root = resolve_path(args.hamer_root, original_cwd)
    args.depth_dir = resolve_path(args.depth_dir, original_cwd) if args.depth_dir else None
    args.checkpoint = resolve_path(args.checkpoint, original_cwd) if args.checkpoint else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    annotated_dir = args.output_dir / "annotated"
    hamer_overlay_dir = args.output_dir / "hamer_overlay"
    mesh_dir = args.output_dir / "meshes"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    hamer_overlay_dir.mkdir(parents=True, exist_ok=True)

    filter_config = build_filter_config(args)
    estimator = HamerHandEstimator(args)

    report: Dict[str, Any] = {
        "video": str(args.video),
        "depth_dir": str(args.depth_dir) if args.depth_dir else None,
        "output_dir": str(args.output_dir),
        "frame_range": {
            "start": args.start,
            "num_frames": args.num_frames,
            "stride": args.stride,
        },
        "filter_config": filter_config.__dict__,
        "frames": [],
    }

    annotated_paths: List[Path] = []
    hamer_overlay_paths: List[Path] = []
    for frame_idx, image_bgr in iter_video_frames(args.video, args.start, args.num_frames, args.stride):
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), image_bgr)

        depth_m, depth_path = load_depth(args.depth_dir, frame_idx)
        candidates = estimator.detect_hand_candidates(image_bgr)
        estimator.reconstruct_meshes(image_bgr, candidates, mesh_dir, frame_idx)

        selection = select_ego_hands(
            candidates,
            image_width=image_bgr.shape[1],
            image_height=image_bgr.shape[0],
            depth_m=depth_m,
            config=filter_config,
        )
        annotated = draw_candidates(image_bgr, selection.candidates)
        annotated_path = annotated_dir / f"frame_{frame_idx:06d}_ego_filter.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        annotated_paths.append(annotated_path)

        hamer_overlay_path = None
        hamer_overlay = estimator.render_full_frame_overlay(image_bgr, selection.candidates)
        if hamer_overlay is not None:
            hamer_overlay_path = hamer_overlay_dir / f"frame_{frame_idx:06d}_hamer_overlay.jpg"
            cv2.imwrite(str(hamer_overlay_path), hamer_overlay)
            hamer_overlay_paths.append(hamer_overlay_path)

        counts = {
            "ego_left": 0,
            "ego_right": 0,
            "ego_left_recovered": 0,
            "ego_right_recovered": 0,
            "other_hands": 0,
            "rejected": 0,
        }
        for candidate in selection.candidates:
            counts[candidate["status"]] = counts.get(candidate["status"], 0) + 1

        report["frames"].append(
            {
                "frame_idx": frame_idx,
                "image": str(frame_path),
                "depth": depth_path,
                "annotated_image": str(annotated_path),
                "hamer_overlay_image": str(hamer_overlay_path) if hamer_overlay_path else None,
                "selected": selection.selected,
                "counts": counts,
                "candidates": [strip_private_candidate_fields(candidate) for candidate in selection.candidates],
            }
        )
        print(
            f"frame {frame_idx:06d}: "
            f"candidates={len(selection.candidates)} "
            f"selected={selection.selected} "
            f"counts={counts}"
        )

    contact_sheet = args.output_dir / "contact_sheet_ego_filter.jpg"
    make_contact_sheet(
        annotated_paths,
        contact_sheet,
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )
    hamer_overlay_contact_sheet = args.output_dir / "contact_sheet_hamer_overlay.jpg"
    make_contact_sheet(
        hamer_overlay_paths,
        hamer_overlay_contact_sheet,
        columns=args.contact_sheet_columns,
        thumb_width=args.contact_sheet_thumb_width,
    )
    report["outputs"] = {
        "candidates_json": str(args.output_dir / "ego_hand_candidates.json"),
        "contact_sheet": str(contact_sheet),
        "hamer_overlay_contact_sheet": str(hamer_overlay_contact_sheet),
        "annotated_dir": str(annotated_dir),
        "hamer_overlay_dir": str(hamer_overlay_dir),
        "frames_dir": str(frames_dir),
        "meshes_dir": str(mesh_dir) if args.save_mesh else None,
    }

    with (args.output_dir / "ego_hand_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, ensure_ascii=False, indent=2)

    print(f"JSON: {args.output_dir / 'ego_hand_candidates.json'}")
    print(f"Contact sheet: {contact_sheet}")
    print(f"HaMeR overlay contact sheet: {hamer_overlay_contact_sheet}")


if __name__ == "__main__":
    main()
