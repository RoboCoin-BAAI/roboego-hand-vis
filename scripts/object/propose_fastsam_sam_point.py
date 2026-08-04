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

from roboego_hand_vis.objects.fastsam_prompt import (
    FastSamPromptConfig,
    SamPointPrompt,
    choose_sam_point_from_candidates,
)
from roboego_hand_vis.objects.manipulated_tracker import MaskCandidate
from roboego_hand_vis.objects.semantic_qwen import call_qwen_object_candidate, call_qwen_object_target


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_frame_joints(hand_joints_npz: Path, frame_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    with np.load(hand_joints_npz) as data:
        left_valid = bool(data["left_valid"][frame_idx])
        right_valid = bool(data["right_valid"][frame_idx])
        left = data["left_joints_21_camera_opencv"][frame_idx].astype(np.float32) if left_valid else None
        right = data["right_joints_21_camera_opencv"][frame_idx].astype(np.float32) if right_valid else None
    return left, right


def fastsam_candidates_from_rgb(
    image_rgb: np.ndarray,
    *,
    frame_idx: int,
    model_path: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> list[MaskCandidate]:
    try:
        from ultralytics import FastSAM
    except ImportError as exc:
        raise RuntimeError("FastSAM requires ultralytics. Install it with: pip install ultralytics") from exc

    model = FastSAM(model_path)
    kwargs: dict[str, Any] = {
        "source": cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "retina_masks": True,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device
    results = model.predict(**kwargs)
    if not results:
        return []
    result = results[0]
    if result.masks is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else None
    candidates: list[MaskCandidate] = []
    for idx, mask_arr in enumerate(masks):
        mask = (mask_arr > 0.5).astype(np.uint8) * 255
        bbox = _bbox_from_mask(mask)
        if bbox is None and boxes is not None:
            x1, y1, x2, y2 = boxes[idx]
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if bbox is None:
            continue
        candidates.append(
            MaskCandidate(
                frame_idx=frame_idx,
                candidate_id=f"fastsam_{idx}",
                bbox_xywh=bbox,
                mask_area_px=int((mask > 0).sum()),
                mask=mask,
                source="fastsam",
            )
        )
    return candidates


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def save_candidate_crop(image_rgb: np.ndarray, candidate: MaskCandidate, output_dir: Path) -> Path:
    x, y, w, h = [int(v) for v in candidate.bbox_xywh]
    height, width = image_rgb.shape[:2]
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_path = output_dir / f"{candidate.candidate_id}.png"
    save_rgb_image(crop_path, crop)
    return crop_path


def choose_prompt_with_optional_qwen(
    *,
    image_rgb: np.ndarray,
    candidates: list[MaskCandidate],
    left_joints: np.ndarray | None,
    right_joints: np.ndarray | None,
    intrinsics: dict[str, float],
    config: FastSamPromptConfig,
    use_qwen: bool,
    qwen_model: str,
    output_dir: Path,
    top_k: int,
) -> tuple[SamPointPrompt | None, dict[str, Any] | None, list[dict[str, Any]]]:
    base_prompt = choose_sam_point_from_candidates(
        candidates=candidates,
        left_joints=left_joints,
        right_joints=right_joints,
        intrinsics=intrinsics,
        config=config,
    )
    if not use_qwen or base_prompt is None:
        return base_prompt, None, []

    frame_path = output_dir / "qwen_frame.png"
    save_rgb_image(frame_path, image_rgb)
    target = call_qwen_object_target(frame_path=frame_path, model=qwen_model)
    if target.get("status") != "ok":
        return base_prompt, target, []

    scored_prompts: list[tuple[float, SamPointPrompt, dict[str, Any]]] = []
    ranked = sorted(candidates, key=lambda cand: cand.mask_area_px, reverse=True)[:top_k]
    crops_dir = output_dir / "qwen_candidate_crops"
    for cand in ranked:
        prompt = choose_sam_point_from_candidates(
            candidates=[cand],
            left_joints=left_joints,
            right_joints=right_joints,
            intrinsics=intrinsics,
            config=config,
        )
        if prompt is None:
            continue
        crop_path = save_candidate_crop(image_rgb, cand, crops_dir)
        qwen_score = call_qwen_object_candidate(crop_path=crop_path, target=target, model=qwen_model)
        if qwen_score.get("status") != "ok":
            continue
        final_score = qwen_candidate_rank_score(prompt, qwen_score)
        if final_score is None:
            continue
        scored_prompts.append((final_score, prompt, qwen_score))

    if not scored_prompts:
        return base_prompt, target, []
    scored_prompts.sort(key=lambda item: item[0], reverse=True)
    best_score, best_prompt, best_qwen = scored_prompts[0]
    best_prompt.score = float(best_score)
    best_prompt.source = f"{best_prompt.source}:qwen_reranked"
    return best_prompt, target, [dict(item[2], candidate_id=item[1].candidate_id, final_score=item[0]) for item in scored_prompts]


def qwen_candidate_rank_score(prompt: SamPointPrompt, qwen_score: dict[str, Any]) -> float | None:
    if qwen_score.get("status") != "ok":
        return None
    if not qwen_score.get("is_whole_object"):
        return None
    if not qwen_score.get("matches_target"):
        return None
    if qwen_score.get("is_part_or_pattern"):
        return None
    semantic_score = float(qwen_score["score"]) * 1000.0
    area_tiebreak = min(float(prompt.mask_area_px), 50000.0) * 0.0001
    geometry_tiebreak = float(prompt.score) * 0.01
    return semantic_score + area_tiebreak + geometry_tiebreak


def write_prompt_payload(
    prompt: SamPointPrompt,
    output_path: Path,
    *,
    qwen_target: dict[str, Any] | None = None,
    qwen_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "frame_idx": prompt.frame_idx,
        "candidate_id": prompt.candidate_id,
        "point_xy": prompt.point_xy,
        "point_labels": prompt.point_labels,
        "bbox_xywh": prompt.bbox_xywh,
        "mask_area_px": prompt.mask_area_px,
        "score": prompt.score,
        "source": prompt.source,
        "sam3_points_arg": ",".join(str(v) for v in prompt.point_xy),
        "sam3_point_labels_arg": ";".join(str(v) for v in prompt.point_labels),
        "sam3_box_arg": _bbox_xywh_to_xyxy_arg(prompt.bbox_xywh),
    }
    if qwen_target is not None:
        payload["qwen_target"] = qwen_target
    if qwen_candidates is not None:
        payload["qwen_candidates"] = qwen_candidates
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _bbox_xywh_to_xyxy_arg(bbox_xywh: list[int]) -> str:
    x, y, w, h = [int(v) for v in bbox_xywh]
    return f"{x},{y},{x + w},{y + h}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose a SAM point from first-frame FastSAM masks near hands.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame_idx", type=int, required=True)
    parser.add_argument("--hand_joints_npz", required=True)
    parser.add_argument("--intrinsics_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--fastsam_model", default="FastSAM-x.pt")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="")
    parser.add_argument("--max_hand_bbox_distance_px", type=float, default=100.0)
    parser.add_argument("--max_hand_points_inside_ratio", type=float, default=0.35)
    parser.add_argument("--use_qwen", action="store_true")
    parser.add_argument("--qwen_model", default="qwen-vl-plus")
    parser.add_argument("--qwen_top_k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intrinsics_data = json.loads(Path(args.intrinsics_json).read_text())
    intrinsics = intrinsics_data.get("left", intrinsics_data)
    image_rgb = read_video_frame(Path(args.video), args.frame_idx)
    left_joints, right_joints = load_frame_joints(Path(args.hand_joints_npz), args.frame_idx)
    candidates = fastsam_candidates_from_rgb(
        image_rgb,
        frame_idx=args.frame_idx,
        model_path=args.fastsam_model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
    )
    output_path = Path(args.output_json)
    prompt, qwen_target, qwen_candidates = choose_prompt_with_optional_qwen(
        image_rgb=image_rgb,
        candidates=candidates,
        left_joints=left_joints,
        right_joints=right_joints,
        intrinsics=intrinsics,
        config=FastSamPromptConfig(
            max_hand_bbox_distance_px=args.max_hand_bbox_distance_px,
            max_hand_points_inside_ratio=args.max_hand_points_inside_ratio,
        ),
        use_qwen=args.use_qwen,
        qwen_model=args.qwen_model,
        output_dir=output_path.parent,
        top_k=args.qwen_top_k,
    )
    if prompt is None:
        raise RuntimeError(f"No FastSAM candidate near hands on frame {args.frame_idx}")
    payload = write_prompt_payload(
        prompt,
        output_path,
        qwen_target=qwen_target,
        qwen_candidates=qwen_candidates,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
