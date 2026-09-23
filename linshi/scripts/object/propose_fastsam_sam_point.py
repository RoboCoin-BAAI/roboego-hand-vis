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


def save_candidate_debug_artifacts(
    *,
    image_rgb: np.ndarray,
    candidates: list[MaskCandidate],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        crop_path = save_candidate_crop_with_name(image_rgb, cand, output_dir, f"{cand.candidate_id}_crop.png")
        mask_name = f"{cand.candidate_id}_mask.png"
        overlay_name = f"{cand.candidate_id}_overlay.png"
        cv2.imwrite(str(output_dir / mask_name), cand.mask)
        overlay = draw_candidate_overlay(image_rgb, cand)
        save_rgb_image(output_dir / overlay_name, overlay)
        rows.append(
            {
                "candidate_id": cand.candidate_id,
                "frame_idx": int(cand.frame_idx),
                "bbox_xywh": [int(v) for v in cand.bbox_xywh],
                "mask_area_px": int(cand.mask_area_px),
                "source": cand.source,
                "crop_path": crop_path.name,
                "mask_path": mask_name,
                "overlay_path": overlay_name,
            }
        )
    (output_dir / "candidates.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def save_candidate_crop_with_name(
    image_rgb: np.ndarray,
    candidate: MaskCandidate,
    output_dir: Path,
    filename: str,
) -> Path:
    x, y, w, h = [int(v) for v in candidate.bbox_xywh]
    height, width = image_rgb.shape[:2]
    pad = 8
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    crop = image_rgb[y1:y2, x1:x2].copy()
    crop_path = output_dir / filename
    save_rgb_image(crop_path, crop)
    return crop_path


def draw_candidate_overlay(image_rgb: np.ndarray, candidate: MaskCandidate) -> np.ndarray:
    overlay = image_rgb.copy()
    mask_bool = candidate.mask > 0
    color = np.array([255, 210, 0], dtype=np.uint8)
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * color).astype(np.uint8)
    x, y, w, h = [int(v) for v in candidate.bbox_xywh]
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 40, 40), 3)
    cv2.putText(
        overlay,
        candidate.candidate_id,
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return overlay


def compute_yellow_mask_score(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask_bool = mask > 0
    mask_pixels = int(mask_bool.sum())
    if mask_pixels == 0:
        return {"yellow_ratio": 0.0, "yellow_pixels": 0, "mask_pixels": 0}
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([15, 55, 60], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    yellow = cv2.inRange(hsv, lower, upper) > 0
    yellow_pixels = int((yellow & mask_bool).sum())
    return {
        "yellow_ratio": float(yellow_pixels / mask_pixels),
        "yellow_pixels": yellow_pixels,
        "mask_pixels": mask_pixels,
    }


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
    semantic_pool_size: int,
    multi_object_top_k: int,
) -> tuple[SamPointPrompt | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    base_prompt = choose_sam_point_from_candidates(
        candidates=candidates,
        left_joints=left_joints,
        right_joints=right_joints,
        intrinsics=intrinsics,
        config=config,
    )
    if not use_qwen or base_prompt is None:
        return base_prompt, None, [], []

    frame_path = output_dir / "qwen_frame.png"
    save_rgb_image(frame_path, image_rgb)
    target = call_qwen_object_target(frame_path=frame_path, model=qwen_model)
    if target.get("status") != "ok":
        return base_prompt, target, [], []

    candidate_prompts: list[tuple[SamPointPrompt, dict[str, Any]]] = []
    for cand in candidates:
        prompt = choose_qwen_prompt_for_candidate(
            cand,
            left_joints=left_joints,
            right_joints=right_joints,
            intrinsics=intrinsics,
            strict_config=config,
        )
        if prompt is not None:
            yellow_score = compute_yellow_mask_score(image_rgb, cand.mask)
            candidate_prompts.append((prompt, yellow_score))
    ranked = build_qwen_candidate_prompt_pool(candidate_prompts, top_k=top_k)

    qwen_scored_prompts: list[tuple[SamPointPrompt, dict[str, Any]]] = []
    crops_dir = output_dir / "qwen_candidate_crops"
    candidates_by_id = {cand.candidate_id: cand for cand in candidates}
    for prompt, yellow_score in ranked:
        cand = candidates_by_id[prompt.candidate_id]
        crop_path = save_candidate_crop(image_rgb, cand, crops_dir)
        qwen_score = call_qwen_object_candidate(crop_path=crop_path, target=target, model=qwen_model)
        if qwen_score.get("status") != "ok":
            continue
        qwen_score["crop_path"] = str(crop_path)
        qwen_score.update(yellow_score)
        if not qwen_candidate_is_semantic_match(qwen_score):
            continue
        qwen_scored_prompts.append((prompt, qwen_score))

    selected = select_prompt_from_qwen_semantic_pool(
        qwen_scored_prompts,
        semantic_pool_size=semantic_pool_size,
    )
    if selected is None:
        return base_prompt, target, [], []
    best_prompt, best_qwen = selected
    semantic_pool = build_qwen_semantic_pool(qwen_scored_prompts, semantic_pool_size=semantic_pool_size)
    semantic_rows = build_qwen_semantic_pool_rows(
        semantic_pool,
        selected_candidate_id=best_prompt.candidate_id,
    )
    multi_objects = [
        prompt_to_sam3_object(prompt, row, object_index=idx)
        for idx, ((prompt, _qwen_score), row) in enumerate(zip(semantic_pool[:multi_object_top_k], semantic_rows))
    ]
    best_score = qwen_candidate_rank_score(best_prompt, best_qwen) or best_prompt.score
    best_prompt.score = float(best_score)
    best_prompt.source = f"{best_prompt.source}:qwen_reranked"
    return best_prompt, target, semantic_rows, multi_objects


def qwen_candidate_is_semantic_match(qwen_score: dict[str, Any]) -> bool:
    return (
        qwen_score.get("status") == "ok"
        and bool(qwen_score.get("is_whole_object"))
        and bool(qwen_score.get("matches_target"))
        and not bool(qwen_score.get("is_part_or_pattern"))
    )


def choose_qwen_prompt_for_candidate(
    candidate: MaskCandidate,
    *,
    left_joints: np.ndarray | None,
    right_joints: np.ndarray | None,
    intrinsics: dict[str, float],
    strict_config: FastSamPromptConfig | None = None,
) -> SamPointPrompt | None:
    prompt = choose_sam_point_from_candidates(
        candidates=[candidate],
        left_joints=left_joints,
        right_joints=right_joints,
        intrinsics=intrinsics,
        config=strict_config,
    )
    if prompt is not None:
        return prompt
    relaxed = FastSamPromptConfig(
        max_hand_bbox_distance_px=1200.0,
        max_hand_points_inside_ratio=1.0,
        max_mask_frame_area_ratio=0.35,
    )
    prompt = choose_sam_point_from_candidates(
        candidates=[candidate],
        left_joints=left_joints,
        right_joints=right_joints,
        intrinsics=intrinsics,
        config=relaxed,
    )
    if prompt is not None:
        prompt.source = f"{prompt.source}:qwen_relaxed_hand_gate"
    return prompt


def build_qwen_candidate_prompt_pool(
    candidate_prompts: list[tuple[SamPointPrompt, dict[str, Any]]],
    *,
    top_k: int,
) -> list[tuple[SamPointPrompt, dict[str, Any]]]:
    if top_k <= 0:
        return []
    ranked_groups = [
        sorted(candidate_prompts, key=lambda item: float(item[0].score), reverse=True),
        sorted(candidate_prompts, key=lambda item: float(item[1].get("yellow_ratio", 0.0)), reverse=True),
        sorted(candidate_prompts, key=lambda item: int(item[1].get("yellow_pixels", 0)), reverse=True),
    ]
    selected: list[tuple[SamPointPrompt, dict[str, Any]]] = []
    seen: set[str] = set()
    group_idx = 0
    while len(selected) < top_k and any(ranked_groups):
        group = ranked_groups[group_idx % len(ranked_groups)]
        while group:
            item = group.pop(0)
            candidate_id = item[0].candidate_id
            if candidate_id not in seen:
                selected.append(item)
                seen.add(candidate_id)
                break
        group_idx += 1
        if len(seen) >= len(candidate_prompts):
            break
    return selected


def build_qwen_semantic_pool(
    scored_prompts: list[tuple[SamPointPrompt, dict[str, Any]]],
    *,
    semantic_pool_size: int,
) -> list[tuple[SamPointPrompt, dict[str, Any]]]:
    return sorted(
        scored_prompts,
        key=lambda item: (
            float(item[1].get("score", 0.0)),
            float(item[1].get("yellow_ratio", 0.0)),
            int(item[1].get("yellow_pixels", 0)),
        ),
        reverse=True,
    )[:semantic_pool_size]


def select_prompt_from_qwen_semantic_pool(
    scored_prompts: list[tuple[SamPointPrompt, dict[str, Any]]],
    *,
    semantic_pool_size: int,
) -> tuple[SamPointPrompt, dict[str, Any]] | None:
    semantic_pool = build_qwen_semantic_pool(scored_prompts, semantic_pool_size=semantic_pool_size)
    if not semantic_pool:
        return None
    return semantic_pool[0]


def build_qwen_semantic_pool_rows(
    semantic_pool: list[tuple[SamPointPrompt, dict[str, Any]]],
    *,
    selected_candidate_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt, qwen_score in semantic_pool:
        rows.append(
            dict(
                qwen_score,
                candidate_id=prompt.candidate_id,
                hand_geometry_score=float(prompt.score),
                final_score=qwen_candidate_rank_score(prompt, qwen_score),
                selected_by_semantic_rank=prompt.candidate_id == selected_candidate_id,
                selected_by_hand_geometry=prompt.candidate_id == selected_candidate_id,
                sam3_points_arg=",".join(str(v) for v in prompt.point_xy),
                sam3_point_labels_arg=";".join(str(v) for v in prompt.point_labels),
                sam3_box_arg=_bbox_xywh_to_xyxy_arg(prompt.bbox_xywh),
                point_xy=prompt.point_xy,
                bbox_xywh=prompt.bbox_xywh,
                mask_area_px=prompt.mask_area_px,
            )
        )
    return rows


def prompt_to_sam3_object(prompt: SamPointPrompt, row: dict[str, Any], *, object_index: int) -> dict[str, Any]:
    return {
        "obj_id": f"manipulated_object_{object_index}",
        "candidate_id": prompt.candidate_id,
        "frame_idx": prompt.frame_idx,
        "point_xy": prompt.point_xy,
        "point_labels": prompt.point_labels,
        "bbox_xywh": prompt.bbox_xywh,
        "mask_area_px": prompt.mask_area_px,
        "score": prompt.score,
        "sam3_points_arg": ",".join(str(v) for v in prompt.point_xy),
        "sam3_point_labels_arg": ";".join(str(v) for v in prompt.point_labels),
        "sam3_box_arg": _bbox_xywh_to_xyxy_arg(prompt.bbox_xywh),
        "qwen_score": row.get("score"),
        "yellow_ratio": row.get("yellow_ratio"),
        "yellow_pixels": row.get("yellow_pixels"),
        "reason": row.get("reason"),
        "crop_path": row.get("crop_path"),
    }


def write_multi_object_prompt_payload(objects: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    payload = {"objects": objects}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def write_qwen_ranked_candidates(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_path = output_dir / "qwen_ranked_candidates.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def qwen_candidate_rank_score(prompt: SamPointPrompt, qwen_score: dict[str, Any]) -> float | None:
    if not qwen_candidate_is_semantic_match(qwen_score):
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
    parser.add_argument("--qwen_top_k", type=int, default=30)
    parser.add_argument("--qwen_semantic_pool_size", type=int, default=3)
    parser.add_argument("--multi_object_top_k", type=int, default=3)
    parser.add_argument("--save_all_candidates", action="store_true")
    parser.add_argument("--candidates_output_dir", default="")
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
    if args.save_all_candidates:
        candidates_dir = (
            Path(args.candidates_output_dir)
            if args.candidates_output_dir
            else Path(args.output_json).parent / f"fastsam_candidates_frame{args.frame_idx:06d}"
        )
        save_candidate_debug_artifacts(
            image_rgb=image_rgb,
            candidates=candidates,
            output_dir=candidates_dir,
        )
    output_path = Path(args.output_json)
    prompt, qwen_target, qwen_candidates, multi_objects = choose_prompt_with_optional_qwen(
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
        semantic_pool_size=args.qwen_semantic_pool_size,
        multi_object_top_k=args.multi_object_top_k,
    )
    if prompt is None:
        raise RuntimeError(f"No FastSAM candidate near hands on frame {args.frame_idx}")
    payload = write_prompt_payload(
        prompt,
        output_path,
        qwen_target=qwen_target,
        qwen_candidates=qwen_candidates,
    )
    if qwen_candidates:
        write_qwen_ranked_candidates(qwen_candidates, output_path.parent)
    if multi_objects:
        write_multi_object_prompt_payload(
            multi_objects,
            output_path.parent / "sam3_multi_object_prompts.json",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
