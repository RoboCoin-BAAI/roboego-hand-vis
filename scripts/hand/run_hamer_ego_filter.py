#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.candidate_refine import CandidateRefineConfig, refine_candidates_file


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for idx, item in enumerate(sys.argv):
        if item == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if item.startswith(prefix):
            return item[len(prefix) :]
    return default


def _has_flag(name: str) -> bool:
    return name in sys.argv


def _strip_wrapper_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    value_args = {
        "--stage1_actions",
        "--refine_min_bottom_y_ratio",
        "--refine_min_area_ratio",
        "--refine_max_area_ratio",
        "--refine_max_center_jump_px",
        "--refine_max_area_change_ratio",
        "--refine_max_depth_jump_m",
        "--refine_max_mesh_size_change_ratio",
        "--stage1_hand_policy",
        "--refine_mode",
    }
    flag_args = {"--save_refined_vis", "--no_save_refined_vis", "--accept_uncertain_refined"}
    for idx, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in value_args:
            skip_next = True
            continue
        if any(item.startswith(f"{name}=") for name in value_args):
            continue
        if item in flag_args:
            continue
        stripped.append(item)
    return stripped


def _float_arg(name: str, default: float) -> float:
    value = _arg_value(name)
    if value is None:
        return default
    return float(value)


def _load_vlm_frame_priors(path: str | None) -> dict[float, dict]:
    if not path:
        return {}
    source = Path(path).expanduser()
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames") or []
    out: dict[float, list[float]] = {}
    if not isinstance(frames, list):
        return out
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        try:
            t = float(frame.get("t"))
        except (TypeError, ValueError):
            continue
        prior = {
            "t": t,
            "hand_visible": bool(frame.get("hand_visible", False)),
            "hands_visible": frame.get("hands_visible") if isinstance(frame.get("hands_visible"), dict) else {},
            "wearer_bbox_xyxy": None,
        }
        region = frame.get("wearer_region") or {}
        if isinstance(region, dict) and bool(region.get("visible", True)):
            bbox = region.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                try:
                    prior["wearer_bbox_xyxy"] = [float(value) for value in bbox[:4]]
                except (TypeError, ValueError):
                    prior["wearer_bbox_xyxy"] = None
        out[t] = prior
    return out


def _load_wearer_regions(path: str | None) -> dict[float, list[float]]:
    return {
        t: prior["wearer_bbox_xyxy"]
        for t, prior in _load_vlm_frame_priors(path).items()
        if prior.get("wearer_bbox_xyxy") is not None
    }


def _vlm_prior_for_frame(
    priors_by_second: dict[float, dict],
    *,
    frame_idx: int,
    fps: float,
    stage1_time_offset_s: float,
    max_delta_s: float = 1.5,
) -> dict | None:
    if not priors_by_second or fps <= 0:
        return None
    t = float(stage1_time_offset_s) + float(frame_idx) / float(fps)
    best_t = min(priors_by_second, key=lambda key: abs(key - t))
    if abs(best_t - t) > max_delta_s:
        return None
    return priors_by_second[best_t]


def _wearer_region_for_frame(
    regions_by_second: dict[float, list[float]],
    *,
    frame_idx: int,
    fps: float,
    stage1_time_offset_s: float,
    max_delta_s: float = 1.5,
) -> list[float] | None:
    if not regions_by_second or fps <= 0:
        return None
    t = float(stage1_time_offset_s) + float(frame_idx) / float(fps)
    best_t = min(regions_by_second, key=lambda key: abs(key - t))
    if abs(best_t - t) > max_delta_s:
        return None
    return regions_by_second[best_t]


def _stamp_ego_mode(output_dir: str | None, ego_mode: str) -> None:
    if not output_dir:
        return
    candidates_json = Path(output_dir).expanduser() / "ego_hand_candidates.json"
    if not candidates_json.exists():
        return
    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload["ego_mode"] = ego_mode
        metadata = payload.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["ego_mode"] = ego_mode
            metadata["ego_mode_note"] = (
                "strict keeps the legacy two-hand-oriented selection behavior; "
                "tolerant marks this run as accepting only_left/only_right frames for downstream stages."
            )
        candidates_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_path(value: str | None, base_dir: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _maybe_refine_and_visualize(output_dir: str | None, manifest: str | None, stage1_actions: str | None, ego_mode: str) -> None:
    if ego_mode != "tolerant" or not output_dir:
        return
    output_root = Path(output_dir).expanduser()
    status_path = output_root / "refine_status.json"
    candidates_json = output_root / "ego_hand_candidates.json"
    if not candidates_json.exists():
        matches = sorted(output_root.glob("*candidate*.json"))
        candidates_json = matches[0] if matches else candidates_json
    if not candidates_json.exists():
        status_path.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": "missing_candidates_json",
                    "searched": str(output_root / "ego_hand_candidates.json"),
                    "output_dir": str(output_root),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return
    refined_json = output_root / "ego_hand_candidates_refined.json"
    stage1_path = Path(stage1_actions).expanduser() if stage1_actions else None
    if stage1_path is not None and not stage1_path.exists():
        stage1_path = None
    refine_source_json = candidates_json
    overlay_filtered_json = output_root / "ego_hand_candidates_overlay_filtered.json"
    if manifest:
        if _write_overlay_filtered_candidates(
            manifest=Path(manifest).expanduser(),
            candidates_json=candidates_json,
            output_json=overlay_filtered_json,
        ):
            refine_source_json = overlay_filtered_json
    refine_candidates_file(
        candidates_json=refine_source_json,
        stage1_actions_json=stage1_path,
        output_json=refined_json,
        config=CandidateRefineConfig(
            refine_mode=_arg_value("--refine_mode", "strict_ego_audit") or "strict_ego_audit",
            fps=30,
            stage1_time_offset_s=_float_arg("--stage1_time_offset_s", 0.0),
            stage1_hand_policy=_arg_value("--stage1_hand_policy", "permissive") or "permissive",
            min_bottom_y_ratio=_float_arg("--refine_min_bottom_y_ratio", 0.42),
            min_area_ratio=_float_arg("--refine_min_area_ratio", 0.0007),
            max_area_ratio=_float_arg("--refine_max_area_ratio", 0.18),
            max_center_jump_px=_float_arg("--refine_max_center_jump_px", 260.0),
            max_area_change_ratio=_float_arg("--refine_max_area_change_ratio", 3.0),
            max_depth_jump_m=_float_arg("--refine_max_depth_jump_m", 0.35),
            max_mesh_size_change_ratio=_float_arg("--refine_max_mesh_size_change_ratio", 2.5),
            accept_uncertain=_has_flag("--accept_uncertain_refined"),
        ),
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "ego_mode": ego_mode,
                "candidates_json": str(candidates_json),
                "refine_source_json": str(refine_source_json),
                "refined_json": str(refined_json),
                "stage1_actions": str(stage1_path) if stage1_path else None,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    save_refined_vis = not _has_flag("--no_save_refined_vis")
    if save_refined_vis and manifest:
        _write_refined_visualization(
            manifest=Path(manifest).expanduser(),
            refined_json=refined_json,
            output_dir=output_root / "refined_annotated",
            contact_sheet_path=output_root / "refined_contact_sheet.jpg",
        )


def _write_refined_visualization(
    *,
    manifest: Path,
    refined_json: Path,
    output_dir: Path,
    contact_sheet_path: Path,
    max_sheet_frames: int = 24,
) -> None:
    payload = json.loads(refined_json.read_text(encoding="utf-8"))
    frames = payload.get("frames") or payload.get("results") or []
    if not isinstance(frames, list):
        return
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    video_path = Path(manifest_payload["videos"]["left"]["path"])
    if not video_path.is_absolute():
        video_path = manifest.parent / video_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    _prepare_refined_visualization_output_dir(output_dir, contact_sheet_path)
    sheet_images: list[np.ndarray] = []
    try:
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_idx = _frame_index(frame)
            overlay_path = _find_hamer_overlay(refined_json.parent, frame_idx)
            overlay_mask = None
            if overlay_path is not None:
                image = cv2.imread(str(overlay_path))
                if image is None:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, raw_image = cap.read()
                if ok:
                    overlay_mask = _hamer_overlay_mask(raw_image, image)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, image = cap.read()
                if not ok:
                    continue
            annotated = _draw_refined_frame(image, frame, overlay_mask=overlay_mask)
            out_path = output_dir / f"frame_{frame_idx:06d}_refined.jpg"
            cv2.imwrite(str(out_path), annotated)
            if len(sheet_images) < max_sheet_frames:
                sheet_images.append(_resize_for_sheet(annotated, width=360))
    finally:
        cap.release()
    if sheet_images:
        sheet = _make_contact_sheet(sheet_images, cols=4)
        cv2.imwrite(str(contact_sheet_path), sheet)


def _prepare_refined_visualization_output_dir(output_dir: Path, contact_sheet_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("frame_*_refined.jpg", "frame_*_refined.png"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    if contact_sheet_path.exists() and contact_sheet_path.is_file():
        contact_sheet_path.unlink()


def _write_overlay_filtered_candidates(*, manifest: Path, candidates_json: Path, output_json: Path) -> bool:
    payload = json.loads(candidates_json.read_text(encoding="utf-8"))
    frames = payload.get("frames") or payload.get("results") or []
    if not isinstance(frames, list):
        return False
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    video_path = Path(manifest_payload["videos"]["left"]["path"])
    if not video_path.is_absolute():
        video_path = manifest.parent / video_path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    filtered_frames: list[dict] = []
    try:
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            item = dict(frame)
            candidates = item.get("candidates")
            if not isinstance(candidates, list):
                filtered_frames.append(item)
                continue
            frame_idx = _frame_index(item)
            overlay_path = _find_hamer_overlay(candidates_json.parent, frame_idx)
            if overlay_path is None:
                filtered_frames.append(item)
                continue
            overlay_image = cv2.imread(str(overlay_path))
            if overlay_image is None:
                filtered_frames.append(item)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, raw_image = cap.read()
            if not ok:
                filtered_frames.append(item)
                continue
            overlay_mask = _hamer_overlay_mask(raw_image, overlay_image)
            kept: list[dict] = []
            rejected: list[dict] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if _bbox_overlaps_overlay_mask(_bbox_xyxy(candidate), overlay_mask, min_pixels=30):
                    kept.append(candidate)
                    continue
                dropped = dict(candidate)
                dropped["overlay_filter_reject_reason"] = "no_overlap_with_hamer_overlay_mask"
                rejected.append(dropped)
            item["candidates"] = kept
            item["overlay_filtered_rejected_candidates"] = rejected
            item["overlay_filter"] = {
                "mode": "hamer_overlay_absdiff_bbox_overlap_v1",
                "input_candidate_count": len(candidates),
                "kept_candidate_count": len(kept),
                "rejected_candidate_count": len(rejected),
            }
            filtered_frames.append(item)
    finally:
        cap.release()
    filtered_payload = dict(payload)
    if isinstance(filtered_payload.get("frames"), list):
        filtered_payload["frames"] = filtered_frames
    else:
        filtered_payload["results"] = filtered_frames
    filtered_payload["overlay_filter"] = {
        "mode": "hamer_overlay_absdiff_bbox_overlap_v1",
        "source_candidates_json": str(candidates_json),
    }
    output_json.write_text(json.dumps(filtered_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def _draw_refined_frame(image: np.ndarray, frame: dict, *, overlay_mask: np.ndarray | None = None) -> np.ndarray:
    out = image.copy()
    candidates = frame.get("refined_candidates")
    if not isinstance(candidates, list):
        candidates = []
    visible_candidates: list[dict] = []
    selected_ids = set((frame.get("refine") or {}).get("selected_candidate_ids", {}).values())
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        bbox = _bbox_xyxy(candidate)
        if overlay_mask is not None and not _bbox_overlaps_overlay_mask(bbox, overlay_mask):
            continue
        visible_candidates.append(candidate)
    visible_candidates = _deduplicate_visible_candidates(visible_candidates)
    for candidate in visible_candidates:
        role = str(candidate.get("refine_role", "unknown"))
        quality = str(candidate.get("refine_quality", ""))
        score = candidate.get("ego_prior_score", "")
        selected = candidate.get("candidate_id", candidate.get("id")) in selected_ids
        color = (40, 255, 40) if selected else _role_color(role)
        thickness = 5 if selected else 2
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label_prefix = "SELECT " if selected else ""
        label = f"{label_prefix}{candidate.get('candidate_id', '?')} {role} {quality} {score}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        risks = candidate.get("refine_risks") or []
        if risks:
            cv2.putText(
                out,
                ",".join(str(risk) for risk in risks[:2]),
                (x1, min(out.shape[0] - 8, y2 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    state = (frame.get("refine") or {}).get("visibility_state", "")
    cv2.putText(
        out,
        f"HaMeR overlay + refined selection: {state}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    return out


def _deduplicate_visible_candidates(candidates: list[dict]) -> list[dict]:
    ordered = sorted(candidates, key=_candidate_draw_priority, reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        if any(_is_duplicate_candidate(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: int(item.get("candidate_id") or item.get("id") or 0))


def _candidate_draw_priority(candidate: dict) -> tuple[float, float, float, float]:
    role = str(candidate.get("refine_role", ""))
    quality = str(candidate.get("refine_quality", ""))
    score = _safe_float(candidate.get("ego_prior_score"), 0.0)
    status = str(candidate.get("status", ""))
    role_bonus = 3.0 if role in {"ego_left", "ego_right"} else 0.0
    if role == "uncertain":
        role_bonus = 1.0
    if status in {"ego_left", "ego_right", "ego_left_recovered", "ego_right_recovered"}:
        role_bonus += 0.5
    quality_bonus = 1.0 if quality == "normal" else 0.0
    return (role_bonus, quality_bonus, score, _bbox_area(_bbox_xyxy(candidate)))


def _is_duplicate_candidate(a: dict, b: dict, *, iou_threshold: float = 0.35, center_threshold_px: float = 80.0) -> bool:
    bbox_a = _bbox_xyxy(a)
    bbox_b = _bbox_xyxy(b)
    if _bbox_iou(bbox_a, bbox_b) >= iou_threshold:
        return True
    return _bbox_center_distance(bbox_a, bbox_b) <= center_threshold_px


def _hamer_overlay_mask(raw_bgr: np.ndarray, overlay_bgr: np.ndarray) -> np.ndarray:
    if raw_bgr.shape[:2] != overlay_bgr.shape[:2]:
        raw_bgr = cv2.resize(raw_bgr, (overlay_bgr.shape[1], overlay_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(raw_bgr, overlay_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = gray > 12
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return mask_u8 > 0


def _bbox_overlaps_overlay_mask(bbox: list[float], overlay_mask: np.ndarray, *, min_pixels: int = 30) -> bool:
    h, w = overlay_mask.shape[:2]
    x1 = max(0, min(w, int(round(bbox[0]))))
    y1 = max(0, min(h, int(round(bbox[1]))))
    x2 = max(0, min(w, int(round(bbox[2]))))
    y2 = max(0, min(h, int(round(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return False
    return int(overlay_mask[y1:y2, x1:x2].sum()) >= min_pixels


def _find_hamer_overlay(output_root: Path, frame_idx: int) -> Path | None:
    overlay_dir = output_root / "hamer_overlay"
    if not overlay_dir.exists():
        return None
    patterns = [
        f"frame_{frame_idx:06d}*.jpg",
        f"*{frame_idx:06d}*.jpg",
        f"frame_{frame_idx:06d}*.png",
        f"*{frame_idx:06d}*.png",
    ]
    for pattern in patterns:
        matches = sorted(overlay_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _frame_index(frame: dict) -> int:
    for key in ("frame_idx", "frame_index", "frame_id", "frame", "index", "idx"):
        value = frame.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for key in ("image_path", "frame_path", "rgb_path", "annotated_path"):
        value = str(frame.get(key, ""))
        import re

        match = re.search(r"frame[_-]?(\d+)", value)
        if match:
            return int(match.group(1))
    return 0


def _bbox_xyxy(candidate: dict) -> list[float]:
    bbox = candidate.get("bbox_xyxy") or candidate.get("bbox")
    if isinstance(bbox, dict):
        if {"x1", "y1", "x2", "y2"} <= set(bbox):
            return [float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])]
        if {"x", "y", "w", "h"} <= set(bbox):
            x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            return [x, y, x + w, y + h]
    if isinstance(bbox, list) and len(bbox) >= 4:
        values = [float(v) for v in bbox[:4]]
        if values[2] <= values[0] or values[3] <= values[1]:
            return [values[0], values[1], values[0] + values[2], values[1] + values[3]]
        return values
    return [0.0, 0.0, 0.0, 0.0]


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = _bbox_area([x1, y1, x2, y2])
    union = _bbox_area(a) + _bbox_area(b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_center_distance(a: list[float], b: list[float]) -> float:
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _role_color(role: str) -> tuple[int, int, int]:
    if role in {"ego_left", "ego_right"}:
        return (60, 220, 60)
    if role == "uncertain":
        return (0, 220, 255)
    if role in {"other", "other_hand"}:
        return (180, 180, 180)
    if role == "rejected":
        return (40, 40, 230)
    return (255, 180, 0)


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
        r, c = divmod(idx, cols)
        sheet[r * h : r * h + img.shape[0], c * w : c * w + img.shape[1]] = img
    return sheet


def main() -> None:
    from scripts.run_ropedia_hamer_ego_filter import main as hamer_filter_impl_main

    original_cwd = Path.cwd()
    ego_mode = _arg_value("--ego_mode", "strict") or "strict"
    output_dir = _resolve_path(_arg_value("--output_dir"), original_cwd)
    manifest = _resolve_path(_arg_value("--manifest"), original_cwd)
    stage1_actions = _resolve_path(_arg_value("--stage1_actions"), original_cwd)
    original_argv = sys.argv[:]
    sys.argv = _strip_wrapper_args(sys.argv)
    hamer_filter_impl_main()
    sys.argv = original_argv
    _stamp_ego_mode(output_dir, ego_mode)
    _maybe_refine_and_visualize(output_dir, manifest, stage1_actions, ego_mode)


if __name__ == "__main__":
    main()
