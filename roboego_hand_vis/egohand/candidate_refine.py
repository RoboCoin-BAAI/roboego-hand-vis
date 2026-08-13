from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateRefineConfig:
    refine_mode: str = "legacy_score"
    fps: int = 30
    stage1_time_offset_s: float = 0.0
    stage1_hand_policy: str = "permissive"
    min_bottom_y_ratio: float = 0.42
    min_area_ratio: float = 0.0007
    max_area_ratio: float = 0.18
    min_center_x_ratio: float = 0.16
    max_center_x_ratio: float = 0.84
    side_edge_penalty: float = 2.0
    max_center_jump_px: float = 260.0
    max_area_change_ratio: float = 3.0
    max_depth_jump_m: float = 0.35
    max_mesh_size_change_ratio: float = 2.5
    accept_uncertain: bool = False


def refine_candidates_file(
    *,
    candidates_json: Path,
    stage1_actions_json: Path | None,
    output_json: Path,
    config: CandidateRefineConfig = CandidateRefineConfig(),
) -> dict[str, Any]:
    payload = json.loads(Path(candidates_json).read_text(encoding="utf-8"))
    stage1 = json.loads(Path(stage1_actions_json).read_text(encoding="utf-8")) if stage1_actions_json else {}
    refined = refine_candidates_payload(payload, stage1, config=config)
    refined["source_candidates_json"] = str(candidates_json)
    if stage1_actions_json:
        refined["source_stage1_actions_json"] = str(stage1_actions_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")
    return refined


def refine_candidates_payload(
    payload: dict[str, Any],
    stage1_actions_payload: dict[str, Any],
    *,
    config: CandidateRefineConfig = CandidateRefineConfig(),
) -> dict[str, Any]:
    if config.refine_mode == "strict_ego_audit":
        return _refine_candidates_payload_strict_ego_audit(payload, stage1_actions_payload, config=config)
    return _refine_candidates_payload_legacy(payload, stage1_actions_payload, config=config)


def _refine_candidates_payload_legacy(
    payload: dict[str, Any],
    stage1_actions_payload: dict[str, Any],
    *,
    config: CandidateRefineConfig,
) -> dict[str, Any]:
    out = dict(payload)
    frames = _frame_list(out)
    stage1_by_t = _stage1_by_second(stage1_actions_payload)
    previous_by_side: dict[str, dict[str, Any]] = {}
    refined_frames: list[dict[str, Any]] = []
    for frame in frames:
        item = dict(frame)
        frame_idx = _frame_index(item)
        stage1_t = int(config.stage1_time_offset_s + frame_idx / max(1, int(config.fps)))
        stage1_frame = stage1_by_t.get(stage1_t, {})
        candidates = _candidate_list(item)
        stage1_sides = _stage1_wearer_sides(stage1_frame)
        stage1_named_side_has_candidate = {
            side: _has_credible_candidate_for_side(candidates, item, side, config=config)
            for side in stage1_sides
        }
        scored = [
            _score_candidate(
                candidate,
                item,
                stage1_frame,
                previous_by_side,
                stage1_named_side_has_candidate=stage1_named_side_has_candidate,
                config=config,
            )
            for candidate in candidates
        ]
        selected = _select_by_side(scored, config=config)
        item["refine"] = {
            "mode": "semantic_candidate_refine_rules_v1",
            "visibility_state": _visibility_state(selected),
            "stage1_t": stage1_frame.get("t"),
            "stage1_activity": stage1_frame.get("activity"),
            "selected_candidate_ids": {
                side: candidate.get("candidate_id")
                for side, candidate in selected.items()
                if candidate is not None
            },
        }
        item["refined_candidates"] = scored
        item["ego_left_refined"] = selected.get("left")
        item["ego_right_refined"] = selected.get("right")
        for side, candidate in selected.items():
            if candidate is not None:
                previous_by_side[side] = candidate
        refined_frames.append(item)
    _set_frame_list(out, refined_frames)
    out["candidate_refine"] = {
        "mode": "semantic_candidate_refine_rules_v1",
        "config": config.__dict__,
        "notes": [
            "Rules are conservative: bottom/size/depth/temporal continuity increase confidence.",
            "Unstable candidates are marked uncertain/rejected instead of being forced into current hands.",
            "This stage does not optimize MANO; it only refines candidate selection labels.",
        ],
    }
    return out


def _refine_candidates_payload_strict_ego_audit(
    payload: dict[str, Any],
    stage1_actions_payload: dict[str, Any],
    *,
    config: CandidateRefineConfig,
) -> dict[str, Any]:
    out = dict(payload)
    frames = _frame_list(out)
    stage1_by_t = _stage1_by_second(stage1_actions_payload)
    refined_frames: list[dict[str, Any]] = []
    for frame in frames:
        item = dict(frame)
        frame_idx = _frame_index(item)
        stage1_t = int(config.stage1_time_offset_s + frame_idx / max(1, int(config.fps)))
        stage1_frame = stage1_by_t.get(stage1_t, {})
        expected_sides = _stage1_wearer_sides(stage1_frame)
        candidates = _candidate_list(item)
        audited = [_audit_candidate_strict(candidate, item, expected_sides, config=config) for candidate in candidates]
        selected = _select_reliable_strict(audited)
        missing_hands = [
            {"side": side, "reason": "stage1_expected_but_no_reliable_candidate"}
            for side in ("left", "right")
            if side in expected_sides and selected.get(side) is None
        ]
        item["refine"] = {
            "mode": "strict_ego_audit_v1",
            "visibility_state": _visibility_state(selected),
            "stage1_t": stage1_frame.get("t"),
            "stage1_activity": stage1_frame.get("activity"),
            "stage1_hands_visible": stage1_frame.get("hands_visible"),
            "selected_candidate_ids": {
                side: candidate.get("candidate_id")
                for side, candidate in selected.items()
                if candidate is not None
            },
        }
        item["refined_candidates"] = audited
        item["ego_left_refined"] = selected.get("left")
        item["ego_right_refined"] = selected.get("right")
        item["ego_left_uncertain"] = _best_strict_candidate(audited, "left", role="uncertain")
        item["ego_right_uncertain"] = _best_strict_candidate(audited, "right", role="uncertain")
        item["other_hands"] = [candidate for candidate in audited if candidate.get("refine_role") == "other_hand"]
        item["rejected_hands"] = [candidate for candidate in audited if candidate.get("refine_role") == "rejected"]
        item["missing_hands"] = missing_hands
        refined_frames.append(item)
    _set_frame_list(out, refined_frames)
    out["candidate_refine"] = {
        "mode": "strict_ego_audit_v1",
        "config": config.__dict__,
        "notes": [
            "Only overlay-filtered HaMeR candidates should be passed into this stage.",
            "HaMeR anatomical left/right is used as the candidate side; image position does not flip side labels.",
            "Stage1 hands_visible constrains which wearer hands are expected; unexpected sides are marked other/uncertain.",
            "This stage audits candidates and marks missing/occluded hands; it does not force hand completion.",
        ],
    }
    return out


def _audit_candidate_strict(
    candidate: dict[str, Any],
    frame: dict[str, Any],
    expected_sides: set[str],
    *,
    config: CandidateRefineConfig,
) -> dict[str, Any]:
    item = dict(candidate)
    item["candidate_id"] = item.get("candidate_id", item.get("id"))
    side = _candidate_side(item)
    bbox = _bbox_xyxy(item)
    width, height = _image_size(frame, item)
    center = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, width, height)
    center_x_ratio = center[0] / max(1.0, width)
    bottom_y_ratio = bbox[3] / max(1.0, height)
    reasons: list[str] = []
    risks: list[str] = []
    score = 0.0

    if side not in {"left", "right"}:
        reasons.append("unknown_hamer_anatomical_side")
        role = "rejected"
        quality = "low_confidence"
        score -= 2.0
    else:
        role = f"ego_{side}"
        quality = "reliable"

    if expected_sides and side in {"left", "right"} and side not in expected_sides:
        reasons.append("semantic_side_not_visible")
        role = "other_hand"
        quality = "semantic_mismatch"
        score -= 2.0
    elif expected_sides and side in expected_sides:
        reasons.append("semantic_expected_side")
        score += 2.0

    if bottom_y_ratio >= config.min_bottom_y_ratio:
        reasons.append("from_wearer_lower_body_region")
        score += 2.0
    else:
        reasons.append("not_from_wearer_lower_body")
        role = "other_hand" if role != "rejected" else role
        quality = "not_ego_prior"
        score -= 2.0

    if center_x_ratio < config.min_center_x_ratio or center_x_ratio > config.max_center_x_ratio:
        reasons.append("enters_from_side_edge")
        role = "other_hand" if role != "rejected" else role
        quality = "not_ego_prior"
        score -= 1.5

    if config.min_area_ratio <= area_ratio <= config.max_area_ratio:
        reasons.append("reasonable_bbox_area")
        score += 1.0
    else:
        risks.append("bbox_area_out_of_range")
        if role.startswith("ego_"):
            role = "uncertain"
            quality = "uncertain"
        score -= 0.5

    depth = _candidate_depth(item)
    if depth is not None:
        reasons.append("has_depth_anchor")
        score += 0.5

    if role.startswith("ego_") and quality == "reliable" and score < 3.0:
        role = "uncertain"
        quality = "uncertain"
        risks.append("weak_ego_prior_score")

    item["refine_role"] = role
    item["refine_quality"] = quality
    item["refine_side"] = side
    item["ego_prior_score"] = round(score, 4)
    item["refine_reasons"] = reasons
    item["refine_risks"] = risks
    item["bbox_area_ratio"] = area_ratio
    item["bbox_center_xy"] = [center[0], center[1]]
    item["bbox_bottom_y_ratio"] = bottom_y_ratio
    return item


def _select_reliable_strict(audited: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        eligible = [
            candidate
            for candidate in audited
            if candidate.get("refine_role") == f"ego_{side}" and candidate.get("refine_quality") == "reliable"
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda item: float(item.get("ego_prior_score", 0.0)), reverse=True)
        selected[side] = eligible[0]
    return selected


def _best_strict_candidate(audited: list[dict[str, Any]], side: str, *, role: str) -> dict[str, Any] | None:
    eligible = [
        candidate
        for candidate in audited
        if candidate.get("refine_side") == side and candidate.get("refine_role") == role
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda item: float(item.get("ego_prior_score", 0.0)), reverse=True)
    return eligible[0]


def _frame_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("frames", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _set_frame_list(payload: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    if isinstance(payload.get("frames"), list):
        payload["frames"] = frames
    elif isinstance(payload.get("results"), list):
        payload["results"] = frames
    else:
        payload["frames"] = frames


def _candidate_list(frame: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidates", "hands", "all_candidates"):
        value = frame.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for side_key in ("ego_left", "ego_right"):
        value = frame.get(side_key)
        if isinstance(value, dict):
            item = dict(value)
            item.setdefault("source_role", side_key)
            out.append(item)
    return out


def _stage1_by_second(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []) or []:
        if isinstance(frame, dict) and "t" in frame:
            out[int(float(frame["t"]))] = frame
    return out


def _frame_index(frame: dict[str, Any]) -> int:
    for key in ("frame_idx", "frame_index", "frame_id", "frame", "index", "idx"):
        value = frame.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for key in ("image_path", "frame_path", "rgb_path", "annotated_path"):
        value = str(frame.get(key, ""))
        match = __import__("re").search(r"frame[_-]?(\d+)", value)
        if match:
            return int(match.group(1))
    return 0


def _score_candidate(
    candidate: dict[str, Any],
    frame: dict[str, Any],
    stage1_frame: dict[str, Any],
    previous_by_side: dict[str, dict[str, Any]],
    *,
    stage1_named_side_has_candidate: dict[str, bool],
    config: CandidateRefineConfig,
) -> dict[str, Any]:
    item = dict(candidate)
    bbox = _bbox_xyxy(item)
    width, height = _image_size(frame, item)
    center = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, width, height)
    side = _candidate_side(item)
    stage1_sides = _stage1_wearer_sides(stage1_frame)
    stage1_side_override = False
    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []

    near_bottom = bbox[3] / max(1.0, height) >= config.min_bottom_y_ratio
    if near_bottom:
        score += 2.0
        reasons.append("near_bottom_or_lower_frame")
    else:
        risks.append("too_high_in_image")

    if config.min_area_ratio <= area_ratio <= config.max_area_ratio:
        score += 1.5
        reasons.append("reasonable_bbox_area")
    else:
        risks.append("bbox_area_out_of_range")

    center_x_ratio = center[0] / max(1.0, width)
    near_side_edge = center_x_ratio < config.min_center_x_ratio or center_x_ratio > config.max_center_x_ratio
    if near_side_edge:
        score -= config.side_edge_penalty
        risks.append("near_side_edge")

    if len(stage1_sides) == 1 and near_bottom and not near_side_edge:
        stage1_side = next(iter(stage1_sides))
        should_override = config.stage1_hand_policy == "conservative"
        if config.stage1_hand_policy == "permissive":
            should_override = side == "unknown" or not stage1_named_side_has_candidate.get(stage1_side, False)
        if side != stage1_side and should_override:
            side = stage1_side
            stage1_side_override = True
            reasons.append("stage1_single_wearer_hand_side_override")

    depth = _candidate_depth(item)
    if depth is not None:
        score += 0.5
        reasons.append("has_depth_anchor")

    previous = previous_by_side.get(side)
    if previous is not None:
        prev_center = _bbox_center(_bbox_xyxy(previous))
        center_jump = ((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) ** 0.5
        if center_jump <= config.max_center_jump_px:
            score += 2.0
            reasons.append("temporal_center_continuity")
        elif stage1_side_override:
            score -= 1.0
            risks.append("stage1_overrode_large_center_jump")
        else:
            score -= 3.0
            risks.append("large_center_jump")
        prev_area = max(_bbox_area_ratio(_bbox_xyxy(previous), width, height), 1e-9)
        area_change = max(area_ratio / prev_area, prev_area / max(area_ratio, 1e-9))
        if area_change > config.max_area_change_ratio:
            score -= 2.0
            risks.append("large_bbox_area_change")
        mesh_size = _mesh_size(item)
        prev_mesh_size = _mesh_size(previous)
        if mesh_size is not None and prev_mesh_size is not None:
            mesh_change = max(mesh_size / max(prev_mesh_size, 1e-9), prev_mesh_size / max(mesh_size, 1e-9))
            if mesh_change > config.max_mesh_size_change_ratio:
                score -= 2.0
                risks.append("large_mesh_size_change")
        prev_depth = _candidate_depth(previous)
        if depth is not None and prev_depth is not None and abs(depth - prev_depth) > config.max_depth_jump_m:
            score -= 2.0
            risks.append("large_depth_jump")

    if _stage1_mentions_wearer_hand(stage1_frame, side):
        score += 1.0
        reasons.append("stage1_mentions_wearer_hand")

    quality = "normal"
    role = f"ego_{side}" if side in {"left", "right"} else "uncertain"
    if "large_center_jump" in risks or "large_bbox_area_change" in risks or "large_mesh_size_change" in risks:
        quality = "unstable"
    if "too_high_in_image" in risks and score < 2.5:
        role = "other"
    if "near_side_edge" in risks and "temporal_center_continuity" not in reasons:
        role = "other" if score < 3.5 else "uncertain"
    if score < 1.5:
        role = "rejected"
        quality = "low_confidence"
    elif quality != "normal" and not config.accept_uncertain:
        role = "uncertain"

    item["candidate_id"] = item.get("candidate_id", item.get("id"))
    item["refine_role"] = role
    item["refine_quality"] = quality
    item["refine_side"] = side
    item["ego_prior_score"] = round(score, 4)
    item["refine_reasons"] = reasons
    item["refine_risks"] = risks
    item["bbox_area_ratio"] = area_ratio
    item["bbox_center_xy"] = [center[0], center[1]]
    item["mesh_size_proxy"] = _mesh_size(item)
    return item


def _select_by_side(scored: list[dict[str, Any]], *, config: CandidateRefineConfig) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        eligible = [
            candidate
            for candidate in scored
            if _candidate_side(candidate) == side
            and candidate.get("refine_role") == f"ego_{side}"
            and (candidate.get("refine_quality") == "normal" or config.accept_uncertain)
        ]
        if not eligible:
            continue
        eligible.sort(key=lambda item: float(item.get("ego_prior_score", 0.0)), reverse=True)
        selected[side] = eligible[0]
    return selected


def _visibility_state(selected: dict[str, dict[str, Any] | None]) -> str:
    left = selected.get("left") is not None
    right = selected.get("right") is not None
    if left and right:
        return "both_hands"
    if left:
        return "only_left"
    if right:
        return "only_right"
    return "none"


def _bbox_xyxy(candidate: dict[str, Any]) -> list[float]:
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


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _bbox_area_ratio(bbox: list[float], width: float, height: float) -> float:
    area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
    return float(area / max(1.0, width * height))


def _image_size(frame: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float]:
    for source in (candidate, frame):
        width = source.get("image_width") or source.get("width")
        height = source.get("image_height") or source.get("height")
        if width and height:
            return float(width), float(height)
    return 1920.0, 1074.0


def _candidate_side(candidate: dict[str, Any]) -> str:
    value = str(
        candidate.get("refine_side")
        or candidate.get("side")
        or candidate.get("hand_side")
        or candidate.get("label")
        or candidate.get("status")
        or ""
    ).lower()
    if "left" in value:
        return "left"
    if "right" in value:
        return "right"
    source_role = str(candidate.get("source_role", "")).lower()
    if "left" in source_role:
        return "left"
    if "right" in source_role:
        return "right"
    return "unknown"


def _candidate_depth(candidate: dict[str, Any]) -> float | None:
    for key in ("median_depth_m", "depth_median", "target_z_median", "target_depth_m", "z_median"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _has_credible_candidate_for_side(
    candidates: list[dict[str, Any]],
    frame: dict[str, Any],
    side: str,
    *,
    config: CandidateRefineConfig,
) -> bool:
    for candidate in candidates:
        if _candidate_side(candidate) != side:
            continue
        bbox = _bbox_xyxy(candidate)
        width, height = _image_size(frame, candidate)
        center = _bbox_center(bbox)
        area_ratio = _bbox_area_ratio(bbox, width, height)
        center_x_ratio = center[0] / max(1.0, width)
        near_bottom = bbox[3] / max(1.0, height) >= config.min_bottom_y_ratio
        near_side_edge = center_x_ratio < config.min_center_x_ratio or center_x_ratio > config.max_center_x_ratio
        if near_bottom and not near_side_edge and config.min_area_ratio <= area_ratio <= config.max_area_ratio:
            return True
    return False


def _mesh_size(candidate: dict[str, Any]) -> float | None:
    for key in ("mesh_bbox_diag", "mesh_size", "mesh_extent", "mano_mesh_size", "vertices_diag"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    bbox = _bbox_xyxy(candidate)
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    if width <= 0 or height <= 0:
        return None
    return (width * width + height * height) ** 0.5


def _stage1_mentions_wearer_hand(frame: dict[str, Any], side: str) -> bool:
    if not frame:
        return False
    hands = frame.get("hands_visible") or {}
    if side in hands and bool(hands[side]):
        return True
    text = f"{frame.get('activity', '')} {frame.get('notes', '')}".lower()
    return "wearer" in text and side in text and "hand" in text


def _stage1_wearer_sides(frame: dict[str, Any]) -> set[str]:
    sides: set[str] = set()
    if not frame:
        return sides
    hands = frame.get("hands_visible") or {}
    for side in ("left", "right"):
        if bool(hands.get(side)):
            sides.add(side)
    text = f"{frame.get('activity', '')} {frame.get('notes', '')}".lower()
    for side in ("left", "right"):
        if "wearer" in text and side in text and "hand" in text:
            sides.add(side)
    return sides
