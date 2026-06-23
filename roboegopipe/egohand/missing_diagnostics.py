from __future__ import annotations

from typing import Any, Dict, List


PRESENT_STATUSES = {
    "ego_left",
    "ego_right",
    "ego_left_recovered",
    "ego_right_recovered",
}


def canonical_side(status: str) -> str | None:
    if status in {"ego_left", "ego_left_recovered"}:
        return "ego_left"
    if status in {"ego_right", "ego_right_recovered"}:
        return "ego_right"
    return None


def missing_sides_from_frame(frame: Dict[str, Any]) -> List[str]:
    selected = frame.get("selected") or {}
    present = {
        side
        for side in ("ego_left", "ego_right")
        if selected.get(side) is not None
    }

    for candidate in frame.get("candidates", []):
        side = canonical_side(str(candidate.get("status", "")))
        if side is not None:
            present.add(side)

    return [side for side in ("ego_left", "ego_right") if side not in present]


def format_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "na"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "na"


def candidate_label(candidate: Dict[str, Any]) -> str:
    candidate_id = candidate.get("candidate_id", candidate.get("id", "?"))
    status = candidate.get("status", "?")
    handedness = candidate.get("handedness", "?")
    conf = format_float(candidate.get("confidence"), 2)
    score = format_float(candidate.get("ego_score"), 2)
    depth = format_float(candidate.get("median_depth_m"), 2, "m")
    area = format_float(candidate.get("bbox_area_ratio"), 4)
    reasons = candidate.get("reject_reasons") or []
    reason_text = "ok" if not reasons else ",".join(str(reason) for reason in reasons)
    return (
        f"id={candidate_id} {status} {handedness} "
        f"conf={conf} score={score} depth={depth} area={area} {reason_text}"
    )


def summarize_missing_frames(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing_left = []
    missing_right = []
    missing_any = []
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        missing = missing_sides_from_frame(frame)
        if missing:
            missing_any.append(frame_idx)
        if "ego_left" in missing:
            missing_left.append(frame_idx)
        if "ego_right" in missing:
            missing_right.append(frame_idx)
    return {
        "missing_left": missing_left,
        "missing_right": missing_right,
        "missing_any": missing_any,
        "missing_left_count": len(missing_left),
        "missing_right_count": len(missing_right),
        "missing_any_count": len(missing_any),
    }
