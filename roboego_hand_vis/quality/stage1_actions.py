from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ACTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("press button", [r"\bpress(?:es|ing|ed)?\s+(?:a\s+|the\s+)?button\b"]),
    ("pull out", [r"\bpull(?:s|ing|ed)?\s+out\b"]),
    ("pull apart", [r"\bpull(?:s|ing|ed)?\s+apart\b"]),
    ("push together", [r"\bpush(?:es|ing|ed)?\s+together\b"]),
    ("take out", [r"\btak(?:e|es|ing|en)\s+out\b", r"\bremov(?:e|es|ing|ed)\b"]),
    ("put into", [r"\bput(?:s|ting)?\s+(?:it\s+|them\s+|.+?\s+)?(?:in|into)\b", r"\bplac(?:e|es|ing|ed)\s+.+?\s+(?:in|into)\b"]),
    ("take from", [r"\btak(?:e|es|ing|en)\s+.+?\s+from\b"]),
    ("put back", [r"\bput(?:s|ting)?\s+.+?\s+back\b", r"\breturn(?:s|ing|ed)?\s+.+?\b"]),
    ("pick", [r"\bpick(?:s|ing|ed)?\s+up\b", r"\bpicks?\b", r"\btak(?:e|es|ing|en)\b(?!\s+(?:out|from)\b)"]),
    ("hold", [r"\bhold(?:s|ing)?\b", r"\bheld\b"]),
    ("carry", [r"\bcarry(?:|ing|ies|ied)\b"]),
    ("support", [r"\bsupport(?:s|ing|ed)?\b", r"\bstabiliz(?:e|es|ing|ed)\b"]),
    ("touch", [r"\btouch(?:es|ing|ed)?\b"]),
    ("reach", [r"\breach(?:es|ing|ed)?\s+(?:toward|for|to)\b", r"\bmov(?:e|es|ing|ed)\s+(?:the\s+)?(?:left|right)?\s*hand\s+(?:toward|for|to)\b"]),
    ("point", [r"\bpoint(?:s|ing|ed)?\s+(?:at|to|toward)\b"]),
    ("scan", [r"\bscan(?:s|ning|ned)?\b", r"\bswip(?:e|es|ing|ed)\b[^,.;&]{0,64}\b(?:scanner|barcode)\b"]),
    ("grasp", [r"\bgrasp(?:s|ing|ed)?\b", r"\bgrab(?:s|bing|bed)?\b"]),
    ("release", [r"\breleas(?:e|es|ing|ed)\b", r"\blet(?:s|ting)?\s+go\b"]),
    ("place", [r"\bplac(?:e|es|ing|ed)\b", r"\bput(?:s|ting)?\s+down\b", r"\bset(?:s|ting)?\s+(?:it\s+|them\s+|.+?\s+)?down\b"]),
    ("press", [r"\bpress(?:es|ing|ed)?\b", r"\btap(?:s|ping|ped)?\b"]),
    ("rotate", [r"\brotat(?:e|es|ing|ed)\b", r"\bturn(?:s|ing|ed)?\b", r"\btwist(?:s|ing|ed)?\b"]),
    ("push", [r"\bpush(?:es|ing|ed)?\b"]),
    ("pull", [r"\bpull(?:s|ing|ed)?\b"]),
    ("flip", [r"\bflip(?:s|ping|ped)?\b"]),
    ("cut", [r"\bcut(?:s|ting)?\b"]),
    ("insert", [r"\binsert(?:s|ing|ed)?\b", r"\bplug(?:s|ging|ged)?\s+in\b"]),
    ("operate", [r"\boperat(?:e|es|ing|ed)\b", r"\binteract(?:s|ing|ed)?\s+with\b", r"\bus(?:e|es|ing|ed)\b", r"\badjust(?:s|ing|ed)?\b", r"\bmanipulat(?:e|es|ing|ed)\b", r"\bhandl(?:e|es|ing|ed)\b"]),
    ("gesture", [r"\bgestur(?:e|es|ing|ed)\b", r"\bwav(?:e|es|ing|ed)\b"]),
    ("stir", [r"\bstir(?:s|ring|red)?\b"]),
    ("wipe", [r"\bwip(?:e|es|ing|ed)\b"]),
    ("fold", [r"\bfold(?:s|ing|ed)?\b"]),
    ("open", [r"\bopen(?:s|ing|ed)?\b"]),
    ("close", [r"\bclos(?:e|es|ing|ed)\b", r"\bshut(?:s|ting)?\b"]),
    ("align", [r"\balign(?:s|ing|ed)?\b"]),
    ("pour", [r"\bpour(?:s|ing|ed)?\b"]),
    ("measure", [r"\bmeasur(?:e|es|ing|ed)\b"]),
    ("sort", [r"\bsort(?:s|ing|ed)?\b"]),
    ("stack", [r"\bstack(?:s|ing|ed)?\b"]),
    ("load", [r"\bload(?:s|ing|ed)?\b"]),
    ("unload", [r"\bunload(?:s|ing|ed)?\b"]),
    ("lower", [r"\blower(?:s|ing|ed)?\b"]),
    ("lift", [r"\blift(?:s|ing|ed)?\b", r"\brais(?:e|es|ing|ed)\b"]),
    ("pass", [r"\bpass(?:es|ing|ed)?\b"]),
    ("handover", [r"\bhand(?:s|ing|ed)?\s+over\b", r"\bhandover\b"]),
    ("unfold", [r"\bunfold(?:s|ing|ed)?\b"]),
    ("uncap", [r"\buncap(?:s|ping|ped)?\b", r"\bopen(?:s|ing|ed)?\s+(?:a\s+|the\s+)?cap\b"]),
    ("pinch", [r"\bpinch(?:es|ing|ed)?\b"]),
    ("clip", [r"\bclip(?:s|ping|ped)?\b", r"\bclamp(?:s|ing|ed)?\b"]),
    ("move", [r"\bmov(?:e|es|ing|ed)\s+(?:a\s+|the\s+)?(?:object|item|product|box|cart|tool|device)\b"]),
]


OBJECT_ALIASES = {
    "phone": ["phone", "mobile phone"],
    "payment_terminal": ["payment terminal", "terminal", "pos", "pos terminal", "scanner"],
    "checkout_terminal": ["checkout terminal", "terminal"],
    "shopping cart": ["shopping cart", "cart", "basket"],
    "product": ["product", "goods", "bottle", "package", "carton"],
    "terminal": ["terminal", "payment terminal", "pos terminal"],
    "box": ["box", "carton", "package"],
    "bag": ["bag"],
    "container": ["container", "bin", "box"],
    "button": ["button"],
    "door": ["door"],
    "shelf": ["shelf", "shelves"],
}

GENERIC_OBJECT_TERMS = {"device", "handheld device", "object", "something", "item", "thing"}


def extract_frame_actions(frame: dict[str, Any]) -> list[dict[str, Any]]:
    activity = str(frame.get("activity", "")).lower()
    actions = _extract_actions(activity)
    if not actions:
        return []

    attrs = _object_attrs(frame, actions=actions)
    results: list[dict[str, Any]] = []
    for action in actions:
        bound = _bind_action_to_object(action, activity, attrs)
        results.append(bound)
    return _deduplicate_actions(results)


def extract_stage1_actions(payload: dict[str, Any]) -> dict[str, Any]:
    frames = payload.get("frames", [])
    output_frames: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_out = dict(frame)
        frame_out["frame_actions"] = extract_frame_actions(frame)
        output_frames.append(frame_out)
    output_frames = smooth_temporal_action_bindings(output_frames)
    action_frame_count = sum(1 for frame in output_frames if frame.get("frame_actions"))
    return {
        "source_video": payload.get("source_video", ""),
        "stage": "stage1_action_extraction",
        "frame_count": len(output_frames),
        "action_frame_count": action_frame_count,
        "frames": output_frames,
    }


def smooth_temporal_action_bindings(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    smoothed = [_copy_frame_with_actions(frame) for frame in frames]
    for idx, frame in enumerate(smoothed):
        for action in frame.get("frame_actions", []):
            if action.get("object_name") != "unknown":
                continue
            side = action.get("hand_side")
            if side not in {"left", "right"}:
                continue
            previous_action = _nearest_temporal_bound_action(smoothed, idx, -1, action["action"], side)
            next_action = _nearest_temporal_bound_action(smoothed, idx, 1, action["action"], side)
            if previous_action is None or next_action is None:
                continue
            if previous_action["object_name"] != next_action["object_name"]:
                continue
            action["object_name"] = previous_action["object_name"]
            action["source"] = "temporal_carryover"
            action["confidence"] = 0.7
            action["supporting_frames"] = [
                _format_timestamp(previous_action["t"]),
                _format_timestamp(next_action["t"]),
            ]
    return smoothed


def write_stage1_action_output(*, input_json: Path, output_json: Path) -> dict[str, Any]:
    payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    output = extract_stage1_actions(payload)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _copy_frame_with_actions(frame: dict[str, Any]) -> dict[str, Any]:
    out = dict(frame)
    out["frame_actions"] = [dict(action) for action in frame.get("frame_actions", [])]
    return out


def _nearest_temporal_bound_action(
    frames: list[dict[str, Any]],
    idx: int,
    direction: int,
    action_name: str,
    hand_side: str,
    *,
    max_gap: int = 2,
) -> dict[str, Any] | None:
    origin_t = float(frames[idx].get("t", idx))
    for step in range(1, max_gap + 1):
        neighbor_idx = idx + direction * step
        if neighbor_idx < 0 or neighbor_idx >= len(frames):
            break
        neighbor = frames[neighbor_idx]
        neighbor_t = float(neighbor.get("t", neighbor_idx))
        if abs(neighbor_t - origin_t) > max_gap:
            break
        for candidate in neighbor.get("frame_actions", []):
            if candidate.get("action") != action_name:
                continue
            if candidate.get("hand_side") != hand_side:
                continue
            if candidate.get("object_name") in {None, "", "unknown"}:
                continue
            return {"t": neighbor.get("t", neighbor_idx), **candidate}
    return None


def _format_timestamp(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _extract_actions(activity: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for action, patterns in ACTION_PATTERNS:
        positions = [match.start() for pattern in patterns if (match := re.search(pattern, activity))]
        if positions:
            matches.append((min(positions), action))
    return [action for _, action in sorted(matches, key=lambda item: item[0])]


def _object_attrs(frame: dict[str, Any], *, actions: list[str]) -> list[dict[str, Any]]:
    attrs = []
    for attr in frame.get("interaction_object_attributes", []):
        if not isinstance(attr, dict):
            continue
        name = str(attr.get("name", "")).strip()
        if not name:
            continue
        attrs.append(
            {
                "name": name,
                "hand_side": attr.get("hand_side", "unknown"),
                "confidence": _as_confidence(attr.get("confidence", 0.8)),
            }
        )
    if attrs:
        return attrs
    return _fallback_task_object_attrs(frame, actions=actions)


def _fallback_task_object_attrs(frame: dict[str, Any], *, actions: list[str]) -> list[dict[str, Any]]:
    names: list[str] = []
    for field in ["attention_objects", "task_objects"]:
        for value in frame.get(field, []):
            name = str(value).strip()
            if name and name.lower() not in GENERIC_OBJECT_TERMS and name not in names:
                names.append(name)
    if not names:
        return []
    visible_side = _single_visible_hand_side(frame)
    use_visible_side = visible_side if any(action in {"hold", "carry", "support", "touch"} for action in actions) else "unknown"
    return [
        {
            "name": name,
            "hand_side": use_visible_side if _task_object_can_use_visible_side(name, actions) else "unknown",
            "confidence": 0.6,
            "weak_source": "task_object_match",
        }
        for name in names
    ]


def _single_visible_hand_side(frame: dict[str, Any]) -> str:
    hands = frame.get("hands_visible")
    if not isinstance(hands, dict):
        return "unknown"
    left = hands.get("left") is True
    right = hands.get("right") is True
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return "unknown"


def _task_object_can_use_visible_side(name: str, actions: list[str]) -> bool:
    if not any(action in {"hold", "carry", "support", "touch"} for action in actions):
        return False
    return name.lower() not in {"shopping cart", "cart", "basket"}


def _bind_action_to_object(action: str, activity: str, attrs: list[dict[str, Any]]) -> dict[str, Any]:
    if action == "gesture":
        side = _hand_side_near_action(action, activity)
        return {
            "action": action,
            "object_name": "unknown",
            "hand_side": side if side in {"left", "right"} else "unknown",
            "source": "unbound",
            "confidence": 0.45,
        }
    matched = _rank_activity_object_matches(action, activity, attrs)
    if len(matched) == 1 or (len(matched) > 1 and matched[0][0] < matched[1][0]):
        return _action_binding(action, matched[0][1], source="activity_object_match", confidence=0.9)
    if len(attrs) == 1:
        return _action_binding(action, attrs[0], source="single_interaction_object", confidence=0.8)

    side = _hand_side_near_action(action, activity)
    if side in {"left", "right"}:
        side_matches = [attr for attr in attrs if attr.get("hand_side") == side]
        if len(side_matches) == 1:
            return _action_binding(action, side_matches[0], source="hand_side_match", confidence=0.75)

    type_match = _object_type_prior(action, attrs)
    if type_match is not None:
        return _action_binding(action, type_match, source="object_type_prior", confidence=0.65)

    return {
        "action": action,
        "object_name": "unknown",
        "hand_side": side if side in {"left", "right"} else "unknown",
        "source": "unbound",
        "confidence": 0.45,
    }


def _action_binding(action: str, attr: dict[str, Any], *, source: str, confidence: float) -> dict[str, Any]:
    source = str(attr.get("weak_source", source))
    return {
        "action": action,
        "object_name": str(attr["name"]),
        "hand_side": str(attr.get("hand_side", "unknown")),
        "source": source,
        "confidence": round(min(confidence, float(attr.get("confidence", confidence))), 3),
    }


def _object_name_in_text(name: str, text: str) -> bool:
    candidates = [name.lower(), *OBJECT_ALIASES.get(name.lower(), [])]
    normalized = text.lower().replace("_", " ")
    return any(re.search(rf"\b{re.escape(candidate.replace('_', ' '))}\b", normalized) for candidate in candidates)


def _action_object_match(action: str, name: str, activity: str) -> bool:
    return _action_object_distance(action, name, activity) is not None


def _rank_activity_object_matches(
    action: str,
    activity: str,
    attrs: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    directional_ranked = []
    for attr in attrs:
        object_terms = _object_terms(str(attr["name"]))
        distance = _directional_action_object_distance(action, object_terms, activity.lower().replace("_", " "))
        if distance is not None:
            directional_ranked.append((distance, attr))
    if directional_ranked:
        return sorted(directional_ranked, key=lambda item: item[0])

    ranked = []
    for attr in attrs:
        distance = _action_object_distance(action, str(attr["name"]), activity)
        if distance is not None:
            ranked.append((distance, attr))
    return sorted(ranked, key=lambda item: item[0])


def _action_object_distance(action: str, name: str, activity: str) -> int | None:
    normalized = activity.lower().replace("_", " ")
    action_terms = _action_terms(action)
    object_terms = _object_terms(name)
    directional = _directional_action_object_distance(action, object_terms, normalized)
    if directional is not None:
        return directional
    best: int | None = None
    for action_term in action_terms:
        action_matches = list(re.finditer(rf"\b{re.escape(action_term)}\w*\b", normalized))
        if not action_matches:
            continue
        for object_term in object_terms:
            object_matches = list(re.finditer(rf"\b{re.escape(object_term)}\b", normalized))
            for action_match in action_matches:
                for object_match in object_matches:
                    if not _same_clause(action_match, object_match, normalized):
                        continue
                    distance = abs(object_match.start() - action_match.start())
                    if distance <= 48 and (best is None or distance < best):
                        best = distance
    return best


def _same_clause(first: re.Match[str], second: re.Match[str], text: str) -> bool:
    left, right = sorted([first, second], key=lambda match: match.start())
    between = text[left.end() : right.start()]
    return re.search(r"[,;]|\b(?:and|then|while)\b", between) is None


def _object_terms(name: str) -> list[str]:
    object_terms = [name.lower(), *OBJECT_ALIASES.get(name.lower(), [])]
    return [term.replace("_", " ") for term in object_terms if term.replace("_", " ") not in GENERIC_OBJECT_TERMS]


def _directional_action_object_distance(action: str, object_terms: list[str], activity: str) -> int | None:
    prepositions = {
        "put into": ["in", "into"],
        "take from": ["from"],
        "put back": ["back", "to", "onto", "on"],
    }.get(action)
    if action == "operate":
        return _operate_action_object_distance(object_terms, activity)
    if prepositions is None:
        return None
    best: int | None = None
    for prep in prepositions:
        for object_term in object_terms:
            for match in re.finditer(rf"\b{re.escape(prep)}\b[^,.;&]{{0,36}}\b{re.escape(object_term)}\b", activity):
                distance = match.end() - match.start()
                if best is None or distance < best:
                    best = distance
    return best


def _operate_action_object_distance(object_terms: list[str], activity: str) -> int | None:
    best: int | None = None
    for object_term in object_terms:
        patterns = [
            rf"\bus(?:e|es|ing|ed)\b[^,.;&]{{0,64}}\bto\b[^,.;&]{{0,32}}\b{re.escape(object_term)}\b",
            rf"\boperat(?:e|es|ing|ed)\b[^,.;&]{{0,48}}\b{re.escape(object_term)}\b",
            rf"\binteract(?:s|ing|ed)?\s+with\b[^,.;&]{{0,32}}\b{re.escape(object_term)}\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, activity):
                distance = match.end() - match.start()
                if best is None or distance < best:
                    best = distance
    return best


def _hand_side_near_action(action: str, activity: str) -> str:
    action_terms = _action_terms(action)
    for term in action_terms:
        for side in ["left", "right"]:
            patterns = [
                rf"{term}\w*[^,.;&]{{0,48}}\bwith\s+(?:the\s+)?{side}\s+hand\b",
                rf"{term}\w*[^,.;&]{{0,48}}\bin\s+(?:the\s+)?{side}\s+hand\b",
            ]
            if any(re.search(pattern, activity) for pattern in patterns):
                return side
    clauses = re.split(r"\s+(?:and|while|then)\s+|[,;]", activity)
    for clause in clauses:
        for term in action_terms:
            if not re.search(rf"\b{re.escape(term)}\w*\b", clause):
                continue
            sides = [side for side in ["left", "right"] if re.search(rf"\b{side}\s+hand\b", clause)]
            if len(sides) == 1:
                return sides[0]
    return "unknown"


def _action_terms(action: str) -> list[str]:
    return {
        "grasp": ["grasp", "grab"],
        "hold": ["hold", "held"],
        "pick": ["pick", "take"],
        "push": ["push"],
        "pull": ["pull"],
        "press": ["press", "tap"],
        "press button": ["press"],
        "pull out": ["pull"],
        "take out": ["take", "remove"],
        "put into": ["put", "place"],
        "take from": ["take"],
        "put back": ["put", "return"],
        "place": ["place", "put", "set"],
        "carry": ["carry"],
        "support": ["support", "stabilize"],
        "touch": ["touch"],
        "reach": ["reach", "move"],
        "point": ["point"],
        "scan": ["scan", "swipe"],
        "operate": ["operate", "interact", "use", "adjust", "manipulate", "handle"],
        "gesture": ["gestur", "wav"],
        "stack": ["stack"],
        "load": ["load"],
        "unload": ["unload"],
    }.get(action, [action.split()[0]])


def _object_type_prior(action: str, attrs: list[dict[str, Any]]) -> dict[str, Any] | None:
    priors = {
        "push": ["shopping cart", "cart", "door", "box"],
        "pull": ["door", "drawer", "cart", "box"],
        "press": ["button", "phone", "payment_terminal", "terminal"],
        "press button": ["button"],
        "scan": ["product", "barcode", "phone", "payment_terminal", "terminal"],
        "operate": ["phone", "payment_terminal", "terminal", "device", "tool"],
        "put into": ["shopping cart", "cart", "basket", "box", "bag", "container"],
        "take from": ["shelf", "box", "bag", "container"],
        "put back": ["shelf", "box", "bag", "container"],
        "open": ["door", "lid", "cap", "bottle"],
        "close": ["door", "lid", "cap", "bottle"],
        "uncap": ["cap", "bottle"],
        "pour": ["bottle", "cup", "container"],
    }.get(action, [])
    matches = [attr for attr in attrs if any(token in str(attr["name"]).lower() for token in priors)]
    if len(matches) == 1:
        return matches[0]
    return None


def _deduplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for action in actions:
        key = (str(action["action"]), str(action["object_name"]), str(action["hand_side"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return out


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.8
    return max(0.0, min(1.0, confidence))
