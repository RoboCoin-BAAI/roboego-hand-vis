#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable

from roboego_hand_vis.objects.semantic_qwen import DEFAULT_QWEN_BASE_URL, resolve_qwen_base_url
from roboego_hand_vis.post_annotation_actions.robolabel_actions import ACTION_TO_PHASE, ACTION_TO_SKILL, PHASE_ACTIONS


DEFAULT_PROMPT = Path("docs/prompts/stage1_subtask_segmentation_en.md")
DEFAULT_OBJECT_PROMPT = Path("docs/prompts/stage1_subtask_object_resolution_en.md")
DEFAULT_OUTPUT_NAME = "stage1_subtasks_left_v1.json"
ALLOWED_HAND_SIDES = {"left", "right", "both", "unknown"}
ALLOWED_SKILLS = set(ACTION_TO_SKILL.values()) | {"idle"}
ALLOWED_ACTIONS = set(PHASE_ACTIONS) | ALLOWED_SKILLS
SELECTABLE_ACTIONS = ALLOWED_ACTIONS | set(ACTION_TO_SKILL) | set(ACTION_TO_PHASE)
SUPPORT_ACTIONS = {"hold"}


def build_compact_subtask_payload(stage1_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "group_resolved_stage1_actions_into_subtasks",
        "source_stage": stage1_payload.get("stage", ""),
        "frame_count": int(stage1_payload.get("frame_count", 0) or 0),
        "action_frame_count": int(stage1_payload.get("action_frame_count", 0) or 0),
        "frames": [_compact_frame(frame) for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)],
    }


def build_object_resolution_payload(stage1_payload: dict[str, Any], subtasks: list[dict[str, Any]]) -> dict[str, Any]:
    frames_by_t = {frame.get("t"): frame for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)}
    return {
        "task": "resolve_subtask_object_aliases",
        "source_stage": stage1_payload.get("stage", ""),
        "subtasks": [
            _object_resolution_subtask_payload(subtask, frames_by_t)
            for subtask in subtasks
            if isinstance(subtask, dict)
        ],
    }


def parse_model_subtask_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return {"status": "parse_failed", "raw_response": text}
    if not isinstance(data, dict) or not isinstance(data.get("subtasks"), list):
        return {"status": "parse_failed", "raw_response": text}
    try:
        subtasks = [_validate_subtask(subtask, idx + 1) for idx, subtask in enumerate(data["subtasks"])]
    except ValueError as exc:
        return {"status": "validation_failed", "error": str(exc), "raw_response": text}
    return {"status": "ok", "subtasks": subtasks}


def parse_model_object_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError:
        return {"status": "parse_failed", "raw_response": text}
    aliases_by_subtask = data.get("object_aliases_by_subtask") if isinstance(data, dict) else None
    if not isinstance(aliases_by_subtask, dict):
        return {"status": "parse_failed", "raw_response": text}
    return {
        "status": "ok",
        "object_aliases_by_subtask": {
            str(subtask_id): _validate_object_aliases(aliases)
            for subtask_id, aliases in aliases_by_subtask.items()
        },
    }


def write_model_subtask_output(
    *,
    input_json: Path,
    output_json: Path,
    resolver: Callable[[dict[str, Any]], dict[str, Any]],
    object_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    prompt_path: Path | None = None,
    model: str | None = None,
    chunk_size: int = 12,
    overlap: int = 1,
) -> dict[str, Any]:
    stage1_payload = json.loads(Path(input_json).read_text(encoding="utf-8"))
    if chunk_size > 0:
        resolved_subtasks, chunk_summaries = resolve_subtasks_in_chunks(
            stage1_payload,
            resolver=resolver,
            chunk_size=chunk_size,
            overlap=overlap,
            object_resolver=object_resolver,
        )
        subtask_stage = "robolabel_stage1_subtasks_from_model_chunked"
    else:
        review_payload = build_compact_subtask_payload(stage1_payload)
        resolved = resolver(review_payload)
        if resolved.get("status") != "ok":
            raise RuntimeError(f"Model subtask resolver failed: {resolved.get('status')}: {resolved.get('error', '')}")
        resolved_subtasks = resolve_objects_and_enrich_subtasks(
            stage1_payload,
            resolved["subtasks"],
            object_resolver=object_resolver,
        )
        chunk_summaries = []
        subtask_stage = "robolabel_stage1_subtasks_from_model"
    output = _build_output_payload(
        stage1_payload,
        subtasks=resolved_subtasks,
        input_json=input_json,
        prompt_path=prompt_path,
        model=model,
        subtask_stage=subtask_stage,
        chunk_size=chunk_size,
        overlap=overlap,
        chunk_summaries=chunk_summaries,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def resolve_subtasks_in_chunks(
    stage1_payload: dict[str, Any],
    *,
    resolver: Callable[[dict[str, Any]], dict[str, Any]],
    chunk_size: int,
    overlap: int,
    object_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = [frame for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)]
    chunks = frame_chunks(frames, chunk_size=chunk_size, overlap=overlap)
    chunk_results = []
    for chunk_index, chunk_frames in enumerate(chunks):
        chunk_payload = _stage1_payload_for_frames(stage1_payload, chunk_frames)
        print(
            f"chunk {chunk_index + 1}/{len(chunks)}: "
            f"t={chunk_frames[0].get('t')}..{chunk_frames[-1].get('t')} "
            f"frames={len(chunk_frames)}",
            file=sys.stderr,
            flush=True,
        )
        resolved = resolver(build_compact_subtask_payload(chunk_payload))
        if resolved.get("status") != "ok":
            raise RuntimeError(
                f"Model subtask resolver failed for chunk {chunk_index}: "
                f"{resolved.get('status')}: {resolved.get('error', '')}"
            )
        chunk_results.append(
            {
                "chunk_index": chunk_index,
                "start_t": chunk_frames[0].get("t") if chunk_frames else None,
                "end_t": chunk_frames[-1].get("t") if chunk_frames else None,
                "frame_count": len(chunk_frames),
                "subtasks": resolve_objects_and_enrich_subtasks(
                    chunk_payload,
                    resolved["subtasks"],
                    object_resolver=object_resolver,
                ),
            }
        )
        print(
            f"chunk {chunk_index + 1}/{len(chunks)}: "
            f"ok subtasks={len(resolved['subtasks'])}",
            file=sys.stderr,
            flush=True,
        )
    return merge_chunk_subtasks(chunk_results), [
        {key: value for key, value in result.items() if key != "subtasks"}
        for result in chunk_results
    ]


def resolve_objects_and_enrich_subtasks(
    stage1_payload: dict[str, Any],
    subtasks: list[dict[str, Any]],
    *,
    object_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if object_resolver is None:
        return enrich_selected_subtasks(stage1_payload, subtasks)
    object_payload = build_object_resolution_payload(stage1_payload, subtasks)
    resolved = object_resolver(object_payload)
    if resolved.get("status") != "ok":
        raise RuntimeError(f"Model object resolver failed: {resolved.get('status')}: {resolved.get('error', '')}")
    return enrich_selected_subtasks(
        stage1_payload,
        _attach_object_aliases_to_subtasks(subtasks, resolved.get("object_aliases_by_subtask", {})),
    )


def _attach_object_aliases_to_subtasks(
    subtasks: list[dict[str, Any]],
    aliases_by_subtask: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for subtask in subtasks:
        item = dict(subtask)
        aliases = aliases_by_subtask.get(str(item.get("id")))
        if aliases:
            item["object_aliases"] = aliases
        out.append(item)
    return out


def frame_chunks(frames: list[dict[str, Any]], *, chunk_size: int, overlap: int = 0) -> list[list[dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(frames), step):
        chunk = frames[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(frames):
            break
    return chunks


def merge_chunk_subtasks(chunk_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    seen: set[tuple[Any, ...]] = set()
    next_id = 1
    for result in chunk_results:
        for subtask in result.get("subtasks", []):
            key = _subtask_dedupe_key(subtask)
            if key in seen:
                continue
            seen.add(key)
            item = dict(subtask)
            item["id"] = next_id
            merged.append(item)
            next_id += 1
    return _coalesce_adjacent_subtasks(merged)


def enrich_selected_subtasks(stage1_payload: dict[str, Any], subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames_by_t = {frame.get("t"): frame for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)}
    enriched = []
    for subtask in subtasks:
        item = dict(subtask)
        source_frames = _validate_source_frames(item.get("source_frames", []), int(item.get("id", 0) or 0))
        span_frames = _frames_for_source_frames(source_frames, frames_by_t)
        action, skill, phase_action = _aggregate_action_for_frames(span_frames, item)
        item["source_frames"] = source_frames
        item["action"] = action
        if phase_action and phase_action != "idle":
            item["phase_action"] = phase_action
        else:
            item.pop("phase_action", None)
        if skill and skill != "idle":
            item["skill"] = skill
        else:
            item.pop("skill", None)
        item.pop("selected_action", None)
        source_object = _dominant_object_name(span_frames, action_name=action)
        canonical_object = _canonical_object_from_aliases(item.get("object_aliases"), source_object=source_object)
        if canonical_object:
            item["object_name"] = canonical_object
        elif source_object and action != "idle":
            item["object_name"] = source_object
        item["description"] = _aggregate_description(action, span_frames, item)
        _merge_span_context(item, span_frames)
        _ensure_phase_spans(item)
        enriched.append(_prune_empty_subtask_fields(item))
    return enriched


def _coalesce_adjacent_subtasks(subtasks: list[dict[str, Any]], *, max_gap: float = 1.0) -> list[dict[str, Any]]:
    if not subtasks:
        return []
    ordered = sorted(subtasks, key=lambda item: (float(item.get("start_ts", 0)), float(item.get("end_ts", 0))))
    coalesced: list[dict[str, Any]] = []
    for subtask in ordered:
        item = dict(subtask)
        if coalesced and _can_coalesce_subtasks(coalesced[-1], item, max_gap=max_gap):
            _merge_subtask_into(coalesced[-1], item)
        else:
            _ensure_phase_spans(item)
            coalesced.append(item)
    coalesced = _trim_boundary_overlaps(coalesced)
    for idx, subtask in enumerate(coalesced, start=1):
        subtask["id"] = idx
    return coalesced


def _trim_boundary_overlaps(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for subtask in subtasks:
        item = dict(subtask)
        if trimmed:
            previous_frames = set(trimmed[-1].get("source_frames", []) or [])
            current_frames = set(item.get("source_frames", []) or [])
            overlap = previous_frames & current_frames
            if overlap:
                _remove_source_frames(trimmed[-1], overlap)
                if not trimmed[-1].get("source_frames"):
                    trimmed.pop()
        trimmed.append(item)
    return trimmed


def _remove_source_frames(subtask: dict[str, Any], frames_to_remove: set[Any]) -> None:
    remaining = [frame for frame in subtask.get("source_frames", []) or [] if frame not in frames_to_remove]
    subtask["source_frames"] = _sorted_unique_values(remaining)
    if not subtask["source_frames"]:
        return
    start = subtask["source_frames"][0]
    end = subtask["source_frames"][-1]
    subtask["start_ts"] = start
    subtask["end_ts"] = end
    subtask["start_frame"] = start
    subtask["end_frame"] = end
    if subtask.get("phases"):
        for phase in subtask["phases"]:
            phase_frames = [frame for frame in phase.get("source_frames", []) or [] if frame not in frames_to_remove]
            phase["source_frames"] = _sorted_unique_values(phase_frames)
            if phase["source_frames"]:
                phase["start_ts"] = phase["source_frames"][0]
                phase["end_ts"] = phase["source_frames"][-1]
                phase["start_frame"] = phase["source_frames"][0]
                phase["end_frame"] = phase["source_frames"][-1]
        subtask["phases"] = [phase for phase in subtask["phases"] if phase.get("source_frames")]


def _can_coalesce_subtasks(first: dict[str, Any], second: dict[str, Any], *, max_gap: float) -> bool:
    if _core_action_key(first) != _core_action_key(second):
        return False
    first_end = _as_float(first.get("end_ts"))
    second_start = _as_float(second.get("start_ts"))
    if first_end is None or second_start is None:
        return False
    if second_start - first_end > max_gap:
        return False
    for field in ["object_name", "hand_side"]:
        first_value = first.get(field)
        second_value = second.get(field)
        if field == "object_name":
            if first_value and second_value and _object_identity_key(first_value) != _object_identity_key(second_value):
                return False
            continue
        if first_value and second_value and first_value != second_value:
            return False
    return True


def _merge_subtask_into(first: dict[str, Any], second: dict[str, Any]) -> None:
    _ensure_phase_spans(first)
    _ensure_phase_spans(second)
    first["end_ts"] = second.get("end_ts", first.get("end_ts"))
    first["end_frame"] = second.get("end_frame", first.get("end_frame"))
    first["source_frames"] = _sorted_unique_values(first.get("source_frames", []) + second.get("source_frames", []))
    first["actions"] = list(first.get("actions", [])) + list(second.get("actions", []))
    first["phase_action"] = _merged_phase_action(first.get("phase_action"), second.get("phase_action"))
    first["phases"] = _coalesce_phase_spans(list(first.get("phases", [])) + list(second.get("phases", [])))
    if not first.get("description") and second.get("description"):
        first["description"] = second["description"]
    if "source_chunk_index" in first or "source_chunk_index" in second:
        first["source_chunk_index"] = first.get("source_chunk_index", second.get("source_chunk_index"))


def _merged_phase_action(first: Any, second: Any) -> str:
    first_text = str(first or "").strip()
    second_text = str(second or "").strip()
    if first_text and first_text != "idle":
        return first_text
    return second_text


def _ensure_phase_spans(subtask: dict[str, Any]) -> None:
    if subtask.get("phases"):
        subtask["phases"] = _coalesce_phase_spans(list(subtask.get("phases", [])))
        return
    phase_action = str(subtask.get("phase_action", "")).strip()
    if not phase_action or phase_action == "idle":
        return
    subtask["phases"] = [
        {
            "phase_action": phase_action,
            "start_ts": subtask.get("start_ts"),
            "end_ts": subtask.get("end_ts"),
            "start_frame": subtask.get("start_frame"),
            "end_frame": subtask.get("end_frame"),
            "source_frames": list(subtask.get("source_frames", [])),
        }
    ]


def _coalesce_phase_spans(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(phases, key=lambda item: (float(item.get("start_ts", 0)), float(item.get("end_ts", 0))))
    merged: list[dict[str, Any]] = []
    for phase in ordered:
        item = dict(phase)
        if (
            merged
            and merged[-1].get("phase_action") == item.get("phase_action")
            and _phase_gap(merged[-1], item) <= 1.0
        ):
            merged[-1]["end_ts"] = item.get("end_ts", merged[-1].get("end_ts"))
            merged[-1]["end_frame"] = item.get("end_frame", merged[-1].get("end_frame"))
            merged[-1]["source_frames"] = _sorted_unique_values(
                list(merged[-1].get("source_frames", [])) + list(item.get("source_frames", []))
            )
        else:
            merged.append(item)
    return merged


def _phase_gap(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_end = _as_float(first.get("end_ts"))
    second_start = _as_float(second.get("start_ts"))
    if first_end is None or second_start is None:
        return 999999.0
    return second_start - first_end


def _core_action_key(subtask: dict[str, Any]) -> tuple[Any, ...]:
    return (
        subtask.get("action"),
        subtask.get("skill"),
    )


def _object_identity_key(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    words = [word for word in text.split() if word not in {"a", "an", "the", "black", "white", "metal", "unit"}]
    if any(word in {"rack", "shelf", "shelves"} for word in words):
        return "rack_shelf"
    return " ".join(words)


def call_qwen_subtask_resolver(
    review_payload: dict[str, Any],
    *,
    prompt_path: Path = DEFAULT_PROMPT,
    model: str = "qwen-plus",
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    system_prompt, user_prompt = read_prompt_markdown(prompt_path)
    content = _call_qwen_json_resolver(
        review_payload,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    if isinstance(content, dict):
        return content
    parsed = parse_model_subtask_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    parsed["raw_response"] = content
    return parsed


def call_qwen_object_resolver(
    review_payload: dict[str, Any],
    *,
    prompt_path: Path = DEFAULT_OBJECT_PROMPT,
    model: str = "qwen-plus",
    api_key_env: str = "DASHSCOPE_API_KEY",
    base_url: str = DEFAULT_QWEN_BASE_URL,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    system_prompt, user_prompt = read_prompt_markdown(prompt_path)
    content = _call_qwen_json_resolver(
        review_payload,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    if isinstance(content, dict):
        return content
    parsed = parse_model_object_response(content)
    parsed["provider"] = "qwen"
    parsed["model"] = model
    parsed["raw_response"] = content
    return parsed


def _call_qwen_json_resolver(
    review_payload: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key_env: str,
    base_url: str,
    timeout_s: float,
) -> str | dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")
    body = {
        "model": model,
        "messages": build_subtask_resolver_messages(
            review_payload,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ),
        "temperature": 0,
    }
    req = urllib.request.Request(
        resolve_qwen_base_url(base_url),
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        return {"status": "request_failed", "error": f"{type(exc).__name__}: {exc}"}
    return payload["choices"][0]["message"]["content"]


def build_subtask_resolver_messages(
    review_payload: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.rstrip() + "\n\n" + json.dumps(review_payload, ensure_ascii=False, indent=2)},
    ]


def read_prompt_markdown(prompt_path: Path) -> tuple[str, str]:
    text = Path(prompt_path).read_text(encoding="utf-8")
    blocks = re.findall(r"```text\n(.*?)\n```", text, flags=re.DOTALL)
    if len(blocks) < 2:
        raise ValueError(f"{prompt_path}: expected at least two ```text blocks")
    return blocks[0].strip(), blocks[1].strip()


def main() -> None:
    args = _parse_args()
    input_json = Path(args.input_json).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else input_json.with_name(DEFAULT_OUTPUT_NAME)
    )
    prompt_path = Path(args.prompt).expanduser().resolve()
    if args.dry_run:
        stage1_payload = json.loads(input_json.read_text(encoding="utf-8"))
        report = build_dry_run_report(stage1_payload, chunk_size=args.chunk_size, overlap=args.overlap)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_json} (use --overwrite)")

    def resolver(review_payload: dict[str, Any]) -> dict[str, Any]:
        return call_qwen_subtask_resolver(
            review_payload,
            prompt_path=prompt_path,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
        )

    object_prompt_path = Path(args.object_prompt).expanduser().resolve()

    def object_resolver(review_payload: dict[str, Any]) -> dict[str, Any]:
        return call_qwen_object_resolver(
            review_payload,
            prompt_path=object_prompt_path,
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_s=args.timeout_s,
        )

    output = write_model_subtask_output(
        input_json=input_json,
        output_json=output_json,
        resolver=resolver,
        object_resolver=None if args.skip_object_resolver else object_resolver,
        prompt_path=prompt_path,
        model=args.model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps({"output_json": str(output_json), "subtask_count": output["subtask_count"]}, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a text model to group resolved Stage 1 actions into subtasks.")
    parser.add_argument("--input_json", required=True, help="stage1_actions_left_v1_resolved.json path.")
    parser.add_argument("--output_json", default="", help=f"Output JSON path; default: sibling {DEFAULT_OUTPUT_NAME}.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT), help="Prompt markdown with system/user text blocks.")
    parser.add_argument(
        "--object_prompt",
        default=str(DEFAULT_OBJECT_PROMPT),
        help="Prompt markdown for per-subtask object alias resolution.",
    )
    parser.add_argument("--model", default="qwen-plus", help="Qwen/OpenAI-compatible text model.")
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--base_url", default=DEFAULT_QWEN_BASE_URL, help="OpenAI-compatible chat completions URL.")
    parser.add_argument("--timeout_s", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--chunk_size", type=int, default=12, help="Frames per model call; use 0 for one request.")
    parser.add_argument("--overlap", type=int, default=1, help="Overlapping frames between adjacent chunks.")
    parser.add_argument("--dry_run", action="store_true", help="Print chunk sizes without calling the API.")
    parser.add_argument("--skip_object_resolver", action="store_true", help="Skip the second model pass for object aliases.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_json if it already exists.")
    return parser.parse_args()


def build_dry_run_report(stage1_payload: dict[str, Any], *, chunk_size: int, overlap: int) -> dict[str, Any]:
    frames = [frame for frame in stage1_payload.get("frames", []) if isinstance(frame, dict)]
    if chunk_size <= 0:
        review_payload = build_compact_subtask_payload(stage1_payload)
        compact = json.dumps(review_payload, ensure_ascii=False)
        return {
            "stage": "dry_run_stage1_subtasks_from_model",
            "source_stage": stage1_payload.get("stage", ""),
            "chunk_count": 1,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_t": frames[0].get("t") if frames else None,
                    "end_t": frames[-1].get("t") if frames else None,
                    "frame_count": len(frames),
                    "action_frame_count": int(review_payload.get("action_frame_count", 0) or 0),
                    "compact_chars": len(compact),
                    "compact_bytes": len(compact.encode("utf-8")),
                }
            ],
        }
    chunks = frame_chunks(frames, chunk_size=chunk_size, overlap=overlap)
    chunk_reports = []
    for chunk_index, chunk_frames in enumerate(chunks):
        chunk_payload = _stage1_payload_for_frames(stage1_payload, chunk_frames)
        review_payload = build_compact_subtask_payload(chunk_payload)
        compact = json.dumps(review_payload, ensure_ascii=False)
        chunk_reports.append(
            {
                "chunk_index": chunk_index,
                "start_t": chunk_frames[0].get("t") if chunk_frames else None,
                "end_t": chunk_frames[-1].get("t") if chunk_frames else None,
                "frame_count": len(chunk_frames),
                "action_frame_count": int(review_payload.get("action_frame_count", 0) or 0),
                "compact_chars": len(compact),
                "compact_bytes": len(compact.encode("utf-8")),
            }
        )
    return {
        "stage": "dry_run_stage1_subtasks_from_model_chunked",
        "source_stage": stage1_payload.get("stage", ""),
        "chunk_count": len(chunks),
        "chunks": chunk_reports,
    }


def _compact_frame(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "t": frame.get("t"),
        "scene": frame.get("scene", ""),
        "activity": frame.get("activity", ""),
    }


def _object_resolution_subtask_payload(
    subtask: dict[str, Any],
    frames_by_t: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    source_frames = list(subtask.get("source_frames", []) or [])
    frames = _frames_for_source_frames(source_frames, frames_by_t)
    return {
        "id": int(subtask.get("id", 0) or 0),
        "source_frames": source_frames,
        "activity_span": [
            {"t": frame.get("t"), "activity": str(frame.get("activity", "")).strip()}
            for frame in frames
            if str(frame.get("activity", "")).strip()
        ],
        "objects": _objects_for_resolution(frames),
    }


def _objects_for_resolution(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[str]] = {}
    for frame in frames:
        for action in frame.get("frame_actions", []) or []:
            if not isinstance(action, dict):
                continue
            name = str(action.get("object_name") or action.get("manipulated_object") or "").strip()
            description = str(action.get("object_description") or "").strip()
            if not name or name == "unknown":
                continue
            descriptions = by_name.setdefault(name, [])
            if description and description != "unknown" and description not in descriptions:
                descriptions.append(description)
    return [
        {"name": name, "descriptions": descriptions}
        for name, descriptions in by_name.items()
    ]


def _build_output_payload(
    stage1_payload: dict[str, Any],
    *,
    subtasks: list[dict[str, Any]],
    input_json: Path,
    prompt_path: Path | None,
    model: str | None,
    subtask_stage: str,
    chunk_size: int,
    overlap: int,
    chunk_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    output = {
        "stage": stage1_payload.get("stage", ""),
        "source_chunk_dir": stage1_payload.get("source_chunk_dir", ""),
        "source_files": stage1_payload.get("source_files", []),
        "frame_count": int(stage1_payload.get("frame_count", 0) or 0),
        "action_frame_count": int(stage1_payload.get("action_frame_count", 0) or 0),
        "subtask_stage": subtask_stage,
        "source_stage1_actions": str(input_json),
        "subtask_count": len(subtasks),
        "subtasks": subtasks,
    }
    if chunk_size > 0:
        output["chunk_size"] = chunk_size
        output["overlap"] = overlap
        output["chunk_count"] = len(chunk_summaries)
        output["chunks"] = chunk_summaries
    if prompt_path is not None:
        output["prompt"] = str(prompt_path)
    if model:
        output["model"] = model
    return output


def _stage1_payload_for_frames(stage1_payload: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_payload = dict(stage1_payload)
    chunk_payload["frames"] = frames
    chunk_payload["frame_count"] = len(frames)
    chunk_payload["action_frame_count"] = sum(1 for frame in frames if frame.get("frame_actions"))
    return chunk_payload


def _validate_subtask(value: Any, expected_id: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"subtask[{expected_id}] must be an object")
    out = dict(value)
    out["id"] = int(out.get("id", expected_id))
    for field in ["start_ts", "end_ts", "start_frame", "end_frame"]:
        if field not in out:
            raise ValueError(f"subtask[{out['id']}] missing {field}")
    out["scene"] = str(out.get("scene", ""))
    if "selected_action" in out:
        out["action"] = _validate_selected_action_name(out.pop("selected_action"), f"subtask[{out['id']}].action")
    elif "action" in out:
        out["action"] = _validate_selected_action_name(out["action"], f"subtask[{out['id']}].action")
    if "action" in out:
        action, skill, phase_action = _normalize_selected_action(out["action"])
        out["action"] = action
        if phase_action and phase_action != "idle":
            out["phase_action"] = phase_action
        else:
            out.pop("phase_action", None)
        if skill and skill != "idle":
            out["skill"] = skill
        else:
            out.pop("skill", None)
    if "object_aliases" in out:
        out["object_aliases"] = _validate_object_aliases(out.get("object_aliases", []))
    out["description"] = str(out.get("description", "")).strip()
    out["source_frames"] = _validate_source_frames(out.get("source_frames", []), out["id"])
    actions = out.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError(f"subtask[{out['id']}].actions must be a list")
    if actions:
        out["actions"] = [_validate_supporting_action(action, out["id"]) for action in actions]
    else:
        out.pop("actions", None)
    return out


def _validate_supporting_action(value: Any, subtask_id: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"subtask[{subtask_id}].actions[] must be an object")
    out = dict(value)
    out["action"] = _validate_optional_action_name(out.get("action", ""), f"subtask[{subtask_id}].actions[].action")
    out["phase_action"] = _validate_phase_action_name(
        out.get("phase_action", out["action"]),
        f"subtask[{subtask_id}].actions[].phase_action",
    )
    out["skill"] = _validate_skill_name(out.get("skill", ""), f"subtask[{subtask_id}].actions[].skill")
    out["hand_side"] = _validate_hand_side(out.get("hand_side", "unknown"), f"subtask[{subtask_id}].actions[].hand_side")
    out["confidence"] = _confidence(out.get("confidence", 0.0))
    return out


def _validate_action_name(value: Any, field: str) -> str:
    name = str(value).strip()
    if name not in ALLOWED_ACTIONS:
        raise ValueError(f"{field} invalid action: {name}")
    return name


def _validate_selected_action_name(value: Any, field: str) -> str:
    name = str(value).strip()
    if name not in SELECTABLE_ACTIONS:
        raise ValueError(f"{field} invalid selected_action: {name}")
    return name


def _validate_optional_action_name(value: Any, field: str) -> str:
    name = str(value).strip()
    if name and name not in ALLOWED_ACTIONS:
        raise ValueError(f"{field} invalid action: {name}")
    return name


def _validate_phase_action_name(value: Any, field: str) -> str:
    name = str(value).strip()
    if name not in PHASE_ACTIONS:
        raise ValueError(f"{field} invalid phase_action: {name}")
    return name


def _validate_skill_name(value: Any, field: str) -> str:
    name = str(value).strip()
    if name in PHASE_ACTIONS:
        return ""
    if name and name not in ALLOWED_SKILLS:
        raise ValueError(f"{field} invalid skill: {name}")
    return name


def _normalize_selected_action(selected_action: str, *, fallback_phase: str = "idle") -> tuple[str, str, str]:
    selected = str(selected_action).strip()
    if not selected:
        return "idle", "idle", "idle"
    if selected in ACTION_TO_SKILL:
        skill = ACTION_TO_SKILL[selected]
        phase_action = ACTION_TO_PHASE.get(selected, fallback_phase)
        return skill, skill, phase_action
    if selected in ALLOWED_SKILLS:
        phase_action = fallback_phase if fallback_phase in PHASE_ACTIONS else "idle"
        return selected, selected, phase_action
    if selected in PHASE_ACTIONS:
        return selected, "", selected
    raise ValueError(f"invalid selected_action: {selected}")


def _dominant_phase_action(source_frames: Any, frames_by_t: dict[Any, dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    if not isinstance(source_frames, list):
        return "idle"
    for t in source_frames:
        frame = frames_by_t.get(t)
        if frame is None:
            try:
                frame = frames_by_t.get(int(t))
            except (TypeError, ValueError):
                frame = None
        if not isinstance(frame, dict):
            continue
        for action in frame.get("frame_actions", []):
            if not isinstance(action, dict):
                continue
            phase_action = str(action.get("phase_action", "")).strip()
            if phase_action in PHASE_ACTIONS:
                counts[phase_action] = counts.get(phase_action, 0) + 1
    if not counts:
        return "idle"
    return max(counts.items(), key=lambda item: item[1])[0]


def _frames_for_source_frames(source_frames: list[Any], frames_by_t: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    frames = []
    for t in source_frames:
        frame = frames_by_t.get(t)
        if frame is None:
            try:
                frame = frames_by_t.get(int(t))
            except (TypeError, ValueError):
                frame = None
        if isinstance(frame, dict):
            frames.append(frame)
    return frames


def _aggregate_action_for_frames(frames: list[dict[str, Any]], item: dict[str, Any]) -> tuple[str, str, str]:
    hinted = str(item.get("action", "")).strip()
    if hinted in SELECTABLE_ACTIONS:
        return _normalize_selected_action(hinted, fallback_phase=_dominant_phase_from_frames(frames))
    action_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    for frame in frames:
        for action in frame.get("frame_actions", []):
            if not isinstance(action, dict):
                continue
            action_name = str(action.get("action", "")).strip()
            phase_action = str(action.get("phase_action", "")).strip()
            if action_name:
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
            if phase_action:
                phase_counts[phase_action] = phase_counts.get(phase_action, 0) + 1
    if action_counts:
        selected = _select_representative_action(action_counts)
        return _normalize_selected_action(selected, fallback_phase=_dominant_phase_from_frames(frames, action_name=selected))
    if phase_counts:
        selected = max(phase_counts.items(), key=lambda entry: entry[1])[0]
        return _normalize_selected_action(selected)
    return "idle", "", ""


def _select_representative_action(action_counts: dict[str, int]) -> str:
    task_counts = {action: count for action, count in action_counts.items() if action not in SUPPORT_ACTIONS}
    if task_counts:
        return max(task_counts.items(), key=lambda entry: entry[1])[0]
    return max(action_counts.items(), key=lambda entry: entry[1])[0]


def _dominant_phase_from_frames(frames: list[dict[str, Any]], *, action_name: str = "") -> str:
    counts: dict[str, int] = {}
    for frame in frames:
        for action in frame.get("frame_actions", []):
            if not isinstance(action, dict):
                continue
            if action_name and str(action.get("action", "")).strip() != action_name:
                continue
            phase_action = str(action.get("phase_action", "")).strip()
            if phase_action in PHASE_ACTIONS:
                counts[phase_action] = counts.get(phase_action, 0) + 1
    if not counts:
        return "idle"
    return max(counts.items(), key=lambda item: item[1])[0]


def _aggregate_description(action: str, frames: list[dict[str, Any]], item: dict[str, Any]) -> str:
    existing = str(item.get("description", "")).strip()
    if existing:
        return existing
    if action == "idle":
        for frame in frames:
            activity = str(frame.get("activity", "")).strip()
            if activity:
                return activity
        return "wearer idle"
    obj = str(item.get("object_name", "")).strip() or _dominant_object_name(frames)
    if obj:
        return f"wearer {action} {obj}"
    return f"wearer {action}"


def _dominant_object_name(frames: list[dict[str, Any]], *, action_name: str = "") -> str:
    counts: dict[str, int] = {}
    for frame in frames:
        for action in frame.get("frame_actions", []):
            if not isinstance(action, dict):
                continue
            if action_name and str(action.get("action", "")).strip() != action_name:
                continue
            name = str(action.get("object_name") or action.get("manipulated_object") or "").strip()
            if name and name != "unknown":
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def _validate_object_aliases(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    aliases = []
    for item in value:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical_name", "")).strip()
        raw_aliases = item.get("aliases", [])
        if not canonical or not isinstance(raw_aliases, list):
            continue
        alias_values = [str(alias).strip() for alias in raw_aliases if str(alias).strip()]
        if canonical not in alias_values:
            alias_values.insert(0, canonical)
        aliases.append({"canonical_name": canonical, "aliases": alias_values})
    return aliases


def _canonical_object_from_aliases(value: Any, *, source_object: str = "") -> str:
    aliases = _validate_object_aliases(value)
    if not aliases:
        return ""
    if source_object:
        source_key = _object_identity_key(source_object)
        for item in aliases:
            alias_values = item.get("aliases", [])
            if any(_object_identity_key(alias) == source_key for alias in alias_values):
                return str(item.get("canonical_name", "")).strip()
        return ""
    return str(aliases[0].get("canonical_name", "")).strip()


def _merge_span_context(item: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    if frames:
        item["scene"] = _first_nonempty(frame.get("scene") for frame in frames)
    item["hand_visible"] = any(bool(frame.get("hand_visible", False)) for frame in frames)
    item["hands_visible"] = {
        "left": any(bool((frame.get("hands_visible") or {}).get("left", False)) for frame in frames),
        "right": any(bool((frame.get("hands_visible") or {}).get("right", False)) for frame in frames),
    }
    task_objects: list[str] = []
    seen = set()
    for frame in frames:
        for obj in frame.get("task_objects", []) or []:
            text = str(obj).strip()
            if text and text not in seen:
                seen.add(text)
                task_objects.append(text)
    item["task_objects"] = task_objects
    objects = _span_objects(frames, aliases=item.get("object_aliases", []))
    if objects:
        item["objects"] = objects


def _first_nonempty(values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _span_objects(frames: list[dict[str, Any]], *, aliases: Any) -> list[dict[str, Any]]:
    alias_groups = _validate_object_aliases(aliases)
    by_name: dict[str, list[str]] = {}
    for frame in frames:
        for action in frame.get("frame_actions", []) or []:
            if not isinstance(action, dict):
                continue
            raw_name = str(action.get("object_name") or action.get("manipulated_object") or "").strip()
            if not raw_name or raw_name == "unknown":
                continue
            name = _canonical_object_from_aliases(alias_groups, source_object=raw_name) or raw_name
            description = str(action.get("object_description") or "").strip()
            if description and description != "unknown":
                descriptions = by_name.setdefault(name, [])
                if description not in descriptions:
                    descriptions.append(description)
            else:
                by_name.setdefault(name, [])
    return [_object_attribute_summary(name, descriptions) for name, descriptions in by_name.items()]


def _object_attribute_summary(name: str, descriptions: list[str]) -> dict[str, str]:
    description = _choose_object_description(descriptions)
    fields = _parse_object_description_fields(description)
    summary = {"name": name}
    for key in ["color", "material", "shape"]:
        value = fields.get(key) or _infer_object_attribute(key, description, name)
        if value:
            summary[key] = value
    if description:
        summary["description"] = description
    return summary


def _choose_object_description(descriptions: list[str]) -> str:
    clean = [description for description in descriptions if description and description != "unknown"]
    if not clean:
        return ""
    return min(clean, key=lambda value: (len(value), value))


def _parse_object_description_fields(description: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in description.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"color", "shape", "material", "visual_description"} and value:
            fields[key] = value
    return fields


def _infer_object_attribute(field: str, description: str, name: str) -> str:
    text = f"{description} {name}".lower()
    if field == "color":
        colors = [
            "black",
            "white",
            "yellow",
            "brown",
            "gold",
            "green",
            "blue",
            "red",
            "silver",
            "gray",
            "grey",
        ]
        return _first_word_in_text(text, colors)
    if field == "material":
        materials = ["metal", "fabric", "glass", "plastic", "paper", "wood", "wooden"]
        material = _first_word_in_text(text, materials)
        return "wood" if material == "wooden" else material
    if field == "shape":
        if "two-tier" in text:
            return f"two-tier {name}"
        shapes = ["cylindrical", "rectangular", "round", "flat", "folded"]
        shape = _first_word_in_text(text, shapes)
        return shape or name
    return ""


def _first_word_in_text(text: str, words: list[str]) -> str:
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return word
    return ""


def _prune_empty_subtask_fields(item: dict[str, Any]) -> dict[str, Any]:
    pruned = dict(item)
    pruned.pop("object_aliases", None)
    pruned.pop("skill", None)
    for field in ["object_name", "manipulated_object", "object_description"]:
        if pruned.get(field) in (None, "", "unknown"):
            pruned.pop(field, None)
    if pruned.get("hand_side") in (None, "", "unknown"):
        pruned.pop("hand_side", None)
    if pruned.get("subject") in (None, "", "unknown"):
        pruned.pop("subject", None)
    if float(pruned.get("confidence", 0.0) or 0.0) == 0.0:
        pruned.pop("confidence", None)
    if not pruned.get("actions"):
        pruned.pop("actions", None)
    if len(pruned.get("phases") or []) <= 1:
        pruned.pop("phases", None)
    if not pruned.get("task_objects"):
        pruned.pop("task_objects", None)
    if pruned.get("hands_visible") == {"left": False, "right": False}:
        pruned.pop("hands_visible", None)
    if pruned.get("hand_visible") is False:
        pruned.pop("hand_visible", None)
    return pruned


def _validate_hand_side(value: Any, field: str) -> str:
    side = str(value).strip()
    if side not in ALLOWED_HAND_SIDES:
        raise ValueError(f"{field} invalid hand_side: {side}")
    return side


def _validate_hands_visible(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {"left": False, "right": False}
    return {"left": bool(value.get("left", False)), "right": bool(value.get("right", False))}


def _validate_source_frames(value: Any, subtask_id: int) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"subtask[{subtask_id}].source_frames must be a non-empty list")
    return value


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _nullable_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_unique_values(values: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return sorted(unique, key=lambda item: float(item))


def _compact_description(verb: str, obj: str | None) -> str:
    if obj:
        return f"wearer {verb} {obj}"
    return f"wearer {verb}".strip()




def _subtask_dedupe_key(subtask: dict[str, Any]) -> tuple[Any, ...]:
    return (
        subtask.get("action"),
        subtask.get("object_name"),
        subtask.get("hand_side"),
        tuple(subtask.get("source_frames", [])),
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


if __name__ == "__main__":
    main()
