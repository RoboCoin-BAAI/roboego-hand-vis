from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


@dataclass
class Sam3HandClipPlan:
    hand_window_dir: Path
    frames_dir: Path
    output_dir: Path
    prompt_text: str
    prompt_frame_idx: int
    object_id: str
    command: list[str]
    metadata_path: Path
    object_interval: dict[str, float | int] | None = None


@dataclass
class Stage1ObjectSpec:
    object_id: str
    object_name: str
    object_shape: str
    object_color: str
    object_material: str
    prompt_text: str
    evidence_count: int
    active_intervals: list[dict[str, float | int]]
    prompt_frame_idx: int


def collect_stage1_evidence(stage1_actions: Path, window: dict[str, Any], *, fps: float) -> list[dict[str, Any]]:
    payload = json.loads(Path(stage1_actions).read_text())
    frames = payload.get("frames", []) if isinstance(payload, dict) else payload
    source_segment_start_s = float(window["source_segment_range"]["start"])
    segment_start_frame = int(window["segment_frame_range"]["start"])
    segment_end_frame = int(window["segment_frame_range"]["end"])

    evidence: list[dict[str, Any]] = []
    for frame in frames:
        if "t" not in frame:
            continue
        session_t = float(frame["t"])
        segment_frame = int(round((session_t - source_segment_start_s) * fps))
        if segment_frame < segment_start_frame or segment_frame > segment_end_frame:
            continue
        item = dict(frame)
        item["session_t"] = frame["t"]
        item["segment_frame"] = segment_frame
        item["clip_frame"] = segment_frame - segment_start_frame
        evidence.append(item)
    return evidence


def build_prompt_text(*, object_name: str, object_shape: str = "", object_color: str = "") -> str:
    parts = [object_color.strip(), object_shape.strip(), object_name.strip()]
    return _join_prompt_parts(parts)


def _join_prompt_parts(parts: list[str]) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = _clean_prompt_part(part)
        if not part:
            continue
        for token in part.split():
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
    return " ".join(tokens)


def _clean_prompt_part(text: str) -> str:
    return re.sub(r"[\\/]+", " ", str(text).strip())


def build_stage1_prompt_text(
    *,
    object_name: str,
    object_shape: str = "",
    object_color: str = "",
    object_material: str = "",
) -> str:
    shape = object_shape.strip()
    material = object_material.strip()
    if shape and material and len(material.split()) == 1 and len(shape.split()) > 1:
        shape_words = shape.split()
        shape = " ".join([*shape_words[:-1], material, shape_words[-1]])
        material = ""
    parts = [object_color.strip(), shape, material, object_name.strip()]
    return _join_prompt_parts(parts)


def safe_object_id(*parts: str) -> str:
    raw = "_".join(part.strip().lower() for part in parts if part and part.strip())
    raw = raw.replace("&", "and")
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "object_0"


def canonical_object_name(name: str, shape: str = "") -> str:
    text = f"{name} {shape}".lower()
    if "phone" in text or "mobile" in text:
        return "phone"
    if "bottle" in text or "product" in text:
        if "bottle" in text:
            return "bottle"
        return "product"
    return str(name).strip().lower() or "object"


def object_group_id(name: str, shape: str = "") -> str:
    canonical_name = canonical_object_name(name, shape)
    return safe_object_id(canonical_name, _shape_for_object_id(shape, object_name=canonical_name))


def infer_object_specs_from_stage1(stage1_actions: Path, window: dict[str, Any], *, fps: float) -> list[Stage1ObjectSpec]:
    evidence = collect_stage1_evidence(Path(stage1_actions), window, fps=fps)
    grouped: dict[str, dict[str, Any]] = {}
    for frame in evidence:
        if frame.get("hand_object_interaction") is not True:
            continue
        attrs = frame.get("interaction_object_attributes", [])
        if not isinstance(attrs, list):
            attrs = []
        for attr in attrs:
            if not isinstance(attr, dict):
                continue
            name = str(attr.get("name", "")).strip()
            if not name:
                continue
            color = str(attr.get("color", "")).strip()
            shape = str(attr.get("shape", "")).strip()
            material = str(attr.get("material", "")).strip()
            canonical_name = canonical_object_name(name, shape)
            group_id = object_group_id(name, shape)
            item = grouped.setdefault(
                group_id,
                {
                    "object_id": group_id,
                    "object_name": canonical_name,
                    "object_shape": shape,
                    "object_color": color,
                    "object_material": material,
                    "evidence_count": 0,
                    "evidence": [],
                },
            )
            item["evidence_count"] += 1
            item["evidence"].append(frame)
            if len(shape) > len(str(item.get("object_shape", ""))):
                item["object_shape"] = shape
            if len(color) > len(str(item.get("object_color", ""))):
                item["object_color"] = color
            if len(material) > len(str(item.get("object_material", ""))):
                item["object_material"] = material

    specs: list[Stage1ObjectSpec] = []
    for item in grouped.values():
        object_id = safe_object_id(
            str(item["object_name"]),
            str(item["object_color"]),
            _shape_for_object_id(str(item["object_shape"]), object_name=str(item["object_name"])),
        )
        intervals = _active_intervals(item.get("evidence", []), fps=fps)
        prompt_frame_idx = _prompt_frame_idx(intervals, window)
        specs.append(
            Stage1ObjectSpec(
                object_id=object_id,
                object_name=str(item["object_name"]),
                object_shape=str(item["object_shape"]),
                object_color=str(item["object_color"]),
                object_material=str(item["object_material"]),
                prompt_text=build_stage1_prompt_text(
                    object_name=str(item["object_name"]),
                    object_shape=str(item["object_shape"]),
                    object_color=str(item["object_color"]),
                    object_material=str(item["object_material"]),
                ),
                evidence_count=int(item["evidence_count"]),
                active_intervals=intervals,
                prompt_frame_idx=prompt_frame_idx,
            )
        )
    specs.sort(key=lambda spec: -spec.evidence_count)
    return specs


def _shape_for_object_id(shape: str, *, object_name: str) -> str:
    words = []
    object_words = set(str(object_name).lower().split())
    for word in str(shape).lower().split():
        if word in object_words or word in {"product", "plastic", "metal"}:
            continue
        words.append(word)
    return " ".join(words)


def _active_intervals(evidence: list[dict[str, Any]], *, fps: float, max_gap_s: float = 1.1) -> list[dict[str, float | int]]:
    if not evidence:
        return []
    items = sorted(evidence, key=lambda item: float(item["session_t"]))
    intervals: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in items:
        if current and float(item["session_t"]) - float(current[-1]["session_t"]) > max_gap_s:
            intervals.append(current)
            current = []
        current.append(item)
    if current:
        intervals.append(current)
    out: list[dict[str, float | int]] = []
    for group in intervals:
        start = group[0]
        end = group[-1]
        out.append(
            {
                "start_s": float(start["session_t"]),
                "end_s": float(end["session_t"]),
                "start_frame": int(start["clip_frame"]),
                "end_frame": int(end["clip_frame"]),
            }
        )
    return out


def _prompt_frame_idx(intervals: list[dict[str, float | int]], window: dict[str, Any]) -> int:
    if intervals:
        best = max(intervals, key=lambda item: int(item["end_frame"]) - int(item["start_frame"]))
        return int(round((int(best["start_frame"]) + int(best["end_frame"])) * 0.5))
    segment_start_frame = int(window["segment_frame_range"]["start"])
    core = window.get("core_segment_frame_range") or window.get("segment_frame_range")
    return max(0, int(core["start"]) - segment_start_frame)


def _expanded_object_intervals(
    intervals: list[dict[str, float | int]],
    window: dict[str, Any],
    *,
    fps: float,
    expand_s: float,
    merge_gap_s: float = 0.0,
) -> list[dict[str, float | int]]:
    if not intervals:
        return []
    source_start_s = float(window["source_segment_range"]["start"])
    clip_start_s = float(window.get("start", source_start_s + int(window["segment_frame_range"]["start"]) / fps))
    clip_end_s = window.get("end")
    frame_range = window["segment_frame_range"]
    max_frame = int(frame_range.get("num_frames", 0)) - 1
    if max_frame < 0:
        max_frame = int(frame_range["end"]) - int(frame_range["start"])
    if clip_end_s is not None:
        max_frame = min(max_frame, max(0, int(round((float(clip_end_s) - clip_start_s) * fps))))
    expand_frames = max(0, int(round(float(expand_s) * fps)))
    expanded: list[dict[str, float | int]] = []
    for idx, interval in enumerate(intervals):
        source_start_frame = int(interval["start_frame"])
        source_end_frame = int(interval["end_frame"])
        start_frame = max(0, source_start_frame - expand_frames)
        end_frame = min(max_frame, source_end_frame + expand_frames)
        expanded.append(
            {
                "start_s": clip_start_s + start_frame / fps,
                "end_s": clip_start_s + end_frame / fps,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "source_start_s": float(interval["start_s"]),
                "source_end_s": float(interval["end_s"]),
                "source_start_frame": source_start_frame,
                "source_end_frame": source_end_frame,
                "clip_index": idx,
            }
        )
    return _merge_object_intervals(expanded, fps=fps, merge_gap_s=merge_gap_s)


def _merge_object_intervals(
    intervals: list[dict[str, float | int]],
    *,
    fps: float,
    merge_gap_s: float,
) -> list[dict[str, float | int]]:
    if not intervals:
        return []
    max_gap_frames = max(0, int(round(float(merge_gap_s) * fps)))
    merged: list[dict[str, float | int]] = []
    current = dict(sorted(intervals, key=lambda item: int(item["start_frame"]))[0])
    for interval in sorted(intervals, key=lambda item: int(item["start_frame"]))[1:]:
        if int(interval["start_frame"]) <= int(current["end_frame"]) + max_gap_frames:
            current["end_frame"] = max(int(current["end_frame"]), int(interval["end_frame"]))
            current["end_s"] = max(float(current["end_s"]), float(interval["end_s"]))
            current["source_start_frame"] = min(
                int(current["source_start_frame"]), int(interval["source_start_frame"])
            )
            current["source_end_frame"] = max(int(current["source_end_frame"]), int(interval["source_end_frame"]))
            current["source_start_s"] = min(float(current["source_start_s"]), float(interval["source_start_s"]))
            current["source_end_s"] = max(float(current["source_end_s"]), float(interval["source_end_s"]))
            continue
        merged.append(current)
        current = dict(interval)
    merged.append(current)
    for idx, interval in enumerate(merged):
        interval["clip_index"] = idx
    return merged


def _interval_prompt_frame_idx(interval: dict[str, float | int]) -> int:
    source_midpoint = int(round((int(interval["source_start_frame"]) + int(interval["source_end_frame"])) * 0.5))
    return max(0, min(source_midpoint, int(interval["end_frame"])) - int(interval["start_frame"]))


def _clip_dir_name(interval: dict[str, float | int]) -> str:
    return (
        f"clip_{int(interval['clip_index']):04d}_"
        f"t{int(round(float(interval['start_s']))):06d}_{int(round(float(interval['end_s']))):06d}"
    )


def materialize_interval_frames(
    source_frames_dir: Path,
    target_dir: Path,
    interval: dict[str, float | int],
) -> Path:
    frame_paths = _frame_paths(source_frames_dir)
    start_frame = int(interval["start_frame"])
    end_frame = int(interval["end_frame"])
    if start_frame < 0 or end_frame < start_frame or start_frame >= len(frame_paths):
        raise ValueError(
            f"Invalid SAM3 interval frame range [{start_frame}, {end_frame}] for {len(frame_paths)} frames."
        )
    end_frame = min(end_frame, len(frame_paths) - 1)
    clip_frames_dir = Path(target_dir)
    if clip_frames_dir.exists():
        shutil.rmtree(clip_frames_dir)
    clip_frames_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, source_path in enumerate(frame_paths[start_frame : end_frame + 1]):
        target = clip_frames_dir / f"{local_idx:06d}{source_path.suffix.lower()}"
        if target.exists():
            continue
        try:
            os.symlink(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)
    return clip_frames_dir


def find_window_by_id(windows_json: Path, window_id: str) -> dict[str, Any]:
    payload = json.loads(Path(windows_json).read_text())
    for window in payload.get("windows", []):
        if window.get("window_id") == window_id:
            return window
    raise ValueError(f"Window id {window_id!r} not found in {windows_json}")


def infer_window_id_from_dir(hand_window_dir: Path) -> str:
    name = _materialized_hand_window_root(Path(hand_window_dir)).name
    if not name.startswith("handwin_"):
        raise ValueError(f"Cannot infer hand window id from directory name: {hand_window_dir}")
    return "_".join(name.split("_")[:2])


def infer_windows_json_from_hand_window_dir(hand_window_dir: Path) -> Path:
    path = Path(hand_window_dir).resolve()
    for parent in [path, *path.parents]:
        if parent.name.startswith("session_") and (parent / "annotation").exists():
            candidate = parent / "annotation" / "hand_estimation_windows_left_v1.json"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        "Could not infer hand_estimation_windows_left_v1.json; pass --windows_json explicitly."
    )


def default_output_dir(hand_window_dir: Path, object_id: str) -> Path:
    root = _materialized_hand_window_root(Path(hand_window_dir))
    return root / "objects" / object_id


def _materialized_hand_window_root(path: Path) -> Path:
    path = Path(path)
    if path.name.startswith("handwin_"):
        return path
    for parent in path.parents:
        if parent.name.startswith("handwin_"):
            return parent
    return path


def build_sam3_hand_clip_plan(
    *,
    hand_window_dir: Path,
    stage1_actions: Path,
    window: dict[str, Any],
    object_name: str,
    object_shape: str = "",
    object_color: str = "",
    output_dir: Path | None = None,
    sam3_script: Path,
    python_executable: str = "python",
    object_id: str = "object_0",
    fps: float = 30.0,
    prompt_frame_idx: int | None = None,
    checkpoint: Path | None = None,
    no_offload_video_to_cpu: bool = False,
    frames_dir_override: Path | None = None,
    object_interval: dict[str, float | int] | None = None,
) -> Sam3HandClipPlan:
    hand_window_dir = Path(hand_window_dir)
    frames_dir = Path(frames_dir_override) if frames_dir_override is not None else resolve_hand_clip_frames_dir(hand_window_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_output_dir(hand_window_dir, object_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_start_frame = int(window["segment_frame_range"]["start"])
    core = window.get("core_segment_frame_range") or window.get("segment_frame_range")
    default_prompt_frame_idx = max(0, int(core["start"]) - segment_start_frame)
    prompt_frame_idx = default_prompt_frame_idx if prompt_frame_idx is None else int(prompt_frame_idx)
    prompt_text = build_prompt_text(
        object_name=object_name,
        object_shape=object_shape,
        object_color=object_color,
    )

    command = [
        python_executable,
        str(sam3_script),
        "--video",
        str(frames_dir),
        "--text",
        prompt_text,
        "--obj_id",
        object_id,
        "--frame_idx",
        str(prompt_frame_idx),
        "--output_dir",
        str(output_dir),
    ]
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if no_offload_video_to_cpu:
        command.append("--no_offload_video_to_cpu")

    evidence = collect_stage1_evidence(Path(stage1_actions), window, fps=fps)
    metadata = {
        "schema_version": "1.0",
        "hand_window_dir": str(hand_window_dir),
        "frames_dir": str(frames_dir),
        "window": window,
        "prompt": {
            "text": prompt_text,
            "object_name": object_name,
            "object_shape": object_shape,
            "object_color": object_color,
            "object_id": object_id,
        },
        "time_alignment": {
            "fps": fps,
            "source_segment_range": window.get("source_segment_range"),
            "segment_frame_range": window.get("segment_frame_range"),
            "core_segment_frame_range": window.get("core_segment_frame_range"),
            "prompt_frame_idx": prompt_frame_idx,
            "object_interval": object_interval,
        },
        "stage1_evidence": evidence,
        "command": command,
    }
    metadata_path = output_dir / "sam3_prompt.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return Sam3HandClipPlan(
        hand_window_dir=hand_window_dir,
        frames_dir=frames_dir,
        output_dir=output_dir,
        prompt_text=prompt_text,
        prompt_frame_idx=prompt_frame_idx,
        object_id=object_id,
        command=command,
        metadata_path=metadata_path,
        object_interval=object_interval,
    )


def build_sam3_hand_clip_plans_from_stage1(
    *,
    hand_window_dir: Path,
    stage1_actions: Path,
    window: dict[str, Any],
    output_root: Path | None = None,
    sam3_script: Path,
    python_executable: str = "python",
    fps: float = 30.0,
    prompt_frame_idx: int | None = None,
    checkpoint: Path | None = None,
    no_offload_video_to_cpu: bool = False,
    clip_by_intervals: bool = False,
    interval_expand_s: float = 0.0,
    interval_merge_gap_s: float = 0.0,
) -> list[Sam3HandClipPlan]:
    specs = infer_object_specs_from_stage1(Path(stage1_actions), window, fps=fps)
    if not specs:
        raise ValueError("No object specs found from stage1 annotations for this hand window.")
    root = Path(output_root) if output_root is not None else Path(hand_window_dir) / "objects"
    plans: list[Sam3HandClipPlan] = []
    for spec in specs:
        object_intervals = [None]
        if clip_by_intervals:
            object_intervals = _expanded_object_intervals(
                spec.active_intervals,
                window,
                fps=fps,
                expand_s=interval_expand_s,
                merge_gap_s=interval_merge_gap_s,
            )
        for interval in object_intervals:
            output_dir = root / spec.object_id
            frames_dir_override = None
            adjusted_prompt_frame_idx = spec.prompt_frame_idx if prompt_frame_idx is None else int(prompt_frame_idx)
            if interval is not None:
                output_dir = output_dir / _clip_dir_name(interval)
                source_frames_dir = resolve_hand_clip_frames_dir(Path(hand_window_dir))
                frames_dir_override = source_frames_dir
                adjusted_prompt_frame_idx = _interval_prompt_frame_idx(interval)
            plan = build_sam3_hand_clip_plan(
                hand_window_dir=Path(hand_window_dir),
                stage1_actions=Path(stage1_actions),
                window=window,
                object_name=spec.object_name,
                object_shape=spec.object_shape,
                object_color=spec.object_color,
                output_dir=output_dir,
                sam3_script=Path(sam3_script),
                python_executable=python_executable,
                object_id=spec.object_id,
                fps=fps,
                prompt_frame_idx=adjusted_prompt_frame_idx,
                checkpoint=checkpoint,
                no_offload_video_to_cpu=no_offload_video_to_cpu,
                frames_dir_override=frames_dir_override,
                object_interval=interval,
            )
            metadata = json.loads(plan.metadata_path.read_text())
            metadata["prompt"]["object_material"] = spec.object_material
            metadata["prompt"]["text"] = spec.prompt_text
            metadata["prompt"]["source"] = "stage1_actions"
            metadata["prompt"]["evidence_count"] = spec.evidence_count
            metadata["prompt"]["active_intervals"] = spec.active_intervals
            metadata["command"][metadata["command"].index("--text") + 1] = spec.prompt_text
            plan.command[plan.command.index("--text") + 1] = spec.prompt_text
            plan.prompt_text = spec.prompt_text
            plan.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            plans.append(plan)
    return plans


def resolve_hand_clip_frames_dir(hand_window_dir: Path) -> Path:
    hand_window_dir = Path(hand_window_dir)
    candidates = [
        hand_window_dir / "frames",
        hand_window_dir / "hand/hamer_raw/frames",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing hand clip frames directory. Searched: {searched}")


def combine_sam3_object_tracks(
    *,
    object_dirs: list[Path],
    frames_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    import cv2
    import numpy as np

    object_dirs = [Path(path) for path in object_dirs if Path(path).is_dir()]
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    indexed_dir = output_dir / "object_id_masks"
    overlays_dir = output_dir / "overlays"
    indexed_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = _frame_paths(frames_dir)
    objects = [{"index": idx + 1, "object_id": path.name, "source_dir": str(path)} for idx, path in enumerate(object_dirs)]
    colors = [
        np.asarray([30, 144, 255], dtype=np.float32),
        np.asarray([60, 220, 120], dtype=np.float32),
        np.asarray([255, 170, 40], dtype=np.float32),
        np.asarray([220, 80, 220], dtype=np.float32),
    ]
    for frame_idx, frame_path in enumerate(frame_paths):
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            continue
        indexed = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        overlay = frame_bgr.astype(np.float32)
        for obj_idx, obj_dir in enumerate(object_dirs, start=1):
            mask_path = obj_dir / "masks" / f"frame_{frame_idx:06d}_masks" / f"{obj_dir.name}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            active = mask > 0
            indexed[active] = obj_idx
            color = colors[(obj_idx - 1) % len(colors)]
            overlay[active] = overlay[active] * 0.45 + color * 0.55
        cv2.imwrite(str(indexed_dir / f"frame_{frame_idx:06d}_object_ids.png"), indexed)
        cv2.imwrite(str(overlays_dir / f"frame_{frame_idx:06d}.png"), overlay.astype(np.uint8))
    summary = {
        "schema_version": "1.0",
        "frames_dir": str(frames_dir),
        "output_dir": str(output_dir),
        "objects": objects,
        "frame_count": len(frame_paths),
    }
    summary_json = output_dir / "object_tracks_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "indexed_masks_dir": str(indexed_dir),
        "overlays_dir": str(overlays_dir),
        "summary_json": str(summary_json),
    }


def _frame_paths(frames_dir: Path) -> list[Path]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [path for path in Path(frames_dir).iterdir() if path.suffix.lower() in image_exts]
    try:
        return sorted(paths, key=lambda path: int(path.stem.split("_")[-1]))
    except ValueError:
        return sorted(paths)
