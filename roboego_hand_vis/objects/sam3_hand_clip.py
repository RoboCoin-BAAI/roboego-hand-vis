from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
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


@dataclass
class Stage1ObjectSpec:
    object_id: str
    object_name: str
    object_shape: str
    object_color: str
    object_material: str
    prompt_text: str
    evidence_count: int


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
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        for token in part.split():
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
    return " ".join(tokens)


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
    tokens: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        for token in part.split():
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(token)
    return " ".join(tokens)


def safe_object_id(*parts: str) -> str:
    raw = "_".join(part.strip().lower() for part in parts if part and part.strip())
    raw = raw.replace("&", "and")
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "object_0"


def infer_object_specs_from_stage1(stage1_actions: Path, window: dict[str, Any], *, fps: float) -> list[Stage1ObjectSpec]:
    evidence = collect_stage1_evidence(Path(stage1_actions), window, fps=fps)
    grouped: dict[str, dict[str, Any]] = {}
    for frame in evidence:
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
            group_id = safe_object_id(name, color)
            item = grouped.setdefault(
                group_id,
                {
                    "object_id": group_id,
                    "object_name": name,
                    "object_shape": shape,
                    "object_color": color,
                    "object_material": material,
                    "evidence_count": 0,
                },
            )
            item["evidence_count"] += 1
            if len(shape) > len(str(item.get("object_shape", ""))):
                item["object_shape"] = shape
            if len(color) > len(str(item.get("object_color", ""))):
                item["object_color"] = color
            if len(material) > len(str(item.get("object_material", ""))):
                item["object_material"] = material

    if not grouped:
        for name in window.get("objects", []) or []:
            object_name = str(name).strip()
            if not object_name:
                continue
            object_id = safe_object_id(object_name)
            grouped[object_id] = {
                "object_id": object_id,
                "object_name": object_name,
                "object_shape": "",
                "object_color": "",
                "object_material": "",
                "evidence_count": 0,
            }

    specs: list[Stage1ObjectSpec] = []
    for item in grouped.values():
        object_id = safe_object_id(
            str(item["object_name"]),
            str(item["object_color"]),
            str(item["object_shape"]),
        )
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
            )
        )
    specs.sort(key=lambda spec: -spec.evidence_count)
    return specs


def find_window_by_id(windows_json: Path, window_id: str) -> dict[str, Any]:
    payload = json.loads(Path(windows_json).read_text())
    for window in payload.get("windows", []):
        if window.get("window_id") == window_id:
            return window
    raise ValueError(f"Window id {window_id!r} not found in {windows_json}")


def infer_window_id_from_dir(hand_window_dir: Path) -> str:
    name = Path(hand_window_dir).name
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
    return Path(hand_window_dir) / "sam3_object_tracks" / object_id


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
) -> Sam3HandClipPlan:
    hand_window_dir = Path(hand_window_dir)
    frames_dir = hand_window_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing hand clip frames directory: {frames_dir}")
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
) -> list[Sam3HandClipPlan]:
    specs = infer_object_specs_from_stage1(Path(stage1_actions), window, fps=fps)
    if not specs:
        raise ValueError("No object specs found from stage1 annotations for this hand window.")
    window_id = str(window.get("window_id") or infer_window_id_from_dir(Path(hand_window_dir)))
    root = Path(output_root) if output_root is not None else Path(hand_window_dir) / "sam3_object_tracks"
    plans: list[Sam3HandClipPlan] = []
    for spec in specs:
        output_dir = root / window_id / spec.object_id
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
            prompt_frame_idx=prompt_frame_idx,
            checkpoint=checkpoint,
            no_offload_video_to_cpu=no_offload_video_to_cpu,
        )
        metadata = json.loads(plan.metadata_path.read_text())
        metadata["prompt"]["object_material"] = spec.object_material
        metadata["prompt"]["text"] = spec.prompt_text
        metadata["prompt"]["source"] = "stage1_actions"
        metadata["prompt"]["evidence_count"] = spec.evidence_count
        metadata["command"][metadata["command"].index("--text") + 1] = spec.prompt_text
        plan.command[plan.command.index("--text") + 1] = spec.prompt_text
        plan.prompt_text = spec.prompt_text
        plan.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        plans.append(plan)
    return plans
