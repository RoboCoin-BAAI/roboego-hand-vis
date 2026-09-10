#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


JOINT_NAMES = [
    "wrist",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_tip",
    "index_1",
    "index_2",
    "index_3",
    "index_tip",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_tip",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_tip",
    "little_1",
    "little_2",
    "little_3",
    "little_tip",
]
AXES = ("x", "y", "z")
VIDEO_NAMES = ("left", "right", "bleft", "bright", "left_rectified", "right_rectified")


def export_auto(*, session: Path, episode_index: int = 0, overwrite: bool = False) -> dict[str, Any]:
    units = discover_export_units(session)
    results = []
    for offset, unit in enumerate(units):
        results.append(
            export_session_hand_parquet(
                session=unit,
                output_root=unit,
                episode_index=episode_index + offset,
                overwrite=overwrite,
            )
        )
    return {
        "input": str(Path(session).expanduser().resolve()),
        "mode": "hand_windows" if _has_complete_hand_windows(Path(session).expanduser().resolve()) else "session",
        "export_count": len(results),
        "results": results,
    }


def discover_export_units(session: Path) -> list[Path]:
    session = Path(session).expanduser().resolve()
    windows = _complete_hand_windows(session)
    return windows if windows else [session]


def _has_complete_hand_windows(session: Path) -> bool:
    return bool(_complete_hand_windows(session))


def _complete_hand_windows(session: Path) -> list[Path]:
    root = session / "hand_windows"
    if not root.is_dir():
        return []
    return sorted(window for window in root.iterdir() if window.is_dir() and _is_complete_export_unit(window))


def _is_complete_export_unit(unit: Path) -> bool:
    required = [
        unit / "hand/final_hand_result/smoothed_joints21.json",
        unit / "dataset_manifest/dataset_manifest.json",
        unit / "videos/left.mp4",
        unit / "videos/right.mp4",
        unit / "videos/left_rectified.mp4",
        unit / "videos/right_rectified.mp4",
    ]
    return all(path.is_file() for path in required)


def export_session_hand_parquet(*, session: Path, output_root: Path, episode_index: int = 0, overwrite: bool = False) -> dict[str, Any]:
    session = Path(session).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and overwrite:
        if output_root == session:
            shutil.rmtree(output_root / "data", ignore_errors=True)
            shutil.rmtree(output_root / "meta", ignore_errors=True)
        else:
            shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = session / "dataset_manifest/dataset_manifest.json"
    hand_path = session / "hand/final_hand_result/smoothed_joints21.json"
    if not hand_path.is_file():
        raise FileNotFoundError(f"Missing smoothed hand trajectory: {hand_path}")
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    hand_payload = _read_json(hand_path)
    frames = hand_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Expected frames list in {hand_path}")

    fps = _infer_fps(manifest)
    window = _read_window_metadata(session, frame_count=len(frames))
    source_range = window.get("source_frame_range") if isinstance(window.get("source_frame_range"), dict) else {}
    rows = [
        _frame_to_row(frame=frame, episode_index=episode_index, fps=fps, source_start_frame=source_range.get("start"))
        for frame in frames
    ]
    df = pd.DataFrame(rows)
    parquet_path = data_dir / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)
    task = infer_task_label(session)

    _write_meta_files(
        output_root=output_root,
        session=session,
        manifest=manifest,
        hand_payload=hand_payload,
        hand_path=hand_path,
        parquet_path=parquet_path,
        episode_index=episode_index,
        frame_count=len(df),
        fps=fps,
        task=task,
        window=window,
        stats=_compute_stats(df),
    )
    return {
        "output_root": str(output_root),
        "parquet_path": str(parquet_path),
        "episode_index": int(episode_index),
        "frame_count": int(len(df)),
        "fps": float(fps),
    }


def _frame_to_row(*, frame: dict[str, Any], episode_index: int, fps: float, source_start_frame: int | None) -> dict[str, Any]:
    frame_idx = int(frame.get("frame_idx", 0))
    hands = frame.get("hands") if isinstance(frame.get("hands"), dict) else {}
    left = _hand_array(hands.get("ego_left"))
    right = _hand_array(hands.get("ego_right"))
    row: dict[str, Any] = {
        "episode_index": int(episode_index),
        "frame_index": frame_idx,
        "source_frame_index": int(source_start_frame) + frame_idx if source_start_frame is not None else frame_idx,
        "timestamp": frame_idx / float(fps) if fps > 0 else math.nan,
        "task_index": 0,
        "observation.left_hand_joints_valid": bool(left is not None),
        "observation.right_hand_joints_valid": bool(right is not None),
    }
    _add_hand_columns(row, "left", left)
    _add_hand_columns(row, "right", right)
    return row


def _hand_array(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        return None
    joints = payload.get("joints21_camera_opencv")
    if joints is None:
        return None
    array = np.asarray(joints, dtype=np.float32)
    if array.shape != (21, 3) or not np.all(np.isfinite(array)):
        return None
    return array


def _add_hand_columns(row: dict[str, Any], side: str, joints: np.ndarray | None) -> None:
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        for axis_idx, axis in enumerate(AXES):
            value = math.nan if joints is None else float(joints[joint_idx, axis_idx])
            row[f"observation.state.{side}.{joint_name}.{axis}"] = value


def _write_meta_files(
    *,
    output_root: Path,
    session: Path,
    manifest: dict[str, Any],
    hand_payload: dict[str, Any],
    hand_path: Path,
    parquet_path: Path,
    episode_index: int,
    frame_count: int,
    fps: float,
    task: str,
    window: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    meta_dir = output_root / "meta"
    path_root = _metadata_path_root(session)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [126],
            "names": _state_feature_names(),
        },
        "observation.left_hand_joints_valid": {"dtype": "bool", "shape": [1], "names": ["valid"]},
        "observation.right_hand_joints_valid": {"dtype": "bool", "shape": [1], "names": ["valid"]},
    }
    videos = _video_metadata(session, path_root)
    for name, video in videos.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [video["height"], video["width"], video["channels"]],
            "names": ["height", "width", "channels"],
            "path": video["path"],
            "info": {
                "video.height": video["height"],
                "video.width": video["width"],
                "video.codec": video["codec"],
                "video.pix_fmt": video["pix_fmt"],
                "video.is_depth_map": False,
                "video.fps": video["fps"],
                "video.channels": video["channels"],
                "video.frame_count": video["frame_count"],
                "video.duration_s": video["duration_s"],
                "has_audio": video["has_audio"],
            },
        }
    info = {
        "codebase_version": "roboego_hand_parquet_v1",
        "robot_type": "ego_hand",
        "fps": float(fps),
        "total_episodes": 1,
        "total_frames": int(frame_count),
        "source_session": _source_session_name(session),
        "features": features,
        "coordinate_frame": hand_payload.get("coordinate_frame", "camera_opencv"),
        "unit": hand_payload.get("unit", "meter"),
        "invalid_hand_value": "NaN",
        "depth": _depth_paths(session, path_root),
        "audio": _audio_paths(session, path_root),
        "annotations": _annotation_paths(session, path_root),
        "window": window,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    episode = {
        "episode_index": int(episode_index),
        "length": int(frame_count),
        "tasks": [task],
        "data_path": str(parquet_path.relative_to(output_root)),
    }
    (meta_dir / "episodes.jsonl").write_text(json.dumps(episode, ensure_ascii=False) + "\n", encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (meta_dir / "roboego_sources.json").write_text(
        json.dumps(_source_metadata(session, manifest, hand_path, path_root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_metadata(session: Path, manifest: dict[str, Any], hand_path: Path, path_root: Path) -> dict[str, Any]:
    return {
        "source_session": _source_session_name(session),
        "source_manifest": _relative_path(session / "dataset_manifest/dataset_manifest.json", path_root),
        "source_hand": _relative_path(hand_path, path_root),
        "manifest_videos": manifest.get("videos", {}),
        "depth": _depth_paths(session, path_root),
        "audio": _audio_paths(session, path_root),
        "annotations": _annotation_paths(session, path_root),
        "window": _read_window_metadata(session, frame_count=0),
    }


def _metadata_path_root(session: Path) -> Path:
    source_root = _media_source_root(session)
    if source_root != session:
        return source_root.parent
    return session.parent


def _source_session_name(session: Path) -> str:
    source_root = _media_source_root(session)
    return source_root.name


def infer_task_label(session: Path) -> str:
    window_task = _task_from_window(Path(session) / "annotation/window.json")
    if window_task:
        return window_task
    window_task = _task_from_hand_windows(Path(session) / "annotation/hand_estimation_windows_left_v1.json")
    if window_task:
        return window_task
    return _task_from_session_name(Path(session).name)


def _task_from_window(path: Path) -> str:
    if not path.is_file():
        return ""
    payload = _read_json(path)
    label = str(payload.get("semantic_label") or "").strip()
    actions = _unique_strings(payload.get("actions") or [])
    objects = _unique_strings(payload.get("objects") or [])
    parts = []
    if label:
        parts.append(label)
    if objects:
        parts.append(", ".join(objects[:3]))
    if actions:
        action_text = ", ".join(actions[:6])
        return f"{' '.join(parts)}: {action_text}" if parts else action_text
    return " ".join(parts)


def _task_from_hand_windows(path: Path) -> str:
    if not path.is_file():
        return ""
    payload = _read_json(path)
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        return ""
    labels: list[str] = []
    actions: list[str] = []
    objects: list[str] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        label = str(window.get("semantic_label") or "").strip()
        if label and label not in labels:
            labels.append(label)
        for action in _unique_strings(window.get("actions") or []):
            if action not in actions:
                actions.append(action)
        for obj in _unique_strings(window.get("objects") or []):
            if obj not in objects:
                objects.append(obj)
    parts: list[str] = []
    if labels:
        parts.append("_".join(labels[:2]))
    if objects:
        parts.append(", ".join(objects[:3]))
    if actions:
        action_text = ", ".join(actions[:6])
        return f"{' '.join(parts)}: {action_text}" if parts else action_text
    return " ".join(parts)


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _read_window_metadata(session: Path, *, frame_count: int) -> dict[str, Any]:
    path = Path(session) / "annotation/window.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    source_start = _optional_int(payload.get("source_start_frame"))
    local_range = payload.get("local_frame_range") if isinstance(payload.get("local_frame_range"), dict) else {}
    if source_start is None:
        source_start = _optional_int((payload.get("segment_frame_range") or {}).get("start")) if isinstance(payload.get("segment_frame_range"), dict) else None
    local_start = _optional_int(local_range.get("start")) if isinstance(local_range, dict) else 0
    local_end = _optional_int(local_range.get("end")) if isinstance(local_range, dict) else None
    if local_end is None and frame_count > 0:
        local_end = int(local_start or 0) + int(frame_count) - 1
    source_end = source_start + int(local_end or 0) - int(local_start or 0) if source_start is not None and local_end is not None else None
    num_frames = int(local_end) - int(local_start or 0) + 1 if local_end is not None else None
    source_session = str(payload.get("source_session") or "")
    return {
        "window_id": str(payload.get("window_id") or Path(session).name),
        "source_session": Path(source_session).name if source_session else "",
        "source_frame_range": _compact_range(source_start, source_end, num_frames),
        "local_frame_range": payload.get("local_frame_range") or {},
    }


def _compact_range(start: int | None, end: int | None, num_frames: int | None) -> dict[str, int] | dict[str, Any]:
    if start is None or end is None or num_frames is None:
        return {}
    return {"start": int(start), "end": int(end), "num_frames": int(num_frames)}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _task_from_session_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "session":
        label = "_".join(parts[3:-1]).strip()
        if label:
            return label
    return name


def _video_metadata(session: Path, path_root: Path) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for name in VIDEO_NAMES:
        path = session / "videos" / f"{name}.mp4"
        if not path.exists() and name in {"bleft", "bright"}:
            path = _media_source_root(session) / "videos" / f"{name}.mp4"
        if not path.exists():
            continue
        probed = _probe_video(path)
        videos[name] = {
            "path": _relative_path(path, path_root),
            "codec": probed.get("codec", "unknown"),
            "pix_fmt": probed.get("pix_fmt", "unknown"),
            "width": int(probed.get("width") or 0),
            "height": int(probed.get("height") or 0),
            "fps": float(probed.get("fps") or 0.0),
            "frame_count": int(probed.get("frame_count") or 0),
            "duration_s": float(probed.get("duration_s") or 0.0),
            "channels": 3,
            "has_audio": bool(probed.get("has_audio", False)),
        }
    return videos


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name,pix_fmt,width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {"codec": "unknown", "pix_fmt": "unknown"}
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return {
        "codec": str(video.get("codec_name") or "unknown"),
        "pix_fmt": str(video.get("pix_fmt") or "unknown"),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": _parse_rate(video.get("r_frame_rate")),
        "frame_count": _parse_frame_count(video),
        "duration_s": float(video.get("duration") or 0.0),
        "has_audio": audio,
    }


def _parse_rate(value: Any) -> float:
    text = str(value or "")
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        den = float(denominator)
        return float(numerator) / den if den else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_frame_count(video: dict[str, Any]) -> int:
    if video.get("nb_frames") not in {None, "N/A"}:
        return int(video["nb_frames"])
    fps = _parse_rate(video.get("r_frame_rate"))
    duration = float(video.get("duration") or 0.0)
    return int(round(fps * duration)) if fps > 0 and duration > 0 else 0


def _depth_paths(session: Path, path_root: Path) -> dict[str, dict[str, Any]]:
    archive = session / "depth/fast_foundation_archive"
    paths = {}
    hevc12 = archive / "depth_hevc12.mkv"
    valid_mask = archive / "valid_mask_ffv1.mkv"
    if hevc12.exists():
        probed = _probe_video(hevc12)
        paths["hevc12"] = {
            "path": _relative_path(hevc12, path_root),
            "codec": "h265",
            "codec_name": probed.get("codec", "unknown"),
            "codec_family": "h265/hevc",
            "pix_fmt": probed.get("pix_fmt", "unknown"),
            "width": int(probed.get("width") or 0),
            "height": int(probed.get("height") or 0),
            "fps": float(probed.get("fps") or 0.0),
            "frame_count": int(probed.get("frame_count") or 0),
            "duration_s": float(probed.get("duration_s") or 0.0),
            "is_depth_map": True,
        }
    if valid_mask.exists():
        probed = _probe_video(valid_mask)
        paths["valid_mask"] = {
            "path": _relative_path(valid_mask, path_root),
            "codec": "ffv1",
            "codec_name": probed.get("codec", "unknown"),
            "codec_family": "ffv1",
            "pix_fmt": probed.get("pix_fmt", "unknown"),
            "width": int(probed.get("width") or 0),
            "height": int(probed.get("height") or 0),
            "fps": float(probed.get("fps") or 0.0),
            "frame_count": int(probed.get("frame_count") or 0),
            "duration_s": float(probed.get("duration_s") or 0.0),
            "is_depth_map": True,
        }
    return paths


def _audio_paths(session: Path, path_root: Path) -> dict[str, dict[str, str]]:
    audio_dir = _media_source_root(session) / "audio"
    return {path.stem: {"path": _relative_path(path, path_root)} for path in sorted(audio_dir.glob("*")) if path.is_file()}


def _annotation_paths(session: Path, path_root: Path) -> dict[str, str]:
    annotation_dir = _media_source_root(session) / "annotation"
    stage1 = annotation_dir / "stage1_actions_left_v1_resolved.json"
    return {"stage1_actions": _relative_path(stage1, path_root)} if stage1.is_file() else {}


def _media_source_root(session: Path) -> Path:
    path = _window_source_session_path(session)
    if path is not None:
        return path
    return session


def _window_source_session_path(session: Path) -> Path | None:
    path = Path(session) / "annotation/window.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    source_session = str(payload.get("source_session") or "")
    if not source_session:
        return None
    source_path = Path(source_session).expanduser()
    return source_path if source_path.is_dir() else None


def _relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def _compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    state_cols = [col for col in df.columns if col.startswith("observation.state.")]
    values = df[state_cols].to_numpy(dtype=np.float32)
    valid = np.isfinite(values)
    return {
        "frame_count": int(len(df)),
        "left_valid_frames": int(df["observation.left_hand_joints_valid"].sum()),
        "right_valid_frames": int(df["observation.right_hand_joints_valid"].sum()),
        "state": {
            "finite_count": int(valid.sum()),
            "nan_count": int((~valid).sum()),
            "mean": float(np.nanmean(values)) if valid.any() else None,
            "std": float(np.nanstd(values)) if valid.any() else None,
            "min": float(np.nanmin(values)) if valid.any() else None,
            "max": float(np.nanmax(values)) if valid.any() else None,
        },
    }


def _state_feature_names() -> list[str]:
    return [f"{side}.{joint}.{axis}" for side in ("left", "right") for joint in JOINT_NAMES for axis in AXES]


def _infer_fps(manifest: dict[str, Any]) -> float:
    videos = manifest.get("videos") if isinstance(manifest.get("videos"), dict) else {}
    for name in ("left", "right"):
        video = videos.get(name)
        if isinstance(video, dict) and video.get("fps"):
            return float(video["fps"])
    return 30.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RoboEgo final hand trajectory to a lightweight LeRobot-style parquet layout.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-root", default="", help="Defaults to --session, writing data/ and meta/ directly under the session.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_root:
        result = export_session_hand_parquet(
            session=Path(args.session),
            output_root=Path(args.output_root),
            episode_index=int(args.episode_index),
            overwrite=bool(args.overwrite),
        )
    else:
        result = export_auto(session=Path(args.session), episode_index=int(args.episode_index), overwrite=bool(args.overwrite))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
