from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def final_segment_name(segment_dir: Path) -> str:
    return segment_dir.name


def clip_stage1_frames(stage1_payload: dict[str, Any], start_s: int, end_s: int) -> dict[str, Any]:
    frames = [
        frame
        for frame in stage1_payload.get("frames", [])
        if start_s <= int(frame.get("t", -1)) <= end_s
    ]
    return {
        "source_video": stage1_payload.get("source_video"),
        "source_range_s": {"start": start_s, "end": end_s},
        "frame_count": len(frames),
        "frames": frames,
    }


def make_window_manifest(window: dict[str, Any], fps: float) -> dict[str, Any]:
    frame_range = window["segment_frame_range"]
    segment_start = int(frame_range["start"])
    segment_end = int(frame_range["end"])
    return {
        "window_id": window["window_id"],
        "window_type": window.get("window_type"),
        "source_segment_id": window.get("source_segment_id"),
        "time_s": {"start": window.get("start"), "end": window.get("end")},
        "core_range_s": window.get("core_range"),
        "frames": {
            "segment_start": segment_start,
            "segment_end": segment_end,
            "local_start": 0,
            "local_end": segment_end - segment_start,
            "num_frames": int(frame_range.get("num_frames", segment_end - segment_start + 1)),
            "fps": fps,
        },
        "hands_required": window.get("hands_required", []),
        "actions": window.get("actions", []),
        "objects": window.get("objects", []),
        "priority": window.get("priority"),
        "semantic_label": window.get("semantic_label"),
        "reason": window.get("reason"),
    }


def copy_path(src: Path, dst: Path, overwrite: bool = False) -> None:
    if not src.exists():
        return
    if dst.exists():
        if not overwrite:
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        ignore = shutil.ignore_patterns("__pycache__", "*.tmp")
        shutil.copytree(src, dst, ignore=ignore)
    else:
        shutil.copy2(src, dst)


def encode_images_to_video(image_paths: Iterable[Path], output_path: Path, fps: float) -> int:
    paths = [path for path in image_paths if path.exists()]
    if not paths:
        return 0
    first = cv2.imread(str(paths[0]))
    if first is None:
        return 0
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    count = 0
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    return count


def frame_paths(directory: Path, pattern: str, start_frame: int, end_frame: int) -> list[Path]:
    return [directory / pattern.format(idx=idx) for idx in range(start_frame, end_frame + 1)]


def extract_video_window(video_path: Path, output_path: Path, start_frame: int, end_frame: int, fps: float) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    count = 0
    frame_idx = start_frame
    try:
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
            frame_idx += 1
    finally:
        writer.release()
        cap.release()
    return count


def object_track_dirs(sam3_root: Path, window_id: str) -> list[Path]:
    root = sam3_root / window_id
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def overlay_object_masks_on_hands(
    hand_visual_dir: Path,
    hand_visual_pattern: str,
    object_dirs: list[Path],
    output_path: Path,
    segment_start: int,
    segment_end: int,
    fps: float,
) -> int:
    first = None
    for first_path in frame_paths(hand_visual_dir, hand_visual_pattern, segment_start, segment_end):
        if first_path.exists():
            first = cv2.imread(str(first_path))
            if first is not None:
                break
    if first is None:
        return 0
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    colors = [
        np.array([30, 144, 255], dtype=np.float32),
        np.array([80, 220, 80], dtype=np.float32),
        np.array([220, 80, 220], dtype=np.float32),
        np.array([0, 220, 220], dtype=np.float32),
    ]
    count = 0
    try:
        for segment_frame in range(segment_start, segment_end + 1):
            frame_path = hand_visual_dir / hand_visual_pattern.format(idx=segment_frame)
            if not frame_path.exists():
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            local_frame = segment_frame - segment_start
            blended = frame.astype(np.float32)
            for obj_idx, obj_dir in enumerate(object_dirs):
                obj_id = obj_dir.name
                mask_path = obj_dir / "masks" / f"frame_{local_frame:06d}_masks" / f"{obj_id}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                active = mask > 0
                color = colors[obj_idx % len(colors)]
                blended[active] = blended[active] * 0.55 + color * 0.45
            writer.write(np.clip(blended, 0, 255).astype(np.uint8))
            count += 1
    finally:
        writer.release()
    return count


def resolve_hand_visual_source(hamer_dir: Path) -> tuple[Path, str, str]:
    candidates = [
        (
            "refined_temporal_gate",
            hamer_dir / "refined_temporal_gate" / "temporal_annotated",
            "frame_{idx:06d}_temporal.jpg",
        ),
        (
            "refined_temporal_size_extreme_filter",
            hamer_dir / "refined_temporal_size_extreme_filter" / "temporal_annotated",
            "frame_{idx:06d}_temporal.jpg",
        ),
        (
            "refined_temporal_jump100",
            hamer_dir / "refined_temporal_jump100" / "temporal_annotated",
            "frame_{idx:06d}_temporal.jpg",
        ),
        (
            "refined_temporal",
            hamer_dir / "refined_temporal" / "temporal_annotated",
            "frame_{idx:06d}_temporal.jpg",
        ),
        (
            "refined_annotated",
            hamer_dir / "refined_annotated",
            "frame_{idx:06d}_refined.jpg",
        ),
        (
            "hamer_overlay",
            hamer_dir / "hamer_overlay",
            "frame_{idx:06d}_hamer_overlay.jpg",
        ),
    ]
    for label, directory, pattern in candidates:
        if directory.exists() and any(directory.glob(pattern.format(idx=0).replace("000000", "*"))):
            return directory, pattern, label
    return hamer_dir / "hamer_overlay", "frame_{idx:06d}_hamer_overlay.jpg", "hamer_overlay"


def find_segment_dir(segments_root: Path, segment_id: str) -> Path | None:
    matches = sorted(path for path in segments_root.iterdir() if path.is_dir() and path.name.startswith(f"{segment_id}_"))
    return matches[0] if matches else None


def find_hamer_window_dir(segment_processed: Path, window_id: str) -> Path | None:
    new_layout = segment_processed / "hand" / "hamer_raw"
    if new_layout.exists():
        return new_layout

    root = segment_processed / "hamer_windows_left_v1"
    if not root.exists():
        return None
    matches = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(f"{window_id}_"))
    return matches[0] if matches else None


def export_session_final_data(
    session_dir: Path,
    output_root: Path,
    overwrite: bool = False,
    fps: float = 30.0,
) -> dict[str, Any]:
    session_dir = session_dir.resolve()
    output_session = output_root.resolve() / session_dir.name
    if output_session.exists() and overwrite:
        shutil.rmtree(output_session)
    output_session.mkdir(parents=True, exist_ok=True)

    for name in ("annotation", "audio", "calibrations", "imu", "timestamps", "videos"):
        copy_path(session_dir / name, output_session / ("raw" if name != "annotation" else name) / name if name != "annotation" else output_session / name, overwrite=overwrite)

    stage1_path = session_dir / "annotation" / "stage1_actions_left_v1.json"
    stage1 = read_json(stage1_path) if stage1_path.exists() else {"frames": []}
    windows_path = session_dir / "annotation" / "hand_estimation_windows_left_v1.json"
    windows_payload = read_json(windows_path)
    windows = windows_payload.get("windows", [])
    segments_root = session_dir / "segments"

    summary: dict[str, Any] = {
        "source_session": str(session_dir),
        "output_session": str(output_session),
        "segments": [],
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        grouped.setdefault(window["source_segment_id"], []).append(window)

    for segment_id, segment_windows in sorted(grouped.items()):
        src_segment = find_segment_dir(segments_root, segment_id)
        if src_segment is None:
            continue
        segment_json = read_json(src_segment / "segment.json")
        dst_segment = output_session / "segments" / final_segment_name(src_segment)
        copy_path(src_segment / "segment.json", dst_segment / "segment.json", overwrite=overwrite)
        copy_path(src_segment / "segment_summary.json", dst_segment / "segment_summary.json", overwrite=overwrite)
        write_json(
            dst_segment / "annotation.json",
            clip_stage1_frames(stage1, int(segment_json["start"]), int(segment_json["end"])),
        )

        processed = src_segment / "rectified"
        stereo_dir = dst_segment / "stereo"
        copy_path(processed / "videos" / "left_rectified.mp4", stereo_dir / "left_rectified.mp4", overwrite=overwrite)
        copy_path(processed / "videos" / "right_rectified.mp4", stereo_dir / "right_rectified.mp4", overwrite=overwrite)
        copy_path(processed / "dataset_manifest" / "dataset_manifest.json", stereo_dir / "dataset_manifest.json", overwrite=overwrite)
        copy_path(processed / "calibrations", stereo_dir / "calibrations", overwrite=overwrite)

        segment_summary = {
            "segment_id": segment_id,
            "segment_dir": str(dst_segment),
            "hand_clips": [],
        }
        for window in segment_windows:
            hamer_dir = find_hamer_window_dir(processed, window["window_id"])
            if hamer_dir is None:
                continue
            manifest = make_window_manifest(window, fps=fps)
            frame_info = manifest["frames"]
            segment_start = int(frame_info["segment_start"])
            segment_end = int(frame_info["segment_end"])

            clip_dir = dst_segment / "hand_clips" / hamer_dir.name
            videos_dir = clip_dir / "videos"
            data_dir = clip_dir / "data"
            write_json(clip_dir / "window.json", manifest)
            write_json(clip_dir / "annotation.json", clip_stage1_frames(stage1, int(window["start"]), int(window["end"])))
            copy_path(hamer_dir / "ego_hand_candidates.json", data_dir / "ego_hand_candidates.json", overwrite=overwrite)
            copy_path(hamer_dir / "ego_hand_candidates_refined.json", data_dir / "ego_hand_candidates_refined.json", overwrite=overwrite)
            hand_visual_dir, hand_visual_pattern, hand_visual_label = resolve_hand_visual_source(hamer_dir)

            rgb_frames = extract_video_window(
                processed / "videos" / "left_rectified.mp4",
                videos_dir / "rgb.mp4",
                segment_start,
                segment_end,
                fps,
            )
            depth_frames = encode_images_to_video(
                (
                    processed / "foundation_depth" / "vis" / f"frame_{idx:06d}_depth_vis.jpg"
                    for idx in range(segment_start, segment_end + 1)
                ),
                videos_dir / "depth_vis.mp4",
                fps,
            )
            hamer_frames = encode_images_to_video(
                frame_paths(hamer_dir / "hamer_overlay", "frame_{idx:06d}_hamer_overlay.jpg", segment_start, segment_end),
                videos_dir / "hamer_overlay.mp4",
                fps,
            )
            hand_final_frames = encode_images_to_video(
                frame_paths(hand_visual_dir, hand_visual_pattern, segment_start, segment_end),
                videos_dir / "hand_final.mp4",
                fps,
            )

            sam3_sources = [
                processed / "objects",
                hamer_dir / "objects",
                processed / "objects" / "sam3_runs",
                hamer_dir / "objects" / "sam3_runs",
                processed / "sam3_object_tracks",
                hamer_dir / "sam3_object_tracks",
            ]
            object_dirs = []
            for sam3_src in sam3_sources:
                object_dirs.extend(object_track_dirs(sam3_src, window["window_id"]))
            deduped_object_dirs = []
            seen_object_ids = set()
            for obj_dir in object_dirs:
                if obj_dir.name in seen_object_ids:
                    continue
                seen_object_ids.add(obj_dir.name)
                deduped_object_dirs.append(obj_dir)
            object_dirs = deduped_object_dirs

            object_video = None
            for obj_dir in object_dirs:
                for tracked in obj_dir.glob("tracked_*.mp4"):
                    copy_path(tracked, videos_dir / f"object_{obj_dir.name}.mp4", overwrite=overwrite)
                    object_video = videos_dir / f"object_{obj_dir.name}.mp4"
                    break
            for sam3_src in sam3_sources:
                copy_path(sam3_src / window["window_id"] / "sam3_object_tracks_summary.json", data_dir / "object_tracks_summary.json", overwrite=overwrite)
            hand_object_frames = overlay_object_masks_on_hands(
                hand_visual_dir,
                hand_visual_pattern,
                object_dirs,
                videos_dir / "hand_object_overlay.mp4",
                segment_start,
                segment_end,
                fps,
            )

            clip_summary = {
                "window_id": window["window_id"],
                "clip_dir": str(clip_dir),
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames,
                "hamer_frames": hamer_frames,
                "hand_final_frames": hand_final_frames,
                "hand_visual_source": hand_visual_label,
                "hand_object_frames": hand_object_frames,
                "object_count": len(object_dirs),
                "object_video_sample": str(object_video) if object_video else None,
            }
            write_json(data_dir / "export_summary.json", clip_summary)
            segment_summary["hand_clips"].append(clip_summary)
        write_json(dst_segment / "segment_export_summary.json", segment_summary)
        summary["segments"].append(segment_summary)

    write_json(output_session / "session_export_summary.json", summary)
    return summary
