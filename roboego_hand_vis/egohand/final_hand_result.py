from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SIDES = ("ego_left", "ego_right")


def export_final_hand_result(
    *,
    tracks_json: Path,
    output_dir: Path,
    temporal_annotated_dir: Path | None = None,
    hamer_overlay_dir: Path | None = None,
    source_frames_dir: Path | None = None,
    copy_meshes: bool = True,
    copy_visuals: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(tracks_json).expanduser().read_text(encoding="utf-8"))
    output_dir = Path(output_dir).expanduser()
    meshes_dir = output_dir / "meshes"
    annotated_dir = output_dir / "annotated"
    if copy_meshes:
        meshes_dir.mkdir(parents=True, exist_ok=True)
    if copy_visuals:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    hands: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    copied_mesh_count = 0
    missing_mesh_count = 0
    copied_visual_count = 0
    copied_mesh_overlay_count = 0
    copied_filtered_mesh_overlay_count = 0
    copied_temporal_visual_count = 0
    for frame in payload.get("frames", []) or []:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", 0))
        frame_hands: dict[str, Any] = {}
        for side in SIDES:
            obs = (frame.get("hands") or {}).get(side)
            if not isinstance(obs, dict):
                frame_hands[side] = None
                continue
            item = _final_hand_item(obs, frame_idx=frame_idx, side=side)
            source_mesh = Path(str(item["source_mesh_obj"])).expanduser() if item.get("source_mesh_obj") else None
            if source_mesh is not None and source_mesh.exists() and copy_meshes:
                mesh_name = f"frame_{frame_idx:06d}_{side}.obj"
                mesh_path = meshes_dir / mesh_name
                shutil.copy2(source_mesh, mesh_path)
                item["mesh_obj"] = str(mesh_path)
                copied_mesh_count += 1
            elif source_mesh is not None:
                missing_mesh_count += 1
            hands.append(item)
            frame_hands[side] = item
        if copy_visuals:
            copied_kind = _copy_final_visual(
                frame_idx=frame_idx,
                frame_hands=frame_hands,
                annotated_dir=annotated_dir,
                source_frames_dir=source_frames_dir,
                hamer_overlay_dir=hamer_overlay_dir,
                temporal_annotated_dir=temporal_annotated_dir,
            )
            if copied_kind is not None:
                copied_visual_count += 1
                if copied_kind == "mesh_overlay":
                    copied_mesh_overlay_count += 1
                elif copied_kind == "filtered_mesh_overlay":
                    copied_filtered_mesh_overlay_count += 1
                elif copied_kind == "temporal_gate":
                    copied_temporal_visual_count += 1
        frames.append({"frame_idx": frame_idx, "hands": frame_hands})

    summary = {
        "stage": "final_hand_result_v1",
        "source_tracks_json": str(tracks_json),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "hand_count": len(hands),
        "copied_mesh_count": copied_mesh_count,
        "missing_mesh_count": missing_mesh_count,
        "copied_visual_count": copied_visual_count,
        "copied_mesh_overlay_count": copied_mesh_overlay_count,
        "copied_filtered_mesh_overlay_count": copied_filtered_mesh_overlay_count,
        "copied_temporal_visual_count": copied_temporal_visual_count,
        "source_summary": payload.get("summary", {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "final_hand_tracks.json"
    summary_path = output_dir / "final_hand_summary.json"
    tracks_path.write_text(
        json.dumps(
            {
                "stage": "final_hand_result_v1",
                "source_tracks_json": str(tracks_json),
                "hands": hands,
                "frames": frames,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        **summary,
        "tracks_json": str(tracks_path),
        "summary_json": str(summary_path),
    }


def _final_hand_item(obs: dict[str, Any], *, frame_idx: int, side: str) -> dict[str, Any]:
    source_mesh = obs.get("mesh_obj")
    return {
        "frame_idx": frame_idx,
        "side": side,
        "candidate_id": obs.get("candidate_id"),
        "status": obs.get("status"),
        "source": obs.get("source"),
        "bbox_xyxy": obs.get("bbox_xyxy"),
        "bbox_center_xy": obs.get("bbox_center_xy"),
        "confidence": obs.get("confidence"),
        "temporal_quality": obs.get("temporal_quality"),
        "anomaly_flags": obs.get("anomaly_flags", []),
        "mesh_obj": source_mesh,
        "source_mesh_obj": source_mesh,
        "source_hand": obs.get("source_hand"),
    }


def _copy_final_visual(
    *,
    frame_idx: int,
    frame_hands: dict[str, Any],
    annotated_dir: Path,
    source_frames_dir: Path | None,
    hamer_overlay_dir: Path | None,
    temporal_annotated_dir: Path | None,
) -> str | None:
    if hamer_overlay_dir is not None:
        overlay_path = Path(hamer_overlay_dir) / f"frame_{frame_idx:06d}_hamer_overlay.jpg"
        raw_path = Path(source_frames_dir) / f"frame_{frame_idx:06d}.jpg" if source_frames_dir is not None else None
        if overlay_path.exists() and raw_path is not None and raw_path.exists():
            if _write_filtered_mesh_overlay(
                frame_hands=frame_hands,
                raw_path=raw_path,
                overlay_path=overlay_path,
                output_path=annotated_dir / f"frame_{frame_idx:06d}_final.jpg",
            ):
                return "filtered_mesh_overlay"
        if overlay_path.exists() and _has_kept_hand(frame_hands):
            shutil.copy2(overlay_path, annotated_dir / f"frame_{frame_idx:06d}_final.jpg")
            return "mesh_overlay"
    if temporal_annotated_dir is not None:
        source = Path(temporal_annotated_dir) / f"frame_{frame_idx:06d}_temporal.jpg"
        if source.exists():
            shutil.copy2(source, annotated_dir / f"frame_{frame_idx:06d}_final.jpg")
            return "temporal_gate"
    return None


def _write_filtered_mesh_overlay(
    *,
    frame_hands: dict[str, Any],
    raw_path: Path,
    overlay_path: Path,
    output_path: Path,
) -> bool:
    raw = cv2.imread(str(raw_path))
    overlay = cv2.imread(str(overlay_path))
    if raw is None or overlay is None or raw.shape != overlay.shape:
        return False
    mask = np.zeros(raw.shape[:2], dtype=np.uint8)
    for obs in frame_hands.values():
        if not isinstance(obs, dict):
            continue
        bbox = obs.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = _expanded_bbox_xyxy(bbox, raw.shape[1], raw.shape[0], margin_ratio=0.25)
        mask[y1:y2, x1:x2] = 255
    if not np.any(mask):
        cv2.imwrite(str(output_path), raw)
        return True
    changed = np.any(np.abs(overlay.astype(np.int16) - raw.astype(np.int16)) > 5, axis=2)
    keep = changed & (mask > 0)
    out = raw.copy()
    out[keep] = overlay[keep]
    cv2.imwrite(str(output_path), out)
    return True


def _expanded_bbox_xyxy(bbox: list[float], width: int, height: int, *, margin_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    margin_x = max(1.0, (x2 - x1) * margin_ratio)
    margin_y = max(1.0, (y2 - y1) * margin_ratio)
    return (
        max(0, min(width, int(np.floor(x1 - margin_x)))),
        max(0, min(height, int(np.floor(y1 - margin_y)))),
        max(0, min(width, int(np.ceil(x2 + margin_x)))),
        max(0, min(height, int(np.ceil(y2 + margin_y)))),
    )


def _has_kept_hand(frame_hands: dict[str, Any]) -> bool:
    return any(isinstance(obs, dict) for obs in frame_hands.values())
