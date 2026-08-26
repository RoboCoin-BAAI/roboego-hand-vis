from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


SIDES = ("ego_left", "ego_right")


def export_final_hand_result(
    *,
    tracks_json: Path,
    output_dir: Path,
    depth_anchored_json: Path | None = None,
    depth_anchored_npz: Path | None = None,
    depth_anchored_summary_json: Path | None = None,
    copy_depth_anchored: bool = True,
    copy_meshes: bool = True,
) -> dict[str, Any]:
    payload = json.loads(Path(tracks_json).expanduser().read_text(encoding="utf-8"))
    output_dir = Path(output_dir).expanduser()
    meshes_dir = output_dir / "meshes"
    if copy_meshes:
        meshes_dir.mkdir(parents=True, exist_ok=True)

    hands: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    copied_mesh_count = 0
    missing_mesh_count = 0
    copied_depth_artifacts: dict[str, str] = {}
    for frame in payload.get("frames", []) or []:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", 0))
        frame_hands: dict[str, Any] = {}
        source_hands = _source_hands_by_side(frame)
        for side in SIDES:
            obs = source_hands.get(side)
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
        frames.append({"frame_idx": frame_idx, "hands": frame_hands})

    if copy_depth_anchored:
        copied_depth_artifacts = _copy_depth_anchored_artifacts(
            output_dir=output_dir,
            depth_anchored_json=depth_anchored_json,
            depth_anchored_npz=depth_anchored_npz,
            depth_anchored_summary_json=depth_anchored_summary_json,
        )

    summary = {
        "stage": "final_hand_result_v1",
        "source_tracks_json": str(tracks_json),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "hand_count": len(hands),
        "copied_mesh_count": copied_mesh_count,
        "missing_mesh_count": missing_mesh_count,
        "depth_anchored_outputs": copied_depth_artifacts,
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


def _source_hands_by_side(frame: dict[str, Any]) -> dict[str, Any]:
    hands = frame.get("hands")
    if isinstance(hands, dict):
        return hands
    out: dict[str, Any] = {}
    candidates = frame.get("candidates")
    if not isinstance(candidates, list):
        return out
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        side = candidate.get("side") or candidate.get("status")
        if side in SIDES and side not in out:
            out[side] = candidate
    return out


def _copy_depth_anchored_artifacts(
    *,
    output_dir: Path,
    depth_anchored_json: Path | None,
    depth_anchored_npz: Path | None,
    depth_anchored_summary_json: Path | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for label, source, dest_name in (
        ("joints21_depth_anchored_json", depth_anchored_json, "joints21_depth_anchored.json"),
        ("joints21_depth_anchored_npz", depth_anchored_npz, "joints21_depth_anchored.npz"),
        ("joints21_depth_anchored_summary_json", depth_anchored_summary_json, "joints21_depth_anchored_summary.json"),
    ):
        if source is None:
            continue
        source = Path(source).expanduser()
        if not source.is_file():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / dest_name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        out[label] = str(dest)
    return out
