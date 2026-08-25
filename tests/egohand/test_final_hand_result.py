import json
from pathlib import Path

import cv2
import numpy as np

from roboego_hand_vis.egohand.final_hand_result import export_final_hand_result


def test_export_final_hand_result_copies_only_temporal_gate_kept_meshes(tmp_path: Path):
    source_mesh_dir = tmp_path / "hamer/meshes"
    source_vis_dir = tmp_path / "hamer/refined_temporal_gate/temporal_annotated"
    source_overlay_dir = tmp_path / "hamer/hamer_overlay"
    source_mesh_dir.mkdir(parents=True)
    source_vis_dir.mkdir(parents=True)
    source_overlay_dir.mkdir(parents=True)
    left_mesh = source_mesh_dir / "frame_000000_candidate_01.obj"
    dropped_mesh = source_mesh_dir / "frame_000001_candidate_02.obj"
    left_mesh.write_text("left mesh", encoding="utf-8")
    dropped_mesh.write_text("dropped mesh", encoding="utf-8")
    (source_vis_dir / "frame_000000_temporal.jpg").write_bytes(b"vis0")
    (source_vis_dir / "frame_000001_temporal.jpg").write_bytes(b"vis1")
    (source_overlay_dir / "frame_000000_hamer_overlay.jpg").write_bytes(b"mesh0")

    tracks_json = tmp_path / "hamer/refined_temporal_gate/refined_temporal_gate_tracks.json"
    tracks_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 0,
                        "hands": {
                            "ego_left": {
                                "frame_idx": 0,
                                "side": "ego_left",
                                "candidate_id": 1,
                                "mesh_obj": str(left_mesh),
                                "bbox_xyxy": [1, 2, 3, 4],
                            },
                            "ego_right": None,
                        },
                        "dropped_hands": {},
                    },
                    {
                        "frame_idx": 1,
                        "hands": {"ego_left": None, "ego_right": None},
                        "dropped_hands": {
                            "ego_right": {
                                "frame_idx": 1,
                                "side": "ego_right",
                                "candidate_id": 2,
                                "mesh_obj": str(dropped_mesh),
                            }
                        },
                    },
                ],
                "summary": {
                    "ego_left": {"normal": 1, "missing": 1, "dropped": 0},
                    "ego_right": {"normal": 0, "missing": 2, "dropped": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "final_hand_result"
    result = export_final_hand_result(
        tracks_json=tracks_json,
        output_dir=output_dir,
        temporal_annotated_dir=source_vis_dir,
        hamer_overlay_dir=source_overlay_dir,
    )

    final_tracks = json.loads((output_dir / "final_hand_tracks.json").read_text(encoding="utf-8"))
    assert result["hand_count"] == 1
    assert len(final_tracks["hands"]) == 1
    assert final_tracks["hands"][0]["mesh_obj"].endswith("meshes/frame_000000_ego_left.obj")
    assert final_tracks["hands"][0]["source_mesh_obj"] == str(left_mesh)
    assert (output_dir / "meshes/frame_000000_ego_left.obj").read_text(encoding="utf-8") == "left mesh"
    assert not (output_dir / "meshes/frame_000001_ego_right.obj").exists()
    assert (output_dir / "annotated/frame_000000_final.jpg").read_bytes() == b"mesh0"
    assert result["copied_mesh_overlay_count"] == 1
    assert result["copied_temporal_visual_count"] == 1


def test_export_final_hand_result_masks_mesh_overlay_to_kept_hand_bboxes(tmp_path: Path):
    source_mesh_dir = tmp_path / "hamer/meshes"
    source_frames_dir = tmp_path / "hamer/frames"
    source_overlay_dir = tmp_path / "hamer/hamer_overlay"
    source_mesh_dir.mkdir(parents=True)
    source_frames_dir.mkdir(parents=True)
    source_overlay_dir.mkdir(parents=True)
    kept_mesh = source_mesh_dir / "frame_000000_candidate_01.obj"
    kept_mesh.write_text("kept mesh", encoding="utf-8")

    raw = np.zeros((20, 20, 3), dtype=np.uint8)
    overlay = raw.copy()
    overlay[2:6, 2:6] = [0, 0, 255]
    overlay[12:18, 12:18] = [0, 255, 0]
    cv2.imwrite(str(source_frames_dir / "frame_000000.jpg"), raw)
    cv2.imwrite(str(source_overlay_dir / "frame_000000_hamer_overlay.jpg"), overlay)
    cv2.imwrite(str(source_frames_dir / "frame_000001.jpg"), raw)
    cv2.imwrite(str(source_overlay_dir / "frame_000001_hamer_overlay.jpg"), overlay)

    tracks_json = tmp_path / "hamer/refined_temporal_gate/refined_temporal_gate_tracks.json"
    tracks_json.parent.mkdir(parents=True)
    tracks_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": 0,
                        "hands": {
                            "ego_left": {
                                "frame_idx": 0,
                                "side": "ego_left",
                                "mesh_obj": str(kept_mesh),
                                "bbox_xyxy": [1, 1, 7, 7],
                            },
                            "ego_right": None,
                        },
                    },
                    {"frame_idx": 1, "hands": {"ego_left": None, "ego_right": None}},
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "final_hand_result"
    result = export_final_hand_result(
        tracks_json=tracks_json,
        output_dir=output_dir,
        source_frames_dir=source_frames_dir,
        hamer_overlay_dir=source_overlay_dir,
    )

    frame0 = cv2.imread(str(output_dir / "annotated/frame_000000_final.jpg"))
    frame1 = cv2.imread(str(output_dir / "annotated/frame_000001_final.jpg"))
    assert frame0[3, 3, 2] > 180
    assert frame0[3, 3, 1] < 80
    assert max(frame0[14, 14].tolist()) < 10
    assert max(frame1[3, 3].tolist()) < 10
    assert result["copied_filtered_mesh_overlay_count"] == 2
