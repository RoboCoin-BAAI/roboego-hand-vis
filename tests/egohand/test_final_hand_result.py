import json
from pathlib import Path

from roboego_hand_vis.egohand.final_hand_result import export_final_hand_result


def test_export_final_hand_result_copies_only_kept_meshes_without_debug_visuals(tmp_path: Path):
    source_mesh_dir = tmp_path / "hamer/meshes"
    source_mesh_dir.mkdir(parents=True)
    left_mesh = source_mesh_dir / "frame_000000_candidate_01.obj"
    dropped_mesh = source_mesh_dir / "frame_000001_candidate_02.obj"
    left_mesh.write_text("left mesh", encoding="utf-8")
    dropped_mesh.write_text("dropped mesh", encoding="utf-8")

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
    )

    final_tracks = json.loads((output_dir / "final_hand_tracks.json").read_text(encoding="utf-8"))
    assert result["hand_count"] == 1
    assert len(final_tracks["hands"]) == 1
    assert final_tracks["hands"][0]["mesh_obj"].endswith("meshes/frame_000000_ego_left.obj")
    assert final_tracks["hands"][0]["source_mesh_obj"] == str(left_mesh)
    assert (output_dir / "meshes/frame_000000_ego_left.obj").read_text(encoding="utf-8") == "left mesh"
    assert not (output_dir / "meshes/frame_000001_ego_right.obj").exists()
    assert not (output_dir / "annotated").exists()
    assert "copied_visual_count" not in result
    assert "copied_mesh_overlay_count" not in result
    assert "copied_temporal_visual_count" not in result


def test_export_final_hand_result_copies_depth_anchored_artifacts(tmp_path: Path):
    tracks_json = tmp_path / "filtered_hand_tracks.json"
    tracks_json.write_text(
        json.dumps({"frames": [{"frame_idx": 0, "hands": {"ego_left": None, "ego_right": None}}]}),
        encoding="utf-8",
    )
    anchored_dir = tmp_path / "joints21_depth_anchored"
    overlay_dir = anchored_dir / "rgb_skeleton_check"
    overlay_dir.mkdir(parents=True)
    anchored_json = anchored_dir / "joints21_depth_anchored.json"
    anchored_npz = anchored_dir / "joints21_depth_anchored.npz"
    anchored_summary = anchored_dir / "joints21_depth_anchored_summary.json"
    contact_sheet = overlay_dir / "contact_sheet_depth_anchored_joints21.jpg"
    anchored_json.write_text('{"stage":"joints21_depth_anchored_v1"}', encoding="utf-8")
    anchored_npz.write_bytes(b"npz")
    anchored_summary.write_text('{"ego_left":{"valid":0}}', encoding="utf-8")
    contact_sheet.write_bytes(b"jpg")

    output_dir = tmp_path / "final_hand_result"
    result = export_final_hand_result(
        tracks_json=tracks_json,
        output_dir=output_dir,
        depth_anchored_json=anchored_json,
        depth_anchored_npz=anchored_npz,
        depth_anchored_summary_json=anchored_summary,
        copy_meshes=False,
    )

    assert (output_dir / "joints21_depth_anchored.json").read_text(encoding="utf-8") == anchored_json.read_text(
        encoding="utf-8"
    )
    assert (output_dir / "joints21_depth_anchored.npz").read_bytes() == b"npz"
    assert (output_dir / "joints21_depth_anchored_summary.json").read_text(
        encoding="utf-8"
    ) == anchored_summary.read_text(encoding="utf-8")
    assert not (output_dir / "contact_sheet_depth_anchored_joints21.jpg").exists()
    assert result["depth_anchored_outputs"]["joints21_depth_anchored_json"] == str(
        output_dir / "joints21_depth_anchored.json"
    )


def test_export_final_hand_result_accepts_joints21_gated_candidates(tmp_path: Path):
    source_mesh_dir = tmp_path / "hand/final_hand_result/meshes"
    source_mesh_dir.mkdir(parents=True)
    kept_mesh = source_mesh_dir / "frame_000003_ego_right.obj"
    kept_mesh.write_text("kept mesh", encoding="utf-8")
    gated_json = tmp_path / "joints21_pose_gate/final_hand_candidates_gated.json"
    gated_json.parent.mkdir(parents=True)
    gated_json.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_idx": 2, "candidates": []},
                    {
                        "frame_idx": 3,
                        "candidates": [
                            {
                                "side": "ego_right",
                                "candidate_id": 7,
                                "mesh_obj": str(kept_mesh),
                                "bbox_xyxy": [10, 20, 30, 40],
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "final_hand_result"
    result = export_final_hand_result(
        tracks_json=gated_json,
        output_dir=output_dir,
        copy_meshes=True,
    )

    final_tracks = json.loads((output_dir / "final_hand_tracks.json").read_text(encoding="utf-8"))
    assert result["hand_count"] == 1
    assert final_tracks["frames"][0]["hands"] == {"ego_left": None, "ego_right": None}
    assert final_tracks["frames"][1]["hands"]["ego_right"]["candidate_id"] == 7
    assert (output_dir / "meshes/frame_000003_ego_right.obj").read_text(encoding="utf-8") == "kept mesh"
