# EgoHand Post-HaMeR Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the post-HaMeR hand pipeline so HaMeR/VLM ROI maximizes recall, temporal filtering only removes extreme bbox/numeric failures, and joints21 pose gating becomes the main quality gate.

**Architecture:** Keep the existing Kedro `egohand` pipeline and script wrappers, but clarify responsibilities and output folders. Replace strong temporal mesh/MANO deletion with a lightweight `hand_filter` stage, preserve the existing `joints21_pose_gate` and `depth_anchored_joints21` stages as the authoritative quality and 3D export steps, then run `final_hand_result` only as the last packaging step using the joints21-gated candidates as its hand source.

**Tech Stack:** Python 3.11, Kedro, pytest, existing HaMeR outputs, JSON/NPZ artifacts, OpenCV debug images.

---

### Task 1: Add Lightweight Hand Filter Semantics

**Files:**
- Modify: `roboego_hand_vis/egohand/refined_temporal.py`
- Modify: `scripts/hand/track_refined_hands.py`
- Test: `tests/egohand/test_refined_temporal.py`

- [ ] **Step 1: Write the failing test**

Add a test proving that mesh and MANO gates can be disabled and that bbox-only filtering preserves a visually selected hand when mesh jumps are large.

```python
def test_temporal_gate_can_preserve_hand_when_mesh_and_pose_gates_disabled(tmp_path: Path):
    mesh_a = tmp_path / "a.obj"
    mesh_b = tmp_path / "b.obj"
    mesh_a.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n", encoding="utf-8")
    mesh_b.write_text("v 100 100 100\nv 101 100 100\nv 100 101 100\n", encoding="utf-8")
    payload = {
        "frames": [
            {
                "frame_idx": 0,
                "ego_right_refined": {
                    "candidate_id": 1,
                    "bbox_xyxy": [100, 100, 200, 200],
                    "mesh_obj": str(mesh_a),
                    "mano_params": {"global_orient_rotmat": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
                },
            },
            {
                "frame_idx": 1,
                "ego_right_refined": {
                    "candidate_id": 2,
                    "bbox_xyxy": [105, 102, 205, 202],
                    "mesh_obj": str(mesh_b),
                    "mano_params": {"global_orient_rotmat": [[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]]},
                },
            },
        ]
    }

    result = build_temporal_tracks_from_refined(
        payload,
        start=0,
        num_frames=2,
        max_bbox_jump_px=360,
        max_bbox_area_change_ratio=8.0,
        max_mano_pose_jump=0,
        max_mesh_centroid_jump=0,
        max_mesh_vertex_rmse=0,
    )

    assert result["frames"][1]["hands"]["ego_right"]["candidate_id"] == 2
    assert result["summary"]["ego_right"]["dropped"] == 0
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/egohand/test_refined_temporal.py -q
```

Expected: PASS if disable semantics already work. If it fails, implement the minimal fix in `roboego_hand_vis/egohand/refined_temporal.py` so values `<= 0` disable MANO and mesh gates consistently.

- [ ] **Step 3: Rename user-facing stage wording without breaking old node**

Keep node name `refined_temporal_gate` for compatibility, but document and configure it as a weak filter:

```yaml
refined_temporal_gate:
  max_bbox_jump_px: 360
  max_bbox_area_change_ratio: 8.0
  max_mano_pose_jump: 0
  max_mesh_centroid_jump: 0
  max_mesh_vertex_rmse: 0
  save_vis: false
```

- [ ] **Step 4: Verify**

Run:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/egohand/test_refined_temporal.py tests/pipelines/test_egohand_hamer_windows_node.py -q
```

Expected: all selected tests pass.

### Task 2: Add Hand Filter Node As the Official Middle Stage

**Files:**
- Modify: `scripts/hand/track_refined_hands.py`
- Modify: `roboego_hand_vis/pipelines/egohand/nodes.py`
- Modify: `roboego_hand_vis/pipelines/egohand/pipeline.py`
- Test: `tests/pipelines/test_egohand_hamer_windows_node.py`

- [ ] **Step 1: Add `--output_name` to the existing filter script**

Use `filtered_hand_tracks.json` for the official `hand_filter` output while preserving the old `refined_temporal_gate_tracks.json` default.

- [ ] **Step 2: Add `hand_filter` node**

The node calls `scripts/hand/track_refined_hands.py` with weak defaults and returns:

```python
{
    "stage": "hand_filter",
    "status": "complete",
    "tracks_json": "<output_dir>/filtered_hand_tracks.json",
}
```

- [ ] **Step 3: Keep `refined_temporal_gate` as a compatibility node**

Do not remove the old node in this pass. Existing commands and artifacts remain readable.

### Task 3: Keep Joints21 Pose Gate as the Main Gate

**Files:**
- Modify: `conf/base/parameters.yml`
- Modify: `README.md`
- Test: `tests/egohand/test_joints21_pose_gate.py`

- [ ] **Step 1: Lock the intended default thresholds**

Use the looser, currently accepted defaults:

```yaml
joints21_pose_gate:
  max_bone_length_ratio: 8.0
  max_joint_local_deviation: 7.0
  max_wrist_local_deviation_m: 0.6
  max_palm_local_deviation_m: 0.6
  max_fingertip_local_deviation_m: 0.7
  min_finger_direction_dot: -0.15
  local_window_radius: 2
```

- [ ] **Step 2: Verify the gate still removes clear 21-point jumps**

Run:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/egohand/test_joints21_pose_gate.py -q
```

Expected: PASS.

### Task 4: Make Final Hand Result the Last Packaging Stage

**Files:**
- Modify: `roboego_hand_vis/egohand/final_hand_result.py`
- Modify: `scripts/hand/export_final_hand_result.py`
- Test: `tests/egohand/test_final_hand_result.py`

- [ ] **Step 1: Add a test for minimal final outputs**

Add a test that `copy_visuals=False` creates only JSON summary and kept mesh files.

```python
def test_export_final_hand_result_minimal_output_has_no_annotated_visuals(tmp_path: Path):
    tracks_json = tmp_path / "tracks.json"
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    mesh_path = mesh_dir / "frame_000000_candidate_00.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")
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
                                "candidate_id": 0,
                                "mesh_obj": str(mesh_path),
                                "bbox_xyxy": [1, 2, 3, 4],
                            },
                            "ego_right": None,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "final"
    result = export_final_hand_result(
        tracks_json=tracks_json,
        output_dir=output_dir,
        copy_meshes=True,
        copy_visuals=False,
    )

    assert (output_dir / "final_hand_tracks.json").is_file()
    assert (output_dir / "final_hand_summary.json").is_file()
    assert (output_dir / "meshes" / mesh_path.name).is_file()
    assert not (output_dir / "annotated").exists()
    assert result["copied_visual_count"] == 0
```

- [ ] **Step 2: Run test**

Run:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/egohand/test_final_hand_result.py -q
```

Expected: PASS. If it fails, adjust `export_final_hand_result` so `copy_visuals=False` never creates `annotated/`.

- [ ] **Step 3: Accept gated candidates as final hand source**

Add support for `frames[].candidates[]` from `final_hand_candidates_gated.json`, so final packaging copies only meshes that survived `joints21_pose_gate`.

- [ ] **Step 4: Copy depth-anchored artifacts into the final result**

When provided, copy these files into `final_hand_result/`:

```text
joints21_depth_anchored.json
joints21_depth_anchored.npz
joints21_depth_anchored_summary.json
contact_sheet_depth_anchored_joints21.jpg
```

### Task 5: Document the Short Official Chain

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the long hand section with the short chain**

Document this as the official hand chain:

```text
hamer_windows
  -> hand_filter as weak bbox/numeric filter
  -> joints21_pose_gate as main pose quality gate
  -> depth_anchored_joints21 as 3D joints export
  -> final_hand_result as final packaging
```

- [ ] **Step 2: Include the recommended command values**

Update the README command for `hand_filter`:

```bash
--params hand_filter.enabled=True,hand_filter.manifest=$MANIFEST,hand_filter.refined_json=$REFINED_JSON,hand_filter.output_dir=$FILTER_DIR,hand_filter.start=0,hand_filter.num_frames=1980,hand_filter.max_bbox_jump_px=360,hand_filter.max_bbox_area_change_ratio=8.0,hand_filter.max_mano_pose_jump=0,hand_filter.max_mesh_centroid_jump=0,hand_filter.max_mesh_vertex_rmse=0,hand_filter.local_window_radius=2,hand_filter.save_vis=False
```

- [ ] **Step 3: Final verification**

Run:

```bash
PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/egohand/test_refined_temporal.py tests/egohand/test_final_hand_result.py tests/egohand/test_joints21_pose_gate.py tests/pipelines/test_egohand_hamer_windows_node.py -q
git diff --check
```

Expected: all tests pass and `git diff --check` has no output.
