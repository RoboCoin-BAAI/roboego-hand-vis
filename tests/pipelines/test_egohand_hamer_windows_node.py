from roboego_hand_vis.pipelines.egohand.nodes import (
    run_depth_anchored_joints21_node,
    run_final_hand_result_node,
    run_final_hand_trajectory_polish_node,
    run_hamer_windows_node,
    run_hand_filter_node,
    run_joints21_pose_gate_node,
    run_legacy_export_joints21_node,
    run_legacy_mesh_depth_alignment_node,
    run_legacy_track_fill_smooth_node,
    run_refined_temporal_gate_node,
)


def test_hamer_windows_node_skips_when_disabled():
    result = run_hamer_windows_node({"enabled": False})

    assert result == {
        "stage": "hamer_windows",
        "status": "skipped",
        "reason": "disabled",
    }


def test_hamer_windows_node_runs_existing_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)

    result = run_hamer_windows_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/run_hamer_windows.py",
            "windows_json": str(tmp_path / "session/annotation/hand_estimation_windows_left_v1.json"),
            "segments_root": str(tmp_path / "session/segments"),
            "output_subdir": "hand/hamer_raw",
            "hamer_root": "/home/liuyou/Documents/hamer",
            "stage1_actions": str(tmp_path / "session/annotation/stage1_action_resolved_chunks/chunk_0000.json"),
            "wearer_regions_json": str(tmp_path / "session/annotation/qwen_frame_annotations_left_v4.checkpoint.json"),
            "hamer_python": "/envs/hamer/bin/python",
            "body_detector": "regnety",
            "batch_size": 1,
            "wearer_roi_x_margin_ratio": 0.0,
            "wearer_roi_y_margin_ratio": 0.1,
            "save_mesh": True,
            "only_window_ids": ["handwin_0000"],
            "only_segment_ids": ["seg_0000"],
            "max_windows": 1,
            "skip_missing_segments": True,
            "overwrite": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/run_hamer_windows.py",
                "--windows_json",
                str(tmp_path / "session/annotation/hand_estimation_windows_left_v1.json"),
                "--segments_root",
                str(tmp_path / "session/segments"),
                "--output_subdir",
                "hand/hamer_raw",
                "--hamer_root",
                "/home/liuyou/Documents/hamer",
                "--stage1_actions",
                str(tmp_path / "session/annotation/stage1_action_resolved_chunks/chunk_0000.json"),
                "--wearer_regions_json",
                str(tmp_path / "session/annotation/qwen_frame_annotations_left_v4.checkpoint.json"),
                "--python",
                "/envs/hamer/bin/python",
                "--body_detector",
                "regnety",
                "--batch_size",
                "1",
                "--wearer_roi_x_margin_ratio",
                "0.0",
                "--wearer_roi_y_margin_ratio",
                "0.1",
                "--save_mesh",
                "--only_window_ids",
                "handwin_0000",
                "--only_segment_ids",
                "seg_0000",
                "--max_windows",
                "1",
                "--skip_missing_segments",
                "--overwrite",
            ],
        }
    ]
    assert result == {
        "stage": "hamer_windows",
        "status": "complete",
        "windows_json": str(tmp_path / "session/annotation/hand_estimation_windows_left_v1.json"),
        "segments_root": str(tmp_path / "session/segments"),
        "plan_path": str(tmp_path / "session/segments/hamer_window_run_plan.json"),
    }


def test_refined_temporal_gate_node_skips_when_disabled():
    result = run_refined_temporal_gate_node({"enabled": False})

    assert result == {
        "stage": "refined_temporal_gate",
        "status": "skipped",
        "reason": "disabled",
    }


def test_refined_temporal_gate_node_runs_existing_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/refined_temporal_gate"

    result = run_refined_temporal_gate_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/track_refined_hands.py",
            "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
            "refined_json": str(tmp_path / "handwin/hamer/ego_hand_candidates_refined.json"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "max_gap": 10,
            "max_bbox_jump_px": 220,
            "max_bbox_area_change_ratio": 3.0,
            "max_cam_t_jump": 0.0,
            "min_cam_t_z": 0.0,
            "max_cam_t_z": 200.0,
            "max_mano_pose_jump": 4.5,
            "max_mesh_centroid_jump": 15.0,
            "max_mesh_vertex_rmse": 8.5,
            "local_window_radius": 2,
            "save_vis": False,
            "contact_sheet_columns": 5,
            "contact_sheet_thumb_width": 320,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/track_refined_hands.py",
                "--manifest",
                str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
                "--refined_json",
                str(tmp_path / "handwin/hamer/ego_hand_candidates_refined.json"),
                "--output_dir",
                str(output_dir),
                "--start",
                "0",
                "--num_frames",
                "240",
                "--max_gap",
                "10",
                "--max_bbox_jump_px",
                "220",
                "--max_bbox_area_change_ratio",
                "3.0",
                "--max_cam_t_jump",
                "0.0",
                "--min_cam_t_z",
                "0.0",
                "--max_cam_t_z",
                "200.0",
                "--max_mano_pose_jump",
                "4.5",
                "--max_mesh_centroid_jump",
                "15.0",
                "--max_mesh_vertex_rmse",
                "8.5",
                "--local_window_radius",
                "2",
                "--no-save_vis",
                "--contact_sheet_columns",
                "5",
                "--contact_sheet_thumb_width",
                "320",
            ],
        }
    ]
    assert result == {
        "stage": "refined_temporal_gate",
        "status": "complete",
        "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
        "refined_json": str(tmp_path / "handwin/hamer/ego_hand_candidates_refined.json"),
        "output_dir": str(output_dir),
        "tracks_json": str(output_dir / "refined_temporal_gate_tracks.json"),
    }


def test_hand_filter_node_skips_when_disabled():
    result = run_hand_filter_node({"enabled": False})

    assert result == {
        "stage": "hand_filter",
        "status": "skipped",
        "reason": "disabled",
    }


def test_hand_filter_node_runs_weak_filter_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/hand/filtered"

    result = run_hand_filter_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/track_refined_hands.py",
            "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
            "refined_json": str(tmp_path / "handwin/hand/hamer_raw/ego_hand_candidates_refined.json"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "max_bbox_jump_px": 360,
            "max_bbox_area_change_ratio": 8.0,
            "max_mano_pose_jump": 0,
            "max_mesh_centroid_jump": 0,
            "max_mesh_vertex_rmse": 0,
            "save_vis": False,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/track_refined_hands.py",
                "--manifest",
                str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
                "--refined_json",
                str(tmp_path / "handwin/hand/hamer_raw/ego_hand_candidates_refined.json"),
                "--output_dir",
                str(output_dir),
                "--output_name",
                "filtered_hand_tracks.json",
                "--start",
                "0",
                "--num_frames",
                "240",
                "--max_bbox_jump_px",
                "360",
                "--max_bbox_area_change_ratio",
                "8.0",
                "--max_mano_pose_jump",
                "0",
                "--max_mesh_centroid_jump",
                "0",
                "--max_mesh_vertex_rmse",
                "0",
                "--no-save_vis",
            ],
        }
    ]
    assert result == {
        "stage": "hand_filter",
        "status": "complete",
        "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
        "refined_json": str(tmp_path / "handwin/hand/hamer_raw/ego_hand_candidates_refined.json"),
        "output_dir": str(output_dir),
        "tracks_json": str(output_dir / "filtered_hand_tracks.json"),
    }


def test_final_hand_result_node_skips_when_disabled():
    result = run_final_hand_result_node({"enabled": False})

    assert result == {
        "stage": "final_hand_result",
        "status": "skipped",
        "reason": "disabled",
    }


def test_final_hand_result_node_runs_existing_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/final_hand_result"

    result = run_final_hand_result_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/export_final_hand_result.py",
            "tracks_json": str(tmp_path / "handwin/refined_temporal_gate/refined_temporal_gate_tracks.json"),
            "output_dir": str(output_dir),
            "depth_anchored_json": str(tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored.json"),
            "depth_anchored_npz": str(tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored.npz"),
            "depth_anchored_summary_json": str(
                tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored_summary.json"
            ),
            "copy_depth_anchored": True,
            "copy_meshes": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/export_final_hand_result.py",
                "--tracks_json",
                str(tmp_path / "handwin/refined_temporal_gate/refined_temporal_gate_tracks.json"),
                "--output_dir",
                str(output_dir),
                "--depth_anchored_json",
                str(tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored.json"),
                "--depth_anchored_npz",
                str(tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored.npz"),
                "--depth_anchored_summary_json",
                str(tmp_path / "handwin/hand/joints21_depth_anchored/joints21_depth_anchored_summary.json"),
                "--copy_depth_anchored",
                "--copy_meshes",
            ],
        }
    ]
    assert result == {
        "stage": "final_hand_result",
        "status": "complete",
        "tracks_json": str(tmp_path / "handwin/refined_temporal_gate/refined_temporal_gate_tracks.json"),
        "output_dir": str(output_dir),
        "final_tracks_json": str(output_dir / "final_hand_tracks.json"),
        "final_summary_json": str(output_dir / "final_hand_summary.json"),
        "joints21_depth_anchored_json": str(output_dir / "joints21_depth_anchored.json"),
        "joints21_depth_anchored_npz": str(output_dir / "joints21_depth_anchored.npz"),
    }


def test_joints21_pose_gate_node_skips_when_disabled():
    result = run_joints21_pose_gate_node({"enabled": False})

    assert result == {
        "stage": "joints21_pose_gate",
        "status": "skipped",
        "reason": "disabled",
    }


def test_final_hand_trajectory_polish_node_runs_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/hand/final_hand_result"
    joints_json = output_dir / "joints21_depth_anchored.json"

    result = run_final_hand_trajectory_polish_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/polish_final_hand_trajectory.py",
            "joints_json": str(joints_json),
            "output_dir": str(output_dir),
            "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
            "max_interp_gap": 10,
            "wrist_jump_m": 0.0,
            "max_local_wrist_deviation_m": 0.0,
            "local_window_radius": 5,
            "smooth_window": 9,
            "smooth_relative_fingers": True,
            "write_preview": True,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/polish_final_hand_trajectory.py",
                "--joints_json",
                str(joints_json),
                "--output_dir",
                str(output_dir),
                "--manifest",
                str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
                "--max_interp_gap",
                "10",
                "--wrist_jump_m",
                "0.0",
                "--max_local_wrist_deviation_m",
                "0.0",
                "--local_window_radius",
                "5",
                "--smooth_window",
                "9",
                "--smooth_relative_fingers",
                "--write_preview",
            ],
        }
    ]
    assert result == {
        "stage": "final_hand_trajectory_polish",
        "status": "complete",
        "joints_json": str(joints_json),
        "output_dir": str(output_dir),
        "smoothed_joints21_json": str(output_dir / "smoothed_joints21.json"),
        "smoothed_joints21_npz": str(output_dir / "smoothed_joints21.npz"),
        "trajectory_polish_report_json": str(output_dir / "trajectory_polish_report.json"),
        "preview_video": str(output_dir / "hand_estimation_preview_smoothed.mp4"),
    }


def test_joints21_pose_gate_node_runs_existing_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/joints21_pose_quality"
    final_tracks_json = tmp_path / "handwin/final_hand_result/final_hand_tracks.json"

    result = run_joints21_pose_gate_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/gate_hand_pose_joints21.py",
            "final_tracks_json": str(final_tracks_json),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "mano_model_dir": "/home/liuyou/Documents/hamer/_DATA/data/mano",
            "max_bone_length_ratio": 8.0,
            "max_joint_local_deviation": 7.0,
            "max_wrist_local_deviation_m": 0.25,
            "max_palm_local_deviation_m": 0.25,
            "max_fingertip_local_deviation_m": 0.30,
            "min_finger_direction_dot": -0.15,
            "drop_fingertips_below_wrist": True,
            "image_height": 1050,
            "max_suspicious_wrist_y_ratio": 0.55,
            "min_fingertip_below_wrist_ratio": 1.0,
            "min_fingertip_below_wrist_delta": 0.02,
            "local_window_radius": 2,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/gate_hand_pose_joints21.py",
                "--final_tracks_json",
                str(final_tracks_json),
                "--output_dir",
                str(output_dir),
                "--start",
                "0",
                "--num_frames",
                "240",
                "--mano_model_dir",
                "/home/liuyou/Documents/hamer/_DATA/data/mano",
                "--max_bone_length_ratio",
                "8.0",
                "--max_joint_local_deviation",
                "7.0",
                "--max_wrist_local_deviation_m",
                "0.25",
                "--max_palm_local_deviation_m",
                "0.25",
                "--max_fingertip_local_deviation_m",
                "0.3",
                "--min_finger_direction_dot",
                "-0.15",
                "--drop_fingertips_below_wrist",
                "--image_height",
                "1050",
                "--max_suspicious_wrist_y_ratio",
                "0.55",
                "--min_fingertip_below_wrist_ratio",
                "1.0",
                "--min_fingertip_below_wrist_delta",
                "0.02",
                "--local_window_radius",
                "2",
            ],
        }
    ]
    assert result == {
        "stage": "joints21_pose_gate",
        "status": "complete",
        "final_tracks_json": str(final_tracks_json),
        "output_dir": str(output_dir),
        "pose_quality_report_json": str(output_dir / "pose_quality_report.json"),
        "final_hand_candidates_gated_json": str(output_dir / "final_hand_candidates_gated.json"),
        "joints21_compact_npz": str(output_dir / "joints21_compact.npz"),
    }


def test_depth_anchored_joints21_node_skips_when_disabled():
    result = run_depth_anchored_joints21_node({"enabled": False})

    assert result == {
        "stage": "depth_anchored_joints21",
        "status": "skipped",
        "reason": "disabled",
    }


def test_depth_anchored_joints21_node_runs_bbox_depth_anchor_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/joints21_depth_anchored"

    result = run_depth_anchored_joints21_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "script": "scripts/hand/export_depth_anchored_joints21.py",
            "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
            "candidates_json": str(tmp_path / "handwin/joints21_pose_quality/final_hand_candidates_gated.json"),
            "depth_dir": str(tmp_path / "handwin/fast_foundation_depth"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "mano_model_dir": "/home/liuyou/Documents/hamer/_DATA/data/mano",
            "anchor_mode": "wrist_palm",
            "anchor_joint_index": 0,
            "depth_window_radius_px": 4,
            "min_depth_m": 0.15,
            "max_depth_m": 2.5,
            "overlay_frames": "237",
            "overlay_thumb_width": 480,
            "overlay_columns": 3,
        }
    )

    assert calls == [
        {
            "check": True,
            "args": [
                "/envs/ego/bin/python",
                "scripts/hand/export_depth_anchored_joints21.py",
                "--manifest",
                str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
                "--candidates_json",
                str(tmp_path / "handwin/joints21_pose_quality/final_hand_candidates_gated.json"),
                "--depth_dir",
                str(tmp_path / "handwin/fast_foundation_depth"),
                "--output_dir",
                str(output_dir),
                "--start",
                "0",
                "--num_frames",
                "240",
                "--mano_model_dir",
                "/home/liuyou/Documents/hamer/_DATA/data/mano",
                "--anchor_mode",
                "wrist_palm",
                "--anchor_joint_index",
                "0",
                "--depth_window_radius_px",
                "4",
                "--min_depth_m",
                "0.15",
                "--max_depth_m",
                "2.5",
                "--overlay_frames",
                "237",
                "--overlay_thumb_width",
                "480",
                "--overlay_columns",
                "3",
            ],
        }
    ]
    assert result == {
        "stage": "depth_anchored_joints21",
        "status": "complete",
        "manifest": str(tmp_path / "handwin/dataset_manifest/dataset_manifest.json"),
        "candidates_json": str(tmp_path / "handwin/joints21_pose_quality/final_hand_candidates_gated.json"),
        "depth_dir": str(tmp_path / "handwin/fast_foundation_depth"),
        "output_dir": str(output_dir),
        "joints21_depth_anchored_json": str(output_dir / "joints21_depth_anchored.json"),
        "joints21_depth_anchored_npz": str(output_dir / "joints21_depth_anchored.npz"),
        "summary_json": str(output_dir / "joints21_depth_anchored_summary.json"),
    }


def test_legacy_mesh_depth_alignment_node_runs_old_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/hand_depth_alignment"

    result = run_legacy_mesh_depth_alignment_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "manifest": str(tmp_path / "manifest.json"),
            "candidates_json": str(tmp_path / "candidates.json"),
            "depth_dir": str(tmp_path / "depth"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "mesh_mask_source": "hamer_renderer",
            "no_save_rrd": True,
        }
    )

    assert calls[0]["args"] == [
        "/envs/ego/bin/python",
        "scripts/hand/align_mesh_depth.py",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--candidates_json",
        str(tmp_path / "candidates.json"),
        "--depth_dir",
        str(tmp_path / "depth"),
        "--output_dir",
        str(output_dir),
                "--start",
                "0",
                "--num_frames",
                "240",
                "--no_save_rrd",
                "--mesh_mask_source",
                "hamer_renderer",
            ]
    assert result["alignment_report_json"] == str(output_dir / "hand_depth_mesh_alignment_report.json")


def test_legacy_track_fill_smooth_node_runs_old_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/ego_hand_tracking"

    result = run_legacy_track_fill_smooth_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "manifest": str(tmp_path / "manifest.json"),
            "alignment_report": str(tmp_path / "alignment.json"),
            "depth_dir": str(tmp_path / "depth"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "max_interp_gap": 10,
            "smooth_alpha": 0.35,
            "apply_recovered_depth_overrides": True,
            "final_smoothing_window": 5,
            "no_save_rrd": True,
        }
    )

    assert calls[0]["args"] == [
        "/envs/ego/bin/python",
        "scripts/hand/track_fill_smooth.py",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--alignment_report",
        str(tmp_path / "alignment.json"),
        "--depth_dir",
        str(tmp_path / "depth"),
        "--output_dir",
        str(output_dir),
        "--start",
        "0",
        "--num_frames",
        "240",
        "--max_interp_gap",
        "10",
        "--smooth_alpha",
        "0.35",
        "--apply_recovered_depth_overrides",
        "--final_smoothing_window",
        "5",
        "--no_save_rrd",
    ]
    assert result["tracks_json"] == str(output_dir / "ego_hand_tracks.json")


def test_legacy_export_joints21_node_runs_old_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check):
        calls.append({"args": args, "check": check})

    monkeypatch.setattr("roboego_hand_vis.pipelines.egohand.nodes.subprocess.run", fake_run)
    output_dir = tmp_path / "handwin/joints21_final"

    result = run_legacy_export_joints21_node(
        {
            "enabled": True,
            "python": "/envs/ego/bin/python",
            "tracks_json": str(tmp_path / "tracks.json"),
            "alignment_report": str(tmp_path / "alignment.json"),
            "output_dir": str(output_dir),
            "start": 0,
            "num_frames": 240,
            "final_smoothing_window": 5,
            "no_save_rrd": True,
        }
    )

    assert calls[0]["args"] == [
        "/envs/ego/bin/python",
        "scripts/hand/export_joints21.py",
        "--tracks_json",
        str(tmp_path / "tracks.json"),
        "--alignment_report",
        str(tmp_path / "alignment.json"),
        "--output_dir",
        str(output_dir),
        "--start",
        "0",
        "--num_frames",
        "240",
        "--final_smoothing_window",
        "5",
        "--no_save_rrd",
    ]
    assert result["joints21_json"] == str(output_dir / "joints_21_camera_opencv.json")
