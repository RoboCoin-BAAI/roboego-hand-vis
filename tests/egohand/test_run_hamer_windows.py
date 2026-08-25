import json
from pathlib import Path

from scripts.hand.run_hamer_windows import build_window_command_plan


def test_build_window_command_plan_uses_window_frame_range_and_policy(tmp_path: Path):
    windows_json = tmp_path / "windows.json"
    windows_json.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0001",
                        "window_type": "two_hand",
                        "hands_required": ["left", "right"],
                        "segment_frame_range": {"start": 150, "num_frames": 90},
                        "recommended_stage1_hand_policy": "permissive",
                    },
                    {
                        "window_id": "handwin_0002",
                        "window_type": "single_hand",
                        "hands_required": ["right"],
                        "segment_frame_range": {"start": 300, "num_frames": 60},
                        "recommended_stage1_hand_policy": "conservative",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_window_command_plan(
        windows_json=windows_json,
        manifest="manifest.json",
        depth_dir="depth",
        output_root=tmp_path / "out",
        hamer_root="/hamer",
        stage1_actions="stage1.json",
        body_detector="regnety",
        batch_size=1,
        save_mesh=True,
        python_executable="/env/bin/python",
        only_window_ids=None,
        accept_uncertain_refined=True,
        wearer_roi_x_margin_ratio=0.0,
        wearer_roi_y_margin_ratio=0.12,
        max_windows=None,
    )

    assert len(plan) == 2
    assert plan[0]["start"] == 150
    assert plan[0]["num_frames"] == 90
    assert plan[0]["stage1_hand_policy"] == "permissive"
    assert plan[0]["output_dir"].endswith("handwin_0001_two_hand_left_right_000150_000239")
    assert plan[0]["command"][0] == "/env/bin/python"
    assert "--accept_uncertain_refined" in plan[0]["command"]
    assert plan[0]["command"][plan[0]["command"].index("--wearer_roi_x_margin_ratio") + 1] == "0.0"
    assert plan[0]["command"][plan[0]["command"].index("--wearer_roi_y_margin_ratio") + 1] == "0.12"
    assert plan[0]["command"][plan[0]["command"].index("--wearer_regions_json") + 1] == "stage1.json"
    assert plan[1]["stage1_hand_policy"] == "conservative"


def test_build_window_command_plan_requires_materialized_hand_window_dirs(tmp_path: Path):
    windows_json = tmp_path / "windows.json"
    windows_json.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0000",
                        "window_type": "two_hand",
                        "source_segment_id": "seg_0000",
                        "hands_required": ["left", "right"],
                        "segment_frame_range": {"start": 120, "num_frames": 210},
                    },
                    {
                        "window_id": "handwin_0001",
                        "window_type": "two_hand",
                        "source_segment_id": "seg_0004",
                        "hands_required": ["left", "right"],
                        "segment_frame_range": {"start": 840, "num_frames": 180},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    segments_root = tmp_path / "segments"
    segment_dir = segments_root / "seg_0000_t000005_000039_operate_place_point_touch"
    (segment_dir / "processed/dataset_manifest").mkdir(parents=True)
    (segment_dir / "processed/foundation_depth").mkdir(parents=True)
    (segment_dir / "processed/dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")

    try:
        build_window_command_plan(
            windows_json=windows_json,
            manifest=None,
            depth_dir=None,
            output_root=None,
            segments_root=segments_root,
            output_subdir="hamer_windows_auto",
            hamer_root="/hamer",
            stage1_actions="stage1.json",
            body_detector="regnety",
            batch_size=1,
            save_mesh=True,
            python_executable="/env/bin/python",
            only_window_ids=["handwin_0000"],
            max_windows=None,
        )
    except FileNotFoundError as exc:
        assert "Missing materialized hand-window directory" in str(exc)
    else:
        raise AssertionError("processed/ fallback should not be used by --segments_root dispatch")


def test_build_window_command_plan_prefers_materialized_hand_window_dirs(tmp_path: Path):
    windows_json = tmp_path / "windows.json"
    windows_json.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0000",
                        "window_type": "two_hand",
                        "source_segment_id": "seg_0000",
                        "source_segment_range": {"start": 6, "end": 15},
                        "start": 8,
                        "end": 15,
                        "hands_required": ["left", "right"],
                        "segment_frame_range": {"start": 60, "num_frames": 240},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    segment_dir = tmp_path / "segments/seg_0000_t000006_000015_place_reach"
    handwin_dir = segment_dir / "hand_windows/handwin_0000_t000008_000015"
    (handwin_dir / "dataset_manifest").mkdir(parents=True)
    (handwin_dir / "fast_foundation_depth").mkdir()
    (handwin_dir / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")

    plan = build_window_command_plan(
        windows_json=windows_json,
        manifest=None,
        depth_dir=None,
        output_root=None,
        segments_root=tmp_path / "segments",
        output_subdir="hamer_windows_left_v1",
        hamer_root="/hamer",
        stage1_actions="stage1.json",
        body_detector="regnety",
        batch_size=1,
        save_mesh=True,
        python_executable="/env/bin/python",
        only_window_ids=None,
        max_windows=None,
    )

    assert len(plan) == 1
    assert plan[0]["start"] == 0
    assert plan[0]["num_frames"] == 240
    assert plan[0]["command"][plan[0]["command"].index("--manifest") + 1].endswith(
        "hand_windows/handwin_0000_t000008_000015/dataset_manifest/dataset_manifest.json"
    )
    assert plan[0]["command"][plan[0]["command"].index("--depth_dir") + 1].endswith(
        "hand_windows/handwin_0000_t000008_000015/fast_foundation_depth"
    )
    assert plan[0]["stage1_time_offset_s"] == 8.0
    assert plan[0]["command"][plan[0]["command"].index("--stage1_time_offset_s") + 1] == "8.0"
    assert plan[0]["output_dir"].endswith(
        "hand_windows/handwin_0000_t000008_000015/hamer_windows_left_v1/handwin_0000_two_hand_left_right_000000_000239"
    )


def test_build_window_command_plan_can_filter_by_source_segment_id(tmp_path: Path):
    windows_json = tmp_path / "windows.json"
    windows_json.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0000",
                        "source_segment_id": "seg_0000",
                        "segment_frame_range": {"start": 0, "num_frames": 30},
                    },
                    {
                        "window_id": "handwin_0001",
                        "source_segment_id": "seg_0004",
                        "segment_frame_range": {"start": 0, "num_frames": 30},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    segments_root = tmp_path / "segments"
    for segment_name, window_id in [("seg_0000_a", "handwin_0000"), ("seg_0004_b", "handwin_0001")]:
        handwin_dir = segments_root / segment_name / "hand_windows" / f"{window_id}_t000000_000001"
        (handwin_dir / "dataset_manifest").mkdir(parents=True)
        (handwin_dir / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")

    plan = build_window_command_plan(
        windows_json=windows_json,
        manifest=None,
        depth_dir=None,
        output_root=None,
        segments_root=segments_root,
        output_subdir="hamer_windows_auto",
        hamer_root="/hamer",
        stage1_actions="stage1.json",
        body_detector="regnety",
        batch_size=1,
        save_mesh=True,
        python_executable="/env/bin/python",
        only_window_ids=None,
        only_segment_ids=["seg_0004"],
        max_windows=None,
    )

    assert [item["source_segment_id"] for item in plan] == ["seg_0004"]


def test_build_window_command_plan_allows_hand_window_without_foundation_depth_for_hamer_only(tmp_path: Path):
    windows_json = tmp_path / "windows.json"
    windows_json.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0000",
                        "source_segment_id": "seg_0000",
                        "segment_frame_range": {"start": 840, "num_frames": 420},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    segment_dir = tmp_path / "segments/seg_0000_t000000_000197_handover_lower_open_operate"
    handwin_dir = segment_dir / "hand_windows/handwin_0000_t000028_000042"
    (handwin_dir / "dataset_manifest").mkdir(parents=True)
    (handwin_dir / "dataset_manifest/dataset_manifest.json").write_text("{}", encoding="utf-8")

    plan = build_window_command_plan(
        windows_json=windows_json,
        manifest=None,
        depth_dir=None,
        output_root=None,
        segments_root=tmp_path / "segments",
        output_subdir="hamer_windows_left_v1",
        hamer_root="/hamer",
        stage1_actions="stage1.json",
        body_detector="regnety",
        batch_size=1,
        save_mesh=True,
        python_executable="/env/bin/python",
        only_window_ids=None,
        max_windows=None,
    )

    assert len(plan) == 1
    assert plan[0]["depth_dir"] is None
    assert plan[0]["depth_missing_for_hamer_only"] is True
    assert "--depth_dir" not in plan[0]["command"]
