import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.alignment_interface import (
    build_legacy_alignment_argv,
    filter_candidates_by_frame_range,
    load_alignment_inputs_from_manifest,
)


def test_load_alignment_inputs_from_manifest_uses_left_video_and_calibration_source(tmp_path: Path):
    calib = tmp_path / "camera_intrinsics.json"
    calib.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_name": "wuxi",
                "sequence_name": "demo",
                "reference_camera": "left",
                "videos": {"left": {"path": str(tmp_path / "left.mp4")}},
                "camera_model": {"calibration_source": str(calib)},
            }
        ),
        encoding="utf-8",
    )

    inputs = load_alignment_inputs_from_manifest(manifest)

    assert inputs.video_path == tmp_path / "left.mp4"
    assert inputs.intrinsics_json == calib
    assert inputs.intrinsics_camera == "left"


def test_filter_candidates_by_frame_range_keeps_requested_frames(tmp_path: Path):
    src = tmp_path / "candidates.json"
    dst = tmp_path / "filtered.json"
    src.write_text(
        json.dumps({"frames": [{"frame_idx": 0}, {"frame_idx": 3}, {"frame_idx": 5}]}),
        encoding="utf-8",
    )

    count = filter_candidates_by_frame_range(src, dst, start=2, num_frames=3)

    data = json.loads(dst.read_text(encoding="utf-8"))
    assert count == 1
    assert [f["frame_idx"] for f in data["frames"]] == [3]


def test_build_legacy_alignment_argv_contains_standard_new_method_parameters(tmp_path: Path):
    candidates = tmp_path / "candidates.json"
    depth_dir = tmp_path / "depth"
    video = tmp_path / "left.mp4"
    overlay_dir = tmp_path / "overlay"
    intrinsics = tmp_path / "camera_intrinsics.json"
    output_dir = tmp_path / "align"

    argv = build_legacy_alignment_argv(
        candidates_json=candidates,
        depth_dir=depth_dir,
        video_path=video,
        hamer_overlay_dir=overlay_dir,
        output_dir=output_dir,
        intrinsics_json=intrinsics,
        intrinsics_camera="left",
        no_save_rrd=True,
    )

    assert argv[:2] == ["mesh_depth_alignment_impl.py", "--candidates_json"]
    assert str(candidates) in argv
    assert "--new_method_only" in argv
    assert ["--mesh_mask_source", "hamer_overlay_diff"] == argv[argv.index("--mesh_mask_source") : argv.index("--mesh_mask_source") + 2]
    assert ["--shape_preserved_depth_source", "finger"] == argv[
        argv.index("--shape_preserved_depth_source") : argv.index("--shape_preserved_depth_source") + 2
    ]
    assert "--orientation_aware_compensation" in argv
    assert "--no_save_rrd" in argv
