import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hand.export_depth_joints21_rerun import (
    iter_export_offsets,
    load_hand_tracking_parquet,
    load_intrinsics,
    project_points,
)


def test_load_intrinsics_reads_fast_foundation_metadata(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "fast_foundation_stereo_video_meta.json").write_text(
        json.dumps({"intrinsics": {"fx": 443.5, "fy": 444.5, "cx": 970.0, "cy": 525.0}}),
        encoding="utf-8",
    )

    intrinsics = load_intrinsics(depth_dir)

    assert intrinsics == {"fx": 443.5, "fy": 444.5, "cx": 970.0, "cy": 525.0}


def test_iter_export_offsets_selects_requested_absolute_frame():
    offsets = list(iter_export_offsets(start=100, num_frames=200, rrd_start=167, rrd_num_frames=1))

    assert offsets == [(67, 167)]


def test_project_points_marks_behind_camera_and_outside_image_invalid():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [10.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    pixels, valid = project_points(points, {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 40.0}, 100, 80)

    assert np.allclose(pixels[0], [50.0, 40.0])
    assert valid.tolist() == [True, False, False]


def test_load_hand_tracking_parquet_reads_present_left_and_right(tmp_path: Path):
    left = np.arange(63, dtype=np.float32).reshape(21, 3).tolist()
    right = (np.arange(63, dtype=np.float32).reshape(21, 3) + 100.0).tolist()
    table = pa.table(
        {
            "frame_idx": [167],
            "left_present": [True],
            "right_present": [True],
            "left_joints_3d": [left],
            "right_joints_3d": [right],
        }
    )
    path = tmp_path / "hand_tracking.parquet"
    pq.write_table(table, path)

    hands = load_hand_tracking_parquet(path)

    assert hands["left"]["valid"].tolist() == [True]
    assert hands["right"]["valid"].tolist() == [True]
    assert np.allclose(hands["left"]["joints"][0], np.asarray(left, dtype=np.float32))
    assert np.allclose(hands["right"]["joints"][0], np.asarray(right, dtype=np.float32))
