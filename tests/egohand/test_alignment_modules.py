import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_alignment_io_selects_normal_and_recovered_candidates():
    from roboego_hand_vis.egohand.alignment_io import canonical_hand_status, selected_candidates

    frame = {
        "candidates": [
            {"status": "other"},
            {"status": "ego_right_recovered"},
            {"status": "ego_left"},
        ]
    }

    assert [c["status"] for c in selected_candidates(frame)] == ["ego_left", "ego_right_recovered"]
    assert canonical_hand_status("ego_left_recovered") == "ego_left"


def test_alignment_geometry_samples_points_deterministically():
    from roboego_hand_vis.egohand.alignment_geometry import sample_points

    points = np.arange(30).reshape(10, 3)

    sampled = sample_points(points, max_points=4)

    assert sampled.tolist() == [[0, 1, 2], [9, 10, 11], [18, 19, 20], [27, 28, 29]]


def test_alignment_rerun_module_exports_logging_helpers():
    from roboego_hand_vis.egohand.alignment_rerun import clear_hand_entities, log_mesh_layer

    assert callable(clear_hand_entities)
    assert callable(log_mesh_layer)
