import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.final_smoothing import smooth_final_mesh_track


def test_smooth_final_mesh_track_outputs_final_vertices_without_changing_count():
    vertices_by_frame = [
        np.zeros((4, 3), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
        np.full((4, 3), 2.0, dtype=np.float32),
    ]

    result = smooth_final_mesh_track(vertices_by_frame, ["normal", "recovered", "normal"], window=3)

    assert len(result) == 3
    assert result[0]["vertices"].shape == (4, 3)
    assert result[1]["center_smoothing"]["status"] == "recovered"
    assert "orientation_smoothing" in result[1]


def test_smooth_final_mesh_track_empty_input_returns_empty_list():
    assert smooth_final_mesh_track([], [], window=3) == []
