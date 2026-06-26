import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.track_and_smooth_ego_hands import (
    hand_mesh_entity_paths,
    hand_rgb_overlay_layers,
    hand_rgb_projected_points_paths,
    iter_rerun_offsets,
)
from scripts.track_and_smooth_ego_hands import apply_alignment_orientation_compensation

import numpy as np


def test_iter_rerun_offsets_can_visualize_subset_of_processed_frames():
    args = SimpleNamespace(start=10, num_frames=600, rrd_start=40, rrd_num_frames=30)

    offsets = list(iter_rerun_offsets(args))

    assert offsets[0] == (30, 40)
    assert offsets[-1] == (59, 69)
    assert len(offsets) == 30


def test_hand_mesh_entity_paths_include_current_and_status_layer():
    obs = {"status": "recovered"}

    paths = hand_mesh_entity_paths("ego_left", obs, log_layered_meshes=True)

    assert paths == [
        "camera_opencv/hands/ego_left/current",
        "camera_opencv/hands/ego_left/recovered",
    ]


def test_hand_mesh_entity_paths_can_log_only_current():
    obs = {"status": "interpolated"}

    paths = hand_mesh_entity_paths("ego_right", obs, log_layered_meshes=False)

    assert paths == ["camera_opencv/hands/ego_right/current"]


def test_hand_rgb_overlay_layers_match_current_and_status_layer():
    assert hand_rgb_overlay_layers("ego_left", {"status": "detected"}, True) == ["current", "normal"]
    assert hand_rgb_overlay_layers("ego_left", {"status": "interpolated"}, True) == ["current", "smooth"]
    assert hand_rgb_overlay_layers("ego_left", {"status": "recovered"}, False) == ["current"]


def test_hand_rgb_projected_points_paths_are_under_camera_rgb_overlay():
    paths = hand_rgb_projected_points_paths("ego_right", {"status": "interpolated"}, True)

    assert paths == [
        "camera_opencv/camera/left/rgb/overlay/current/ego_right/mesh_projected_points",
        "camera_opencv/camera/left/rgb/overlay/smooth/ego_right/mesh_projected_points",
    ]


def test_alignment_orientation_compensation_moves_mesh_farther_when_enabled():
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[:, 2] = 0.5
    vertices[0] = [0.0, 0.0, 0.5]
    vertices[1] = [0.04, 0.0, 0.5]
    vertices[4] = [0.0, 0.04, 0.5]
    regressor = np.zeros((16, 778), dtype=np.float32)
    regressor[0, 0] = 1.0
    regressor[1, 1] = 1.0
    regressor[4, 4] = 1.0

    compensated = apply_alignment_orientation_compensation(
        vertices,
        {"side": "ego_left"},
        {"ego_left": regressor},
        {
            "enabled": True,
            "depth_bias_m": 0.035,
            "surface_min_m": 0.005,
            "surface_max_m": 0.020,
            "ray_blend": 0.3,
        },
    )

    assert float(np.median(compensated[:, 2])) > float(np.median(vertices[:, 2]))
