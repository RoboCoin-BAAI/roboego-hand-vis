import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.orientation import apply_orientation_compensation, vertices_to_hamer21_joints


def test_vertices_to_hamer21_joints_uses_regressor_and_tip_vertices():
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[744] = [1.0, 2.0, 3.0]
    regressor = np.zeros((16, 778), dtype=np.float32)
    regressor[0, 0] = 1.0
    vertices[0] = [0.1, 0.2, 0.3]

    joints = vertices_to_hamer21_joints(vertices, regressor)

    assert joints.shape == (21, 3)
    np.testing.assert_allclose(joints[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(joints[4], [1.0, 2.0, 3.0])


def test_apply_orientation_compensation_can_be_disabled():
    vertices = np.zeros((778, 3), dtype=np.float32)
    args = SimpleNamespace(orientation_aware_compensation=False)

    out, info = apply_orientation_compensation(vertices, "ego_left", {}, args)

    np.testing.assert_allclose(out, vertices)
    assert info["enabled"] is False
