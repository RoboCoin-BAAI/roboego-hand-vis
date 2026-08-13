import numpy as np

from roboego_hand_vis.egohand.joints21 import HAND21_BONES, joints_to_lines, vertices_by_side_to_joints21


def test_vertices_by_side_to_joints21_outputs_valid_masks():
    vertices = np.zeros((778, 3), dtype=np.float32)
    vertices[0] = [0.1, 0.2, 0.3]
    vertices[744] = [1.0, 2.0, 3.0]
    regressor = np.zeros((16, 778), dtype=np.float32)
    regressor[0, 0] = 1.0

    joints, valid = vertices_by_side_to_joints21(
        {"ego_left": [vertices, None], "ego_right": [None, vertices]},
        {"ego_left": regressor, "ego_right": regressor},
    )

    assert joints["ego_left"].shape == (2, 21, 3)
    assert valid["ego_left"].tolist() == [True, False]
    assert valid["ego_right"].tolist() == [False, True]
    np.testing.assert_allclose(joints["ego_left"][0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(joints["ego_left"][0, 4], [1.0, 2.0, 3.0])


def test_joints_to_lines_uses_hand21_bones():
    joints = np.arange(63, dtype=np.float32).reshape(21, 3)

    lines = joints_to_lines(joints)

    assert len(lines) == len(HAND21_BONES)
    np.testing.assert_allclose(lines[0], joints[[0, 1]])
    np.testing.assert_allclose(lines[-1], joints[[19, 20]])
