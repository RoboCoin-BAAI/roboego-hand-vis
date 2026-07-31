import numpy as np

from roboego_hand_vis.objects.contact import (
    FINGERTIP_NAMES,
    compute_fingertip_contact,
    extract_fingertips_from_joints21,
)


def test_extract_fingertips_from_joints21_uses_openpose_indices():
    joints = np.zeros((21, 3), dtype=np.float32)
    for idx in [4, 8, 12, 16, 20]:
        joints[idx] = [idx, idx + 1, idx + 2]
    fingertips = extract_fingertips_from_joints21(joints)
    assert list(fingertips.keys()) == FINGERTIP_NAMES
    assert fingertips["index_tip"].tolist() == [8.0, 9.0, 10.0]


def test_compute_fingertip_contact_marks_close_tip():
    object_points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    fingertips = {
        "thumb_tip": np.array([0.01, 0.0, 0.0], dtype=np.float32),
        "index_tip": np.array([0.2, 0.0, 0.0], dtype=np.float32),
    }
    result = compute_fingertip_contact(fingertips, object_points, threshold_m=0.025)
    assert result.contact is True
    assert result.min_distance_m == 0.01
    assert result.contact_points == ["thumb_tip"]
