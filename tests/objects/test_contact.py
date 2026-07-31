import numpy as np
import pytest

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
    assert result.min_distance_m == pytest.approx(0.01)
    assert result.contact_points == ["thumb_tip"]


@pytest.mark.parametrize(
    "object_points",
    [np.array([]), np.zeros((3,), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)],
)
def test_compute_fingertip_contact_rejects_invalid_object_point_shapes(object_points):
    with pytest.raises(ValueError, match="shape"):
        compute_fingertip_contact({}, object_points, threshold_m=0.025)


def test_compute_fingertip_contact_empty_object_cloud_returns_no_contact():
    result = compute_fingertip_contact(
        {"thumb_tip": np.zeros(3, dtype=np.float32)},
        np.zeros((0, 3), dtype=np.float32),
        threshold_m=0.025,
    )
    assert result.contact is False
    assert result.min_distance_m is None
    assert result.contact_points == []


def test_compute_fingertip_contact_empty_fingertips_returns_no_contact():
    result = compute_fingertip_contact(
        {}, np.zeros((1, 3), dtype=np.float32), threshold_m=0.025
    )
    assert result.contact is False
    assert result.min_distance_m is None
    assert result.contact_points == []


def test_compute_fingertip_contact_includes_threshold_boundary():
    result = compute_fingertip_contact(
        {"index_tip": np.array([0.025, 0.0, 0.0], dtype=np.float32)},
        np.zeros((1, 3), dtype=np.float32),
        threshold_m=0.025,
    )
    assert result.contact is True
    assert result.contact_points == ["index_tip"]


def test_compute_fingertip_contact_reports_nearest_distance_when_far():
    result = compute_fingertip_contact(
        {"index_tip": np.array([0.03, 0.0, 0.0], dtype=np.float32)},
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        threshold_m=0.025,
    )
    assert result.contact is False
    assert result.min_distance_m == pytest.approx(0.03)
    assert result.contact_points == []
