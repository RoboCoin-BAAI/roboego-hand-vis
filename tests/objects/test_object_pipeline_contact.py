from pathlib import Path

import numpy as np
import pytest

from roboego_hand_vis.objects.contact import contact_payload_from_joints_npz_and_object_points


def _write_joints_npz(
    path: Path,
    *,
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
) -> None:
    np.savez(
        path,
        left_joints_21_camera_opencv=left,
        right_joints_21_camera_opencv=right,
        left_valid=left_valid,
        right_valid=right_valid,
    )


def test_contact_payload_from_joints_npz_and_object_points_marks_left_contact(tmp_path: Path):
    left = np.ones((1, 21, 3), dtype=np.float32)
    right = np.ones((1, 21, 3), dtype=np.float32)
    left[:, :, 2] = 0.2
    right[:, :, 2] = 0.2
    left[0, 8] = [0.01, 0.0, 0.0]
    npz_path = tmp_path / "joints_21_camera_opencv.npz"
    np.savez(
        npz_path,
        left_joints_21_camera_opencv=left,
        right_joints_21_camera_opencv=right,
        left_valid=np.array([True]),
        right_valid=np.array([True]),
    )
    object_points = {0: np.array([[0.0, 0.0, 0.0]], dtype=np.float32)}
    payload = contact_payload_from_joints_npz_and_object_points(
        object_id="yellow_object_0",
        hand_joints_npz=npz_path,
        object_points_by_frame=object_points,
        threshold_m=0.025,
    )
    frame = payload["frames"][0]
    assert payload["status"] == "ok"
    assert frame["left_contact"] is True
    assert frame["right_contact"] is False
    assert "left.index_tip" in frame["contact_points"]


def test_contact_payload_marks_empty_object_points_missing(tmp_path: Path):
    joints = np.zeros((1, 21, 3), dtype=np.float32)
    npz_path = tmp_path / "joints.npz"
    _write_joints_npz(
        npz_path,
        left=joints,
        right=joints,
        left_valid=np.array([True]),
        right_valid=np.array([True]),
    )

    payload = contact_payload_from_joints_npz_and_object_points(
        object_id="yellow_object_0",
        hand_joints_npz=npz_path,
        object_points_by_frame={0: np.zeros((0, 3), dtype=np.float32)},
        threshold_m=0.025,
    )

    frame = payload["frames"][0]
    assert "object_points_missing" in frame["status"]
    assert frame["left_contact"] is False
    assert frame["right_contact"] is False
    assert frame["left_min_distance_m"] is None
    assert frame["right_min_distance_m"] is None


def test_contact_payload_evaluates_valid_side_when_other_hand_is_missing(tmp_path: Path):
    left = np.zeros((1, 21, 3), dtype=np.float32)
    right = np.ones((1, 21, 3), dtype=np.float32)
    right[0, 8] = [0.01, 0.0, 0.0]
    npz_path = tmp_path / "joints.npz"
    _write_joints_npz(
        npz_path,
        left=left,
        right=right,
        left_valid=np.array([False]),
        right_valid=np.array([True]),
    )

    payload = contact_payload_from_joints_npz_and_object_points(
        object_id="yellow_object_0",
        hand_joints_npz=npz_path,
        object_points_by_frame={0: np.zeros((1, 3), dtype=np.float32)},
        threshold_m=0.025,
    )

    frame = payload["frames"][0]
    assert frame["left_contact"] is False
    assert frame["right_contact"] is True
    assert "left_hand_missing" in frame["status"]


def test_contact_payload_marks_nonfinite_geometry_and_evaluates_later_frame(tmp_path: Path):
    left = np.ones((2, 21, 3), dtype=np.float32)
    right = np.ones((2, 21, 3), dtype=np.float32)
    left[0, 8, 0] = np.nan
    right[1, 8] = [0.01, 0.0, 0.0]
    npz_path = tmp_path / "joints.npz"
    _write_joints_npz(
        npz_path,
        left=left,
        right=right,
        left_valid=np.array([True, True]),
        right_valid=np.array([True, True]),
    )

    payload = contact_payload_from_joints_npz_and_object_points(
        object_id="yellow_object_0",
        hand_joints_npz=npz_path,
        object_points_by_frame={
            0: np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
            1: np.zeros((1, 3), dtype=np.float32),
        },
        threshold_m=0.025,
    )

    assert "object_points_invalid" in payload["frames"][0]["status"]
    assert "left_invalid_geometry" in payload["frames"][0]["status"]
    assert payload["frames"][0]["left_min_distance_m"] is None
    assert payload["frames"][1]["right_contact"] is True


def test_contact_payload_rejects_missing_npz_key(tmp_path: Path):
    npz_path = tmp_path / "missing_key.npz"
    joints = np.zeros((1, 21, 3), dtype=np.float32)
    np.savez(
        npz_path,
        left_joints_21_camera_opencv=joints,
        right_joints_21_camera_opencv=joints,
        left_valid=np.array([True]),
    )

    with pytest.raises(ValueError) as exc_info:
        contact_payload_from_joints_npz_and_object_points(
            object_id="yellow_object_0",
            hand_joints_npz=npz_path,
            object_points_by_frame={},
            threshold_m=0.025,
        )

    assert str(npz_path) in str(exc_info.value)
    assert "right_valid" in str(exc_info.value)


@pytest.mark.parametrize(
    ("left_shape", "right_shape", "left_valid_shape", "right_valid_shape", "expected"),
    [
        ((1, 20, 3), (1, 21, 3), (1,), (1,), "left_joints_21_camera_opencv"),
        ((1, 21, 3), (2, 21, 3), (1,), (2,), "length"),
    ],
)
def test_contact_payload_rejects_malformed_npz_shapes_and_lengths(
    tmp_path: Path,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    left_valid_shape: tuple[int, ...],
    right_valid_shape: tuple[int, ...],
    expected: str,
):
    npz_path = tmp_path / "malformed.npz"
    _write_joints_npz(
        npz_path,
        left=np.zeros(left_shape, dtype=np.float32),
        right=np.zeros(right_shape, dtype=np.float32),
        left_valid=np.ones(left_valid_shape, dtype=bool),
        right_valid=np.ones(right_valid_shape, dtype=bool),
    )

    with pytest.raises(ValueError, match=expected):
        contact_payload_from_joints_npz_and_object_points(
            object_id="yellow_object_0",
            hand_joints_npz=npz_path,
            object_points_by_frame={},
            threshold_m=0.025,
        )
