from pathlib import Path

import numpy as np

from roboego_hand_vis.objects.contact import contact_payload_from_joints_npz_and_object_points


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
