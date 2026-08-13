from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def quat_wxyz_to_rotation(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError(f"Invalid zero-norm quaternion: {q}")
    q = q / norm
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CameraWorldPose:
    """Camera-to-world transform for one frame.

    `R_cw` and `t_cw` are the dataset pose that maps world points into camera
    points. The camera-to-world conversion is therefore:

        p_world = R_cw.T @ (p_camera - t_cw)

    This matches the Ropedia `direct_T_cam_world` validation.
    """

    frame_idx: int
    R_cw: np.ndarray
    t_cw: np.ndarray

    @property
    def camera_center_world(self) -> np.ndarray:
        return -self.R_cw.T @ self.t_cw

    def camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera, dtype=np.float64)
        return (self.R_cw.T @ (points - self.t_cw).T).T.astype(np.float32)


class RopediaSlamTrajectory:
    def __init__(self, annotation_hdf5: Path):
        self.annotation_hdf5 = Path(annotation_hdf5).expanduser().resolve()
        with h5py.File(self.annotation_hdf5, "r") as f:
            self.trans_xyz = np.asarray(f["slam/trans_xyz"], dtype=np.float64)
            self.quat_wxyz = np.asarray(f["slam/quat_wxyz"], dtype=np.float64)

    def pose(self, frame_idx: int) -> CameraWorldPose:
        if frame_idx < 0 or frame_idx >= len(self.trans_xyz):
            raise IndexError(f"Frame {frame_idx} outside trajectory length {len(self.trans_xyz)}")
        return CameraWorldPose(
            frame_idx=frame_idx,
            R_cw=quat_wxyz_to_rotation(self.quat_wxyz[frame_idx]),
            t_cw=self.trans_xyz[frame_idx],
        )
