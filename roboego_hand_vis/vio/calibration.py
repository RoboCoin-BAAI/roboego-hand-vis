"""Conversion from Kalibr calibration files to OpenVINS configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class _OpenCvYamlDumper(yaml.SafeDumper):
    """Emit the restricted YAML layout accepted by cv::FileStorage."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> int:
        return super().increase_indent(flow, False)

    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass(frozen=True)
class OpenVinsCalibration:
    output_dir: Path
    camera_config_path: Path
    imu_config_path: Path
    estimator_config_path: Path
    transform_imu_camera: np.ndarray
    timeshift_cam_imu_s: float
    camera_resolutions: tuple[tuple[int, int], tuple[int, int]]
    camera_models: tuple[str, str]
    distortion_models: tuple[str, str]


def _estimator_config(output_dir: Path, *, downsample_cameras: bool) -> dict[str, Any]:
    return {
        "verbosity": "INFO",
        "use_fej": True,
        "integration": "rk4",
        "use_stereo": True,
        "max_cameras": 2,
        "calib_cam_extrinsics": False,
        "calib_cam_intrinsics": False,
        "calib_cam_timeoffset": False,
        "calib_imu_intrinsics": False,
        "calib_imu_g_sensitivity": False,
        "max_clones": 11,
        "max_slam": 50,
        "max_slam_in_update": 25,
        "max_msckf_in_update": 40,
        "dt_slam_delay": 1,
        "gravity_mag": 9.81,
        "feat_rep_msckf": "GLOBAL_3D",
        "feat_rep_slam": "ANCHORED_MSCKF_INVERSE_DEPTH",
        "feat_rep_aruco": "ANCHORED_MSCKF_INVERSE_DEPTH",
        "try_zupt": False,
        "zupt_chi2_multipler": 0,
        "zupt_max_velocity": 0.1,
        "zupt_noise_multiplier": 10,
        "zupt_max_disparity": 0.5,
        "zupt_only_at_beginning": False,
        "init_window_time": 2.0,
        "init_imu_thresh": 1.5,
        "init_max_disparity": 10.0,
        "init_max_features": 50,
        "init_dyn_use": True,
        "init_dyn_mle_opt_calib": False,
        "init_dyn_mle_max_iter": 50,
        "init_dyn_mle_max_time": 0.05,
        "init_dyn_mle_max_threads": 6,
        "init_dyn_num_pose": 6,
        "init_dyn_min_deg": 10.0,
        "init_dyn_inflation_ori": 10,
        "init_dyn_inflation_vel": 100,
        "init_dyn_inflation_bg": 10,
        "init_dyn_inflation_ba": 100,
        "init_dyn_min_rec_cond": 1e-12,
        "init_dyn_bias_g": [0.0, 0.0, 0.0],
        "init_dyn_bias_a": [0.0, 0.0, 0.0],
        "record_timing_information": True,
        "record_timing_filepath": str(output_dir / "openvins_timing.txt"),
        "save_total_state": False,
        "filepath_est": str(output_dir / "openvins_state_estimate.txt"),
        "filepath_std": str(output_dir / "openvins_state_deviation.txt"),
        "filepath_gt": str(output_dir / "openvins_groundtruth.txt"),
        "use_klt": True,
        "num_pts": 200,
        "fast_threshold": 20,
        "grid_x": 5,
        "grid_y": 5,
        "min_px_dist": 10,
        "knn_ratio": 0.70,
        "track_frequency": 30.1,
        "downsample_cameras": downsample_cameras,
        "num_opencv_threads": 4,
        "histogram_method": "HISTOGRAM",
        "use_aruco": False,
        "num_aruco": 1024,
        "downsize_aruco": True,
        "up_msckf_sigma_px": 1,
        "up_msckf_chi2_multipler": 1,
        "up_slam_sigma_px": 1,
        "up_slam_chi2_multipler": 1,
        "up_aruco_sigma_px": 1,
        "up_aruco_chi2_multipler": 1,
        "use_mask": False,
        "relative_config_imu": "kalibr_imu_chain.yaml",
        "relative_config_imucam": "kalibr_imucam_chain.yaml",
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:"):
        text = text.split("\n", 1)[1]
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _write_opencv_yaml(path: Path, data: dict[str, Any]) -> None:
    body = yaml.dump(
        data,
        Dumper=_OpenCvYamlDumper,
        sort_keys=False,
        default_flow_style=None,
        width=4096,
    )
    path.write_text("%YAML:1.0\n" + body, encoding="utf-8")


def _validate_transform(value: Any, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not one")
    return transform


def _camera_config(
    source: dict[str, Any], source_id: int, output_id: int
) -> tuple[dict[str, Any], np.ndarray]:
    key = f"cam{source_id}"
    if key not in source:
        raise KeyError(f"Camera calibration does not contain {key}")
    raw = dict(source[key])
    required = (
        "T_cam_imu",
        "camera_model",
        "distortion_coeffs",
        "distortion_model",
        "intrinsics",
        "resolution",
        "timeshift_cam_imu",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"{key} calibration missing: {', '.join(missing)}")
    transform_cam_imu = _validate_transform(
        raw.pop("T_cam_imu"), name=f"{key}.T_cam_imu"
    )
    transform_imu_cam = np.linalg.inv(transform_cam_imu)
    raw.pop("T_cn_cnm1", None)
    raw["T_imu_cam"] = transform_imu_cam.tolist()
    raw["cam_overlaps"] = [1 - output_id]
    raw["rostopic"] = f"/roboego_vio/cam{output_id}/image_raw"
    return raw, transform_imu_cam


def prepare_openvins_calibration(
    camera_imu_path: str | Path,
    imu_path: str | Path,
    output_dir: str | Path,
    *,
    camera_ids: tuple[int, int] = (0, 1),
    downsample_cameras: bool = True,
) -> OpenVinsCalibration:
    """Write an OpenVINS config set for two raw Kalibr cameras."""
    camera_source_path = Path(camera_imu_path).expanduser().resolve()
    imu_source_path = Path(imu_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera_source = _load_yaml(camera_source_path)
    imu_source = _load_yaml(imu_source_path)
    cameras: dict[str, Any] = {}
    transforms: list[np.ndarray] = []
    for output_id, source_id in enumerate(camera_ids):
        camera, transform = _camera_config(camera_source, source_id, output_id)
        cameras[f"cam{output_id}"] = camera
        transforms.append(transform)

    if "imu0" not in imu_source:
        raise KeyError("IMU calibration does not contain imu0")
    imu = dict(imu_source["imu0"])
    imu["rostopic"] = "/roboego_vio/imu"
    identity = np.eye(3).tolist()
    zeros = np.zeros((3, 3)).tolist()
    imu.setdefault("Tw", identity)
    imu.setdefault("R_IMUtoGYRO", identity)
    imu.setdefault("Ta", identity)
    imu.setdefault("R_IMUtoACC", identity)
    imu.setdefault("Tg", zeros)

    camera_config_path = output / "kalibr_imucam_chain.yaml"
    imu_config_path = output / "kalibr_imu_chain.yaml"
    estimator_config_path = output / "estimator_config.yaml"
    _write_opencv_yaml(camera_config_path, cameras)
    _write_opencv_yaml(imu_config_path, {"imu0": imu})
    _write_opencv_yaml(
        estimator_config_path,
        _estimator_config(output.parent, downsample_cameras=downsample_cameras),
    )
    return OpenVinsCalibration(
        output_dir=output,
        camera_config_path=camera_config_path,
        imu_config_path=imu_config_path,
        estimator_config_path=estimator_config_path,
        transform_imu_camera=transforms[0],
        timeshift_cam_imu_s=float(cameras["cam0"]["timeshift_cam_imu"]),
        camera_resolutions=tuple(
            tuple(int(value) for value in cameras[f"cam{camera_id}"]["resolution"])
            for camera_id in range(2)
        ),
        camera_models=tuple(
            str(cameras[f"cam{camera_id}"]["camera_model"]) for camera_id in range(2)
        ),
        distortion_models=tuple(
            str(cameras[f"cam{camera_id}"]["distortion_model"])
            for camera_id in range(2)
        ),
    )
