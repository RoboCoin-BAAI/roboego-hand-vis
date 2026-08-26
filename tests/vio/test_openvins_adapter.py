from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from roboego_hand_vis.vio.calibration import prepare_openvins_calibration
from roboego_hand_vis.vio.prepare import prepare_openvins_run
from roboego_hand_vis.vio.preview import fit_topdown_projection
from roboego_hand_vis.vio.session import (
    ImuSample,
    load_camera_timestamps,
    load_imu_samples,
    match_stereo_timestamps,
    prepare_session,
)
from roboego_hand_vis.vio.timing import summarize_openvins_timing
from roboego_hand_vis.vio.trajectory import (
    camera_pose_from_imu_pose,
    trajectory_metrics,
)


def write_timestamps(path: Path, values: list[float]) -> None:
    path.write_text(
        "".join(f"{idx} {value:.9f}\n" for idx, value in enumerate(values)),
        encoding="utf-8",
    )


def write_imu(path: Path, timestamps: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w",
                "angular_velocity_x",
                "angular_velocity_y",
                "angular_velocity_z",
                "linear_acceleration_x",
                "linear_acceleration_y",
                "linear_acceleration_z",
            ]
        )
        for timestamp in timestamps:
            writer.writerow([timestamp, 0, 0, 0, 0, 0.1, 0.2, 0.3, 1.0, 2.0, 9.5])


def write_calibration(camchain_path: Path, imu_path: Path) -> None:
    camchain_path.parent.mkdir(parents=True, exist_ok=True)
    camchain = {
        "cam0": {
            "T_cam_imu": [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "cam_overlaps": [1],
            "camera_model": "pinhole",
            "distortion_coeffs": [0.1, 0.2, 0.3, 0.4],
            "distortion_model": "equidistant",
            "intrinsics": [500.0, 501.0, 320.0, 240.0],
            "resolution": [640, 480],
            "rostopic": "/source/left",
            "timeshift_cam_imu": -0.04,
        },
        "cam1": {
            "T_cam_imu": [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "T_cn_cnm1": np.eye(4).tolist(),
            "cam_overlaps": [0],
            "camera_model": "pinhole",
            "distortion_coeffs": [0.1, 0.2, 0.3, 0.4],
            "distortion_model": "equidistant",
            "intrinsics": [502.0, 503.0, 321.0, 241.0],
            "resolution": [640, 480],
            "rostopic": "/source/right",
            "timeshift_cam_imu": -0.041,
        },
    }
    imu = {
        "imu0": {
            "T_i_b": np.eye(4).tolist(),
            "accelerometer_noise_density": 0.01,
            "accelerometer_random_walk": 0.001,
            "gyroscope_noise_density": 0.0001,
            "gyroscope_random_walk": 0.00001,
            "model": "calibrated",
            "rostopic": "/source/imu",
            "time_offset": 0.0,
            "update_rate": 200,
        }
    }
    camchain_path.write_text(
        yaml.safe_dump(camchain, sort_keys=False), encoding="utf-8"
    )
    imu_path.write_text(yaml.safe_dump(imu, sort_keys=False), encoding="utf-8")


def load_opencv_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("%YAML:1.0\n")
    return yaml.safe_load(text.split("\n", 1)[1])


def test_match_stereo_timestamps_uses_time_not_frame_index(tmp_path: Path) -> None:
    left_path = tmp_path / "left.txt"
    right_path = tmp_path / "right.txt"
    write_timestamps(left_path, [1.1, 1.2, 1.3, 1.3, 1.5])
    write_timestamps(right_path, [1.0, 1.1, 1.2, 1.3, 1.5])

    left = load_camera_timestamps(left_path)
    right = load_camera_timestamps(right_path)
    result = match_stereo_timestamps(left, right, tolerance_s=0.001)

    assert [
        (pair.left_frame_index, pair.right_frame_index) for pair in result.pairs
    ] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (4, 4),
    ]
    assert result.left_duplicate_count == 1
    assert result.right_duplicate_count == 0
    assert result.left_unmatched_count == 1
    assert result.right_unmatched_count == 1


def test_load_imu_samples_drops_duplicate_timestamps(tmp_path: Path) -> None:
    imu_path = tmp_path / "imu.csv"
    write_imu(imu_path, [1.0, 1.005, 1.005, 1.010])

    samples, report = load_imu_samples(imu_path)

    assert [sample.timestamp for sample in samples] == [1.0, 1.005, 1.010]
    assert all(isinstance(sample, ImuSample) for sample in samples)
    assert report["duplicate_count"] == 1
    assert samples[0].angular_velocity == (0.1, 0.2, 0.3)
    assert samples[0].linear_acceleration == (1.0, 2.0, 9.5)


def test_prepare_openvins_calibration_inverts_extrinsics(tmp_path: Path) -> None:
    camchain = tmp_path / "camchain.yaml"
    imu = tmp_path / "imu.yaml"
    write_calibration(camchain, imu)

    result = prepare_openvins_calibration(camchain, imu, tmp_path / "output")

    generated_camchain = load_opencv_yaml(result.camera_config_path)
    generated_imu = load_opencv_yaml(result.imu_config_path)
    generated_estimator = load_opencv_yaml(result.estimator_config_path)
    transform = np.asarray(generated_camchain["cam0"]["T_imu_cam"])
    np.testing.assert_allclose(transform[:3, 3], [-0.1, -0.2, -0.3])
    assert "T_cam_imu" not in generated_camchain["cam0"]
    assert generated_camchain["cam0"]["rostopic"] == "/roboego_vio/cam0/image_raw"
    assert generated_camchain["cam1"]["rostopic"] == "/roboego_vio/cam1/image_raw"
    assert generated_imu["imu0"]["rostopic"] == "/roboego_vio/imu"
    assert generated_imu["imu0"]["Tw"] == np.eye(3).tolist()
    assert generated_estimator["use_stereo"] is True
    assert generated_estimator["max_cameras"] == 2
    assert generated_estimator["calib_cam_extrinsics"] is False
    assert generated_estimator["downsample_cameras"] is True
    assert generated_estimator["init_dyn_use"] is True
    assert result.timeshift_cam_imu_s == -0.04
    for path in (
        result.camera_config_path,
        result.imu_config_path,
        result.estimator_config_path,
    ):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        assert storage.isOpened()
        storage.release()


def test_prepare_session_writes_sync_index_and_report(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "timestamps").mkdir(parents=True)
    write_timestamps(session / "timestamps" / "left_timestamps.txt", [1.1, 1.2])
    write_timestamps(session / "timestamps" / "right_timestamps.txt", [1.0, 1.1, 1.2])
    write_imu(session / "imu" / "imu0.csv", [1.0, 1.005, 1.010])

    result = prepare_session(session, tmp_path / "prepared")

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(result.stereo_pairs_path.open(encoding="utf-8")))
    assert report["stereo"]["matched_pair_count"] == 2
    assert report["imu"]["sample_count_clean"] == 3
    assert rows[0]["left_frame_index"] == "0"
    assert rows[0]["right_frame_index"] == "1"


def test_camera_pose_from_imu_pose_applies_camera_extrinsic() -> None:
    position_world_imu = np.array([1.0, 2.0, 3.0])
    orientation_world_imu_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    transform_imu_camera = np.eye(4)
    transform_imu_camera[:3, 3] = [0.1, -0.2, 0.3]

    transform_world_camera, orientation_world_camera = camera_pose_from_imu_pose(
        position_world_imu,
        orientation_world_imu_xyzw,
        transform_imu_camera,
    )

    np.testing.assert_allclose(transform_world_camera[:3, 3], [1.1, 1.8, 3.3])
    np.testing.assert_allclose(orientation_world_camera, [0.0, 0.0, 0.0, 1.0])


def test_trajectory_metrics_reports_rate_distance_and_alignment() -> None:
    transform_a = np.eye(4)
    transform_b = np.eye(4)
    transform_b[:3, 3] = [3.0, 4.0, 0.0]
    poses = [
        {
            "timestamp_camera": 10.0,
            "T_world_camera": transform_a,
            "time_alignment_error_s": -0.001,
        },
        {
            "timestamp_camera": 12.0,
            "T_world_camera": transform_b,
            "time_alignment_error_s": 0.002,
        },
    ]

    metrics = trajectory_metrics(poses, wall_duration_s=4.0)

    assert metrics["trajectory_duration_s"] == 2.0
    assert metrics["trajectory_rate_hz"] == 0.5
    assert metrics["wall_output_rate_hz"] == 0.5
    assert metrics["path_length_m"] == 5.0
    assert metrics["net_displacement_m"] == 5.0
    assert metrics["max_abs_time_alignment_error_s"] == 0.002


def test_summarize_openvins_timing_reports_estimator_cost(tmp_path: Path) -> None:
    timing_path = tmp_path / "timing.txt"
    timing_path.write_text(
        "# timestamp,tracking,propagation,msckf,slam,delayed,marg,total\n"
        "1.0,0.002,0.001,0.001,0.000,0.000,0.001,0.005\n"
        "2.0,0.004,0.001,0.002,0.000,0.000,0.001,0.008\n",
        encoding="utf-8",
    )

    summary = summarize_openvins_timing(timing_path)

    assert summary["sample_count"] == 2
    assert summary["mean_total_ms"] == pytest.approx(6.5)
    assert summary["median_total_ms"] == pytest.approx(6.5)
    assert summary["equivalent_rate_hz"] == pytest.approx(1 / 0.0065)
    assert summary["mean_stage_ms"]["tracking"] == pytest.approx(3.0)


def prepare_fake_run_session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    (session / "timestamps").mkdir(parents=True)
    (session / "videos").mkdir()
    write_timestamps(session / "timestamps" / "left_timestamps.txt", [1.1, 1.2])
    write_timestamps(session / "timestamps" / "right_timestamps.txt", [1.0, 1.1, 1.2])
    write_imu(session / "imu" / "imu0.csv", [1.0, 1.05, 1.1, 1.15, 1.2])
    write_calibration(
        session / "calibrations" / "camera_imu-camchain-imucam.yaml",
        session / "calibrations" / "camera_imu-imu.yaml",
    )
    return session


def test_prepare_run_rejects_video_calibration_resolution_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = prepare_fake_run_session(tmp_path)

    def fake_video_info(path: Path) -> dict[str, object]:
        frame_count = 2 if path.name == "left.mp4" else 3
        return {
            "path": str(path),
            "frame_count": frame_count,
            "fps": 30.0,
            "width": 320,
            "height": 240,
        }

    monkeypatch.setattr("roboego_hand_vis.vio.prepare._video_info", fake_video_info)

    with pytest.raises(ValueError, match="does not match calibration"):
        prepare_openvins_run(session, tmp_path / "output", max_imu_gap_s=0.1)


def test_prepare_run_rejects_incomplete_imu_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = prepare_fake_run_session(tmp_path)
    write_imu(session / "imu" / "imu0.csv", [1.0, 1.05, 1.10])

    def fake_video_info(path: Path) -> dict[str, object]:
        frame_count = 2 if path.name == "left.mp4" else 3
        return {
            "path": str(path),
            "frame_count": frame_count,
            "fps": 30.0,
            "width": 640,
            "height": 480,
        }

    monkeypatch.setattr("roboego_hand_vis.vio.prepare._video_info", fake_video_info)

    with pytest.raises(ValueError, match="ends before"):
        prepare_openvins_run(
            session,
            tmp_path / "output",
            max_imu_gap_s=0.1,
            max_imu_boundary_gap_s=0.01,
        )


def test_openvins_root_option_controls_default_install_path(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_openvins"
    runner = Path(__file__).parents[2] / "scripts" / "vio" / "run_openvins_session.sh"
    environment = {**os.environ, "OPENVINS_INSTALL": ""}

    result = subprocess.run(
        [
            str(runner),
            "--session",
            str(tmp_path / "session"),
            "--output",
            str(tmp_path / "output"),
            "--openvins_root",
            str(missing_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert str(missing_root / "install_ros2" / "setup.bash") in result.stderr


def test_topdown_projection_preserves_scale_and_flips_world_y() -> None:
    positions = np.array([[-2.0, -1.0], [2.0, 1.0]])
    projection = fit_topdown_projection(positions, width=400, height=300, margin=20)

    pixels = projection.project(np.array([[0.0, 0.0], [1.0, 1.0]]))

    np.testing.assert_array_equal(pixels[0], [200, 150])
    assert pixels[1, 0] > pixels[0, 0]
    assert pixels[1, 1] < pixels[0, 1]
    assert pixels[1, 0] - pixels[0, 0] == pixels[0, 1] - pixels[1, 1]
