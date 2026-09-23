"""High-level preparation of a raw session for OpenVINS replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from roboego_hand_vis.common.video_paths import find_named_video

try:
    from .calibration import (
        OpenVinsCalibration,
        prepare_openvins_calibration,
        prepare_rectified_openvins_calibration,
    )
    from .session import PreparedSession, prepare_session
except ImportError:
    from calibration import (
        OpenVinsCalibration,
        prepare_openvins_calibration,
        prepare_rectified_openvins_calibration,
    )
    from session import PreparedSession, prepare_session


@dataclass(frozen=True)
class PreparedOpenVinsRun:
    session: PreparedSession
    calibration: OpenVinsCalibration
    manifest_path: Path


def _video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        return {
            "path": str(path),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def prepare_openvins_run(
    session_dir: str | Path,
    output_dir: str | Path,
    *,
    input_mode: str = "raw",
    tolerance_s: float = 0.001,
    downsample_cameras: bool = True,
    max_imu_gap_s: float = 0.05,
    max_imu_boundary_gap_s: float = 0.1,
) -> PreparedOpenVinsRun:
    if input_mode not in {"raw", "rectified"}:
        raise ValueError("input_mode must be either 'raw' or 'rectified'")
    if max_imu_gap_s <= 0:
        raise ValueError("max_imu_gap_s must be positive")
    if max_imu_boundary_gap_s < 0:
        raise ValueError("max_imu_boundary_gap_s must be non-negative")
    session = Path(session_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if input_mode == "rectified":
        manifest_path = session / "dataset_manifest" / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        left_video = Path(manifest["videos"]["left"]["path"]).expanduser().resolve()
        right_video = Path(manifest["videos"]["right"]["path"]).expanduser().resolve()
    else:
        left_video = find_named_video(session / "videos", "left")
        right_video = find_named_video(session / "videos", "right")
    videos = {"left": _video_info(left_video), "right": _video_info(right_video)}
    prepared_session = prepare_session(
        session, output / "input", tolerance_s=tolerance_s
    )
    if input_mode == "rectified":
        stereo_camera_path = session / "calibrations" / "camera-camchain.yaml"
        calibration = prepare_rectified_openvins_calibration(
            session / "dataset_manifest" / "dataset_manifest.json",
            session / "calibrations" / "camera_imu-camchain-imucam.yaml",
            session / "calibrations" / "camera_imu-imu.yaml",
            session / "calibrations" / "rectify_params.yaml",
            output / "config",
            stereo_camera_path=(
                stereo_camera_path if stereo_camera_path.exists() else None
            ),
            downsample_cameras=downsample_cameras,
        )
    else:
        calibration = prepare_openvins_calibration(
            session / "calibrations" / "camera_imu-camchain-imucam.yaml",
            session / "calibrations" / "camera_imu-imu.yaml",
            output / "config",
            downsample_cameras=downsample_cameras,
        )
    input_report = json.loads(prepared_session.report_path.read_text(encoding="utf-8"))

    print("\n========== VIO INPUT DEBUG ==========")
    print(f"session: {session}")
    print(f"input_report: {prepared_session.report_path}")
    print(f"stereo report:")
    print(json.dumps(input_report["stereo"], indent=2, ensure_ascii=False))

    for camera_id, camera in enumerate(("left", "right")):
        timestamp_count = int(input_report["stereo"][f"{camera}_source_count"])

        print(f"\n[{camera}]")
        print(f"video path       : {videos[camera]['path']}")
        print(f"video frame_count: {videos[camera]['frame_count']}")
        print(f"video fps        : {videos[camera]['fps']}")
        print(f"video resolution : {videos[camera]['width']}x{videos[camera]['height']}")
        print(f"timestamp_count  : {timestamp_count}")
        print(
            f"frame-ts diff    : "
            f"{videos[camera]['frame_count'] - timestamp_count}"
        )

        if videos[camera]["frame_count"] != timestamp_count:
            raise ValueError(
                f"{camera} video has {videos[camera]['frame_count']} frames "
                f"but {timestamp_count} timestamps"
            )
    # for camera_id, camera in enumerate(("left", "right")):
    #     timestamp_count = int(input_report["stereo"][f"{camera}_source_count"])
    #     if videos[camera]["frame_count"] != timestamp_count:
    #         raise ValueError(
    #             f"{camera} video has {videos[camera]['frame_count']} frames but {timestamp_count} timestamps"
    #         )
        video_resolution = (videos[camera]["width"], videos[camera]["height"])
        calibration_resolution = calibration.camera_resolutions[camera_id]
        if video_resolution != calibration_resolution:
            raise ValueError(
                f"{camera} video resolution {video_resolution} does not match calibration "
                f"{calibration_resolution}"
            )
        if calibration.camera_models[camera_id] != "pinhole":
            raise ValueError(
                f"Unsupported {camera} camera model: {calibration.camera_models[camera_id]}"
            )
        if calibration.distortion_models[camera_id] not in {"equidistant", "radtan"}:
            raise ValueError(
                f"Unsupported {camera} distortion model: {calibration.distortion_models[camera_id]}"
            )

    imu_report = input_report["imu"]
    stereo_report = input_report["stereo"]
    if int(imu_report["sample_count_clean"]) < 2:
        raise ValueError("At least two unique IMU samples are required")
    first_required_imu_timestamp = (
        float(stereo_report["first_timestamp"]) + calibration.timeshift_cam_imu_s
    )
    last_required_imu_timestamp = (
        float(stereo_report["last_timestamp"]) + calibration.timeshift_cam_imu_s
    )
    first_boundary_gap_s = max(
        0.0, float(imu_report["first_timestamp"]) - first_required_imu_timestamp
    )
    last_boundary_gap_s = max(
        0.0, last_required_imu_timestamp - float(imu_report["last_timestamp"])
    )
    if first_boundary_gap_s > max_imu_boundary_gap_s:
        raise ValueError(
            "IMU data starts after the first synchronized camera measurement"
        )
    if last_boundary_gap_s > max_imu_boundary_gap_s:
        raise ValueError(
            "IMU data ends before the last synchronized camera measurement"
        )
    max_period_s = float(imu_report["max_period_s"])
    if max_period_s > max_imu_gap_s:
        raise ValueError(
            f"IMU gap {max_period_s:.6f}s exceeds limit {max_imu_gap_s:.6f}s"
        )
    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "input_mode": input_mode,
        "session_dir": str(session),
        "output_dir": str(output),
        "videos": videos,
        "stereo_pairs_path": str(prepared_session.stereo_pairs_path),
        "imu_clean_path": str(prepared_session.imu_clean_path),
        "input_report_path": str(prepared_session.report_path),
        "estimator_config_path": str(calibration.estimator_config_path),
        "camera_config_path": str(calibration.camera_config_path),
        "imu_config_path": str(calibration.imu_config_path),
        "T_imu_camera0": calibration.transform_imu_camera.tolist(),
        "timeshift_cam_imu_s": calibration.timeshift_cam_imu_s,
        "camera_model": "rectified_pinhole" if input_mode == "rectified" else "raw_equidistant",
        "downsample_cameras": downsample_cameras,
        "input_validation": {
            "max_imu_gap_s": max_imu_gap_s,
            "max_imu_boundary_gap_s": max_imu_boundary_gap_s,
            "first_imu_boundary_gap_s": first_boundary_gap_s,
            "last_imu_boundary_gap_s": last_boundary_gap_s,
            "video_calibration_resolution_match": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreparedOpenVinsRun(
        session=prepared_session,
        calibration=calibration,
        manifest_path=manifest_path,
    )
