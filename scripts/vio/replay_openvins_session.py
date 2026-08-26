#!/usr/bin/env python3
"""Replay synchronized raw stereo and IMU measurements into OpenVINS."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu

from roboego_hand_vis.vio.trajectory import (
    camera_pose_from_imu_pose,
    quaternion_xyzw_to_rotation,
    trajectory_metrics,
)


@dataclass(frozen=True)
class PairRow:
    timestamp: float
    left_frame_index: int
    right_frame_index: int


@dataclass(frozen=True)
class ImuRow:
    timestamp: float
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]


def timestamp_message(timestamp: float) -> TimeMessage:
    nanoseconds = round(timestamp * 1_000_000_000)
    message = TimeMessage()
    message.sec = nanoseconds // 1_000_000_000
    message.nanosec = nanoseconds % 1_000_000_000
    return message


def message_timestamp(message: TimeMessage) -> float:
    return float(message.sec) + float(message.nanosec) * 1e-9


def load_pairs(path: Path) -> list[PairRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            PairRow(
                timestamp=float(row["timestamp"]),
                left_frame_index=int(row["left_frame_index"]),
                right_frame_index=int(row["right_frame_index"]),
            )
            for row in csv.DictReader(handle)
        ]


def load_imu(path: Path) -> list[ImuRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            ImuRow(
                timestamp=float(row["timestamp"]),
                angular_velocity=(
                    float(row["angular_velocity_x"]),
                    float(row["angular_velocity_y"]),
                    float(row["angular_velocity_z"]),
                ),
                linear_acceleration=(
                    float(row["linear_acceleration_x"]),
                    float(row["linear_acceleration_y"]),
                    float(row["linear_acceleration_z"]),
                ),
            )
            for row in csv.DictReader(handle)
        ]


class VideoCursor:
    def __init__(self, path: Path, start_frame_index: int = 0) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        if start_frame_index < 0:
            raise ValueError("start_frame_index must be non-negative")
        self.next_frame_index = start_frame_index
        if start_frame_index and not self.capture.set(
            cv2.CAP_PROP_POS_FRAMES, start_frame_index
        ):
            raise RuntimeError(f"Could not seek {path} to frame {start_frame_index}")

    def read(self, frame_index: int) -> np.ndarray:
        if frame_index < self.next_frame_index:
            raise ValueError(f"Video frame requests went backwards for {self.path}")
        frame: np.ndarray | None = None
        while self.next_frame_index <= frame_index:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {self.next_frame_index} from {self.path}"
                )
            self.next_frame_index += 1
        return frame

    def close(self) -> None:
        self.capture.release()


class ReplayNode(Node):
    def __init__(
        self,
        pairs: list[PairRow],
        transform_imu_camera: np.ndarray,
        timeshift_cam_imu_s: float,
        pose_topic: str,
    ) -> None:
        super().__init__("roboego_openvins_replay")
        image_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.left_publisher = self.create_publisher(
            Image, "/roboego_vio/cam0/image_raw", image_qos
        )
        self.right_publisher = self.create_publisher(
            Image, "/roboego_vio/cam1/image_raw", image_qos
        )
        imu_qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE)
        self.imu_publisher = self.create_publisher(Imu, "/roboego_vio/imu", imu_qos)
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            pose_topic,
            self.record_pose,
            100,
        )
        self.pairs = pairs
        self.pair_timestamps = [pair.timestamp for pair in pairs]
        self.transform_imu_camera = transform_imu_camera
        self.timeshift_cam_imu_s = timeshift_cam_imu_s
        self.poses: list[dict[str, Any]] = []
        self._pose_timestamps: set[int] = set()

    def ready(self) -> bool:
        return (
            self.left_publisher.get_subscription_count() > 0
            and self.right_publisher.get_subscription_count() > 0
            and self.imu_publisher.get_subscription_count() > 0
        )

    def publish_imu(self, row: ImuRow) -> None:
        message = Imu()
        message.header.stamp = timestamp_message(row.timestamp)
        message.header.frame_id = "imu"
        message.orientation_covariance[0] = -1.0
        (
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
        ) = row.angular_velocity
        (
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        ) = row.linear_acceleration
        self.imu_publisher.publish(message)

    @staticmethod
    def image_message(frame: np.ndarray, timestamp: float, frame_id: str) -> Image:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        message = Image()
        message.header.stamp = timestamp_message(timestamp)
        message.header.frame_id = frame_id
        message.height, message.width = gray.shape
        message.encoding = "mono8"
        message.is_bigendian = 0
        message.step = int(gray.strides[0])
        message.data = gray.tobytes()
        return message

    def publish_pair(self, row: PairRow, left: np.ndarray, right: np.ndarray) -> None:
        self.left_publisher.publish(self.image_message(left, row.timestamp, "cam0"))
        self.right_publisher.publish(self.image_message(right, row.timestamp, "cam1"))

    def record_pose(self, message: PoseWithCovarianceStamped) -> None:
        timestamp_imu = message_timestamp(message.header.stamp)
        timestamp_key = round(timestamp_imu * 1_000_000_000)
        if timestamp_key in self._pose_timestamps:
            return
        self._pose_timestamps.add(timestamp_key)
        timestamp_camera = timestamp_imu - self.timeshift_cam_imu_s
        insertion = bisect.bisect_left(self.pair_timestamps, timestamp_camera)
        candidates = [
            idx for idx in (insertion - 1, insertion) if 0 <= idx < len(self.pairs)
        ]
        if not candidates:
            return
        pair_index = min(
            candidates,
            key=lambda idx: abs(self.pairs[idx].timestamp - timestamp_camera),
        )
        pair = self.pairs[pair_index]
        position = np.array(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        transform_world_imu = np.eye(4, dtype=np.float64)
        transform_world_imu[:3, :3] = quaternion_xyzw_to_rotation(quaternion)
        transform_world_imu[:3, 3] = position
        transform_world_camera, camera_quaternion = camera_pose_from_imu_pose(
            position,
            quaternion,
            self.transform_imu_camera,
        )
        self.poses.append(
            {
                "timestamp_camera": pair.timestamp,
                "timestamp_imu": timestamp_imu,
                "frame_index": pair.left_frame_index,
                "right_frame_index": pair.right_frame_index,
                "time_alignment_error_s": timestamp_camera - pair.timestamp,
                "tracking_status": "valid",
                "position_world_imu": position.tolist(),
                "orientation_world_imu_xyzw": quaternion.tolist(),
                "T_world_imu": transform_world_imu.tolist(),
                "position_world_camera": transform_world_camera[:3, 3].tolist(),
                "orientation_world_camera_xyzw": camera_quaternion.tolist(),
                "T_world_camera": transform_world_camera.tolist(),
                "pose_covariance": list(message.pose.covariance),
            }
        )


def wait_until(node: ReplayNode, target_time: float) -> None:
    while True:
        remaining = target_time - time.monotonic()
        if remaining <= 0:
            return
        rclpy.spin_once(node, timeout_sec=min(remaining, 0.002))


def save_trajectory(
    output_dir: Path, poses: list[dict[str, Any]], report: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trajectory.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for pose in poses:
            handle.write(json.dumps(pose, separators=(",", ":")) + "\n")
    np.savez_compressed(
        output_dir / "trajectory.npz",
        timestamp_camera=np.asarray(
            [pose["timestamp_camera"] for pose in poses], dtype=np.float64
        ),
        timestamp_imu=np.asarray(
            [pose["timestamp_imu"] for pose in poses], dtype=np.float64
        ),
        frame_index=np.asarray([pose["frame_index"] for pose in poses], dtype=np.int64),
        right_frame_index=np.asarray(
            [pose["right_frame_index"] for pose in poses], dtype=np.int64
        ),
        T_world_imu=np.asarray(
            [pose["T_world_imu"] for pose in poses], dtype=np.float64
        ).reshape((-1, 4, 4)),
        T_world_camera=np.asarray(
            [pose["T_world_camera"] for pose in poses], dtype=np.float64
        ).reshape((-1, 4, 4)),
        pose_covariance=np.asarray(
            [pose["pose_covariance"] for pose in poses], dtype=np.float64
        ).reshape((-1, 6, 6)),
    )
    report["artifacts"] = {
        "trajectory_jsonl": str(jsonl_path),
        "trajectory_npz": str(output_dir / "trajectory.npz"),
    }
    (output_dir / "run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_pairs = load_pairs(Path(manifest["stereo_pairs_path"]))
    all_imu = load_imu(Path(manifest["imu_clean_path"]))
    if not all_pairs or not all_imu:
        raise RuntimeError("Prepared session contains no stereo pairs or IMU samples")
    start_timestamp = all_pairs[0].timestamp + args.start_s
    end_timestamp = (
        start_timestamp + args.duration_s
        if args.duration_s is not None
        else float("inf")
    )
    pairs = [
        pair for pair in all_pairs if start_timestamp <= pair.timestamp <= end_timestamp
    ]
    imu_start = start_timestamp - args.imu_preroll_s
    imu_end = end_timestamp + args.imu_postroll_s
    imu = [row for row in all_imu if imu_start <= row.timestamp <= imu_end]
    if not pairs or not imu:
        raise ValueError("Selected replay interval contains no synchronized data")

    rclpy.init()
    node = ReplayNode(
        all_pairs,
        np.asarray(manifest["T_imu_camera0"], dtype=np.float64),
        float(manifest["timeshift_cam_imu_s"]),
        args.pose_topic,
    )
    left_cursor = VideoCursor(
        Path(manifest["videos"]["left"]["path"]),
        start_frame_index=pairs[0].left_frame_index,
    )
    right_cursor = VideoCursor(
        Path(manifest["videos"]["right"]["path"]),
        start_frame_index=pairs[0].right_frame_index,
    )
    decoder_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stereo_decode")
    wall_start = time.monotonic()
    try:
        deadline = time.monotonic() + args.subscriber_timeout_s
        while not node.ready() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.ready():
            raise RuntimeError("OpenVINS subscribers did not connect before timeout")

        event_start = min(imu[0].timestamp, pairs[0].timestamp)
        wall_event_start = time.monotonic() + 0.25
        imu_index = 0
        pair_index = 0
        published_imu = 0
        published_pairs = 0
        pacing_clock_slip_s = 0.0
        max_event_lateness_s = 0.0
        while imu_index < len(imu) or pair_index < len(pairs):
            next_imu = (
                imu[imu_index].timestamp if imu_index < len(imu) else float("inf")
            )
            next_pair = (
                pairs[pair_index].timestamp if pair_index < len(pairs) else float("inf")
            )
            event_timestamp = min(next_imu, next_pair)
            target_wall_time = (
                wall_event_start + (event_timestamp - event_start) / args.rate
            )
            lateness_s = time.monotonic() - target_wall_time
            if lateness_s > 0:
                # Slow the replay clock after a blocking decode instead of
                # bursting queued 200 Hz IMU samples into OpenVINS.
                wall_event_start += lateness_s
                target_wall_time += lateness_s
                pacing_clock_slip_s += lateness_s
                max_event_lateness_s = max(max_event_lateness_s, lateness_s)
            wait_until(node, target_wall_time)
            if next_imu <= next_pair:
                node.publish_imu(imu[imu_index])
                imu_index += 1
                published_imu += 1
            else:
                pair = pairs[pair_index]
                left_future = decoder_pool.submit(
                    left_cursor.read, pair.left_frame_index
                )
                right_future = decoder_pool.submit(
                    right_cursor.read, pair.right_frame_index
                )
                left = left_future.result()
                right = right_future.result()
                node.publish_pair(pair, left, right)
                pair_index += 1
                published_pairs += 1
            rclpy.spin_once(node, timeout_sec=0.0)

        drain_deadline = time.monotonic() + args.drain_s
        while time.monotonic() < drain_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        poses = sorted(node.poses, key=lambda pose: pose["timestamp_camera"])
        output_dir = Path(manifest["output_dir"]) / "trajectory"
        wall_duration_s = time.monotonic() - wall_start
        report = {
            "schema_version": "1.0",
            "manifest_path": str(manifest_path),
            "start_s": args.start_s,
            "duration_s": args.duration_s,
            "replay_rate": args.rate,
            "published_stereo_pairs": published_pairs,
            "published_imu_samples": published_imu,
            "pose_count": len(poses),
            "initialized": bool(poses),
            "wall_duration_s": wall_duration_s,
            "effective_dataset_rate": (
                (pairs[-1].timestamp - pairs[0].timestamp) / wall_duration_s
                if wall_duration_s > 0
                else 0.0
            ),
            "pacing_clock_slip_s": pacing_clock_slip_s,
            "max_event_lateness_s": max_event_lateness_s,
            "imu_publisher_qos": {"reliability": "reliable", "depth": 1000},
        }
        if poses:
            report["first_pose_camera_timestamp"] = poses[0]["timestamp_camera"]
            report["last_pose_camera_timestamp"] = poses[-1]["timestamp_camera"]
            report["first_pose_frame_index"] = poses[0]["frame_index"]
            report["last_pose_frame_index"] = poses[-1]["frame_index"]
            report["trajectory_metrics"] = trajectory_metrics(poses, wall_duration_s)
        save_trajectory(output_dir, poses, report)
        print(json.dumps(report, indent=2))
        return 0 if poses else 2
    finally:
        left_cursor.close()
        right_cursor.close()
        decoder_pool.shutdown(wait=True)
        node.destroy_node()
        rclpy.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--duration_s", type=float, default=None)
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Dataset seconds per wall-clock second."
    )
    parser.add_argument("--imu_preroll_s", type=float, default=0.2)
    parser.add_argument("--imu_postroll_s", type=float, default=0.2)
    parser.add_argument("--subscriber_timeout_s", type=float, default=15.0)
    parser.add_argument("--drain_s", type=float, default=3.0)
    parser.add_argument("--pose_topic", default="/ov_msckf/poseimu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
