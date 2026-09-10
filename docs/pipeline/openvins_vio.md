# Standalone OpenVINS VIO

This adapter runs OpenVINS independently of depth, HaMeR, hand tracking, and
semantic annotation. Its output can later be joined to those stages by camera
timestamp or source frame index.

## Inputs

The session must contain:

```text
session/
├── videos/left.mp4
├── videos/right.mp4
├── timestamps/left_timestamps.txt
├── timestamps/right_timestamps.txt
├── imu/imu0.csv
└── calibrations/
    ├── camera_imu-camchain-imucam.yaml
    └── camera_imu-imu.yaml
```

The adapter deliberately uses the **raw, unrectified** videos. OpenVINS tracks
features with the calibrated equidistant camera model. Rectifying first would
require a matching replacement calibration and would add an unnecessary image
resampling step.

Preparation does the following before launching the estimator:

1. Drops repeated camera and IMU timestamps.
2. Matches left and right frames one-to-one by timestamp rather than by frame
   number.
3. Verifies that timestamp counts agree with video frame counts.
4. Verifies raw video dimensions and camera models against the calibration.
5. Verifies IMU coverage and rejects gaps longer than 50 ms. Up to 100 ms is
   allowed at a session boundary because capture streams do not start on the
   same clock tick.
6. Converts Kalibr `T_cam_imu` into OpenVINS `T_imu_cam`.
7. Generates the OpenCV-YAML configuration consumed by OpenVINS.

## Environment

This workstation has the ROS 2 build at:

```text
/home/user/open_vins/install_ros2
```

The runner also supports a normal OpenVINS ROS 2 build. The standard dependency
and build commands are:

```bash
sudo apt-get install libeigen3-dev libboost-all-dev libceres-dev
cd /home/user/open_vins
source /opt/ros/humble/setup.bash
colcon --log-base log_ros2 build --build-base build_ros2 --install-base install_ros2 \
  --packages-select ov_core ov_init ov_msckf \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_ARUCO_TAGS=OFF
```

On this machine Ceres is installed without root under
`/home/user/.local/openvins-deps/usr`; the runner adds that library path
automatically.

## Run

Run a short interval first:

```bash
cd /home/user/roboego-hand-vis
scripts/vio/run_openvins_session.sh \
  --session /home/user/ego_data/session_20260804_224442 \
  --output tmp/openvins_session_224442_20s \
  --duration_s 20 \
  --rate 1.0
```

Useful options:

```text
--start_s SECONDS       Start relative to the first synchronized stereo pair
--duration_s SECONDS    Omit to process the rest of the session
--rate RATE             Dataset seconds replayed per wall-clock second
--no-downsample         Track at full 1920x1088 instead of OpenVINS half resolution
--openvins_root PATH    Use PATH/install_ros2 instead of /home/user/open_vins
```

The command launches OpenVINS in its own ROS domain, replays timestamped stereo
and IMU messages, records poses, and shuts all ROS processes down when finished.
No VLM credentials are used. IMU messages use a reliable, deep publisher queue.
When MP4 decoding falls behind, the replay clock slows down rather than sending
the accumulated 200 Hz IMU measurements as a burst into OpenVINS.

## Outputs

```text
output/
├── input/
│   ├── stereo_pairs.csv
│   ├── imu_clean.csv
│   └── input_report.json
├── config/
│   ├── estimator_config.yaml
│   ├── kalibr_imucam_chain.yaml
│   └── kalibr_imu_chain.yaml
├── trajectory/
│   ├── trajectory.jsonl
│   ├── trajectory.npz
│   └── run_report.json
├── run_manifest.json
├── openvins.log
├── openvins_timing.txt
├── openvins_state_estimate.txt
└── openvins_state_deviation.txt
```

`trajectory.npz` is the integration artifact. It contains camera and IMU
timestamps, original left/right frame indices, pose covariance, `T_world_imu`,
and `T_world_camera`. The convention is:

```text
p_world = T_world_camera @ p_camera
```

`run_report.json` records initialization status, pose count, effective replay
rate, path length, displacement, pose rate, and timestamp alignment error. These
are runtime and consistency measurements, not absolute trajectory accuracy;
accuracy requires ground truth or an external loop-closure reference.

## Rerun Visualization

Export the trajectory to Rerun from the project Python environment:

```bash
python \
  scripts/vio/export_trajectory_rerun.py \
  --trajectory tmp/openvins_session_224442_20s/trajectory/trajectory.npz \
  --output tmp/openvins_session_224442_20s/trajectory/trajectory.rrd

rerun \
  tmp/openvins_session_224442_20s/trajectory/trajectory.rrd
```

The Rerun recording reveals the trajectory as the timeline advances, highlights
the current camera center in bright yellow, and shows the estimated camera axes.

To include a scrub-friendly camera reference without storing all 30 FPS images,
sample the left camera as JPEG frames. For a long recording, the Rerun view can
also sample poses while the source `trajectory.npz` remains at its full rate:

```bash
python \
  scripts/vio/export_trajectory_rerun.py \
  --trajectory tmp/openvins_session_224442_full/trajectory/trajectory.npz \
  --output tmp/openvins_session_224442_full/trajectory/trajectory_with_left.rrd \
  --camera_video /home/user/ego_data/session_20260804_224442/videos/left.mp4 \
  --camera_timestamps /home/user/ego_data/session_20260804_224442/timestamps/left_timestamps.txt \
  --camera_sample_fps 0.5 \
  --camera_max_width 480 \
  --jpeg_quality 65 \
  --pose_sample_fps 2 \
  --trajectory_chunk_seconds 5
```

The path is appended in five-second chunks instead of being logged as a static
full-session line, so future positions are not visible before their timestamps.
The bright yellow marker is the current position. Sampling the camera and Rerun
pose logs keeps long recordings responsive and memory-bounded in the viewer.

For a long recording, do not put camera images into an RRD. Generate a
streaming H.264 comparison video instead. It holds only the current frame and
plot canvas in memory:

```bash
python \
  scripts/vio/export_camera_trajectory_video.py \
  --video /home/user/ego_data/session_20260804_224442/videos/left.mp4 \
  --trajectory tmp/openvins_session_224442_full/trajectory/trajectory.npz \
  --output tmp/openvins_session_224442_full/trajectory/left_camera_vio.mp4 \
  --fps 10
```

The left panel is the raw left camera. The right panel is the accumulated
OpenVINS XY trajectory and current camera heading. This exporter writes MP4
incrementally and does not create an RRD.

## Current Performance Boundary

The estimator and the replay adapter are measured separately. OpenVINS writes
per-update timings to `openvins_timing.txt`. The current Python/OpenCV replay
path must decode two 1920x1088 H.264 streams and publish ROS images, so it can be
slower than real time even when the estimator itself is comfortably faster than
30 Hz. This adapter is the offline reference path; a real-time version should
receive camera and IMU measurements directly from capture callbacks instead of
decoding MP4 files.
