# Local Validation - 2026-09-26

Status: dependency/build environment validated; full VIO trajectory NOT validated.

Image: `roboego-vio-env:dev`

Image ID:
`sha256:6383981a433e6b433ff299e4ed2c873ad71042ebf19e8833ff7fe82200021ffe`

Base: Ubuntu 22.04 pinned by digest. Compiler GCC 11.4.0; CMake 3.22.1;
OpenCV 4.5.4; Eigen 3.4.0; Boost 1.74.0; Ceres 2.0.0; system Python 3.10.
All runtime and compilation checks used `--network none` and read-only sources.
No host Conda, ROS libraries, CUDA libraries or host-built binaries were mounted
as runtime dependencies. No algorithm source or parameters were changed.

## Passed

- Docker build completed.
- Doctor imports: cv2, numpy, scipy, yaml, project VIO module.
- Existing `tests/vio` suite: 1 passed.
- Official OpenVINS `ov_msckf_lib` compiled from source: all 35 build steps passed.
- Built `libov_msckf_lib.so` dynamically loaded successfully inside container.
- Launcher and entry script passed `bash -n`.
- Failure logs retained traceback and nonzero exit status on host.

Source versions: roboego-hand-vis `b6328a91522c1ab1dfe2f77f851eb12977b2e9bf`;
OpenVINS `69488123ed9362dd44b6f28e7f4680abbff1442b`.
OpenVINS has pre-existing untracked build/install/log directories; these were
not changed or used to compile the container's core library.

## Blockers Confirmed By Execution

1. Default `build` fails with `unknown target 'roboego_openvins_offline'`.
   The current project stage requires this custom CLI, absent from local official
   OpenVINS. User expects the other machine's checkout to be official too.
   Whether an adapter was omitted or entrypoint migration is incomplete remains
   unknown. Old ROS2 replay scripts exist, but were not substituted silently.
2. `prepare` on `/home/user/下载/sample` fails with `Expected a YAML mapping`.
   Both `camera_imu-camchain-imucam.yaml` and `camera_imu-imu.yaml` contain only
   `calibration incomplete`. Stereo-only calibration is insufficient for VIO.
   No unrelated calibration was substituted. No trajectory was produced.

Build products: `tmp/vio_container/build/openvins/`.
Logs: `tmp/vio_container/output/logs/`.
Partial preparation outputs: `tmp/vio_container/output/run/`; do not treat
these as a successful prepared session.

Next acceptance needs the offline runner integration and a raw stereo session
with valid camera-IMU calibration, synchronized timestamps and IMU. Small-motion
initialization/tracking behavior has not been investigated or changed.
