# Local Validation - 2026-09-28

## Compatibility Boundary - 2026-09-29

The user requires this repository to change only under `docker/`. All earlier
changes to project VIO modules and tests have been restored to HEAD, including
the original machine-specific default configuration. `git diff --name-only`
now lists only Docker files. The following ownership-migration section records
the earlier intermediate state, not the final project integration.

OpenVINS now generates the original `roboego_openvins_offline` CLI as an
executable compatibility launcher, paired with `openvins_offline_native`.
The original project orchestrator calls it without changes; native replay,
trajectory export and compatibility tests are all owned by OpenVINS.
The launcher's output directory semantics match the old caller exactly.
No image rebuild is needed; reconfigure/rebuild the OpenVINS target.

Final checks: combined suite 13 passed; OpenVINS-only offline container suite
12 passed. The restored, unmodified project CLI completed a 20-second real raw
replay with 539 poses and reports/JSONL/NPZ in the expected directory, without
an extra nested `trajectory` directory. Output:
`/home/user/ego_data/vio_legacy_compat_20260929`. Rectified real-data replay is
still not part of this compatibility check.

## Ownership Migration - 2026-09-29 (Intermediate)

Replay/export/tests moved to `/home/user/open_vins/ov_offline`. The official
estimator source and defaults are unchanged. The native target is part of the
OpenVINS ROS-free CMake build. This repository retains data preparation and the
public `python -m roboego_hand_vis.vio.run` orchestration interface, including
`--input_mode rectified`. Its old native adapter and trajectory exporter were
removed rather than duplicated.

- Existing dependency image reused; no new packages or image rebuild.
- Build and 10 OpenVINS tests passed with only OpenVINS/build mounts, no project
  mount, and networking disabled.
- Combined project and OpenVINS tests: 13 passed, including rectified CLI
  delegation and independent input flags without a project manifest.
- Four full raw-session replays under `/home/user/ego_data/vio_migration_20260929`
  match the previous output frame indices and timestamps exactly.
- Maximum absolute camera-transform element differences versus the preceding
  run: 1.53e-5, 1.18e-6, 2.86e-6 and 0.0911 respectively. The divergent cashier
  case is not numerically identical and still fails plausibility; no improvement
  or bitwise estimator determinism is claimed.
- Real rectified replay has not been revalidated in this migration; its CLI
  parameter contract is covered, while raw mode supplies the numerical replay
  comparison above.

Both working trees must be versioned/transferred together. Changes are local,
not published to a GitHub fork by this migration.

## Offline Integration Update

The earlier missing-runner blocker below is resolved by the adapter in
`roboego_hand_vis/vio/native` (subsequently moved to OpenVINS `ov_offline`), linked
against the unchanged official library.
Default `run.sh build` now compiles both. The existing image is reused; the
launcher executes the mounted entry script. Integration tests exercise the
actual binary, including empty initialization, stationary initialization with
official ZUPT enabled, trajectory export, input errors and selected time ranges.
They do not replace real-device acceptance.

Latest result: default adapter build passed; `tests/vio` **11 passed** inside
the offline container; shell syntax checks and `git diff --check` passed.

A 100-frame synthetic stationary scene reproduces `no accel jerk detected`
with default settings. Enabling official ZUPT in this test permits initialization
and finite stationary poses. The adapter also correctly exports ZUPT-only states
before the first visual update. Production initialization parameters are unchanged.

All four sessions in `/home/user/ego_data/processed_data` originally had camera-IMU
YAMLs containing only `calibration incomplete`. At the user's direction, both
files were replaced from `/home/user/下载/session_20260804_223748/calibrations`.
Original placeholders are backed up under
`/home/user/ego_data/calibration_backups/20260928_225959`. Other calibration and
input files were not modified. All four sessions now pass input preparation.
Physical device applicability of this calibration is supplied by the user,
not independently established by these software checks.

Reproduce checks from the repository root:

```bash
export INPUT_DIR=/home/user/ego_data/processed_data
bash docker/vio-env/run.sh build
bash docker/vio-env/run.sh test
```

## Real-Data Replay - 2026-09-28

Results: `/home/user/ego_data/vio_full_20260928_230109`. Default parameters,
raw stereo inputs, no algorithm changes. All four complete replay and emit
finite poses at approximately 30 Hz, with no missing pose after initialization.
The first two camera frames are outside shifted IMU coverage and are skipped.
Initialization takes approximately 2.0-2.13 seconds after the first fed frame.

| Session suffix | Source frames | Poses | Pose duration | Path length | Replay wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260827190358 | 2137 | 2071 | 69.0 s | 19.04 m | 21.10 s |
| 20260828102934 | 714 | 652 | 21.7 s | 3.99 m | 7.23 s |
| 20260831144710 | 2799 | 2737 | 91.2 s | 24.01 m | 28.25 s |
| 20260903162506 | 1360 | 1298 | 43.23 s | **1983.76 m** | 11.98 s |

The cashier session (last row) **fails trajectory plausibility**, despite exit 0
and `status: complete`. These fields describe execution, not accuracy or quality
acceptance. It has 1185/1298 initialized frames with neither exported MSCKF
features nor SLAM landmarks, and a peak pose-difference speed near 694 m/s.
This is a concrete divergence case, not a failure to initialize. The first three
have no comparable magnitude anomaly, but there is no ground truth accuracy
validation for any session.

Controlled probes are under `/home/user/ego_data/vio_cashier_diagnosis_20260928`:
the first 10 seconds with INFO logging produce 37.32 m path length; changing only
`try_zupt` to true produces essentially the same result. Both dynamically
initialize successfully. Starting at 10 seconds with default settings still
produces 63.01 m over an 18.43-second output. Enabling stationary initialization
alone is therefore not a demonstrated remedy for this real-data failure.
Visual-constraint loss, initialization quality and calibration/time alignment
remain hypotheses; the root cause is not yet established. No production
threshold or estimator parameter was changed based on these probes.

## Historical Environment Checks - 2026-09-26

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
