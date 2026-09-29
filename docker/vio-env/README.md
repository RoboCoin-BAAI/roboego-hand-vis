# ROS-Free VIO Environment

Environment-only image for the current `roboego_hand_vis/vio` entrypoint.
No OpenVINS source, host binaries, session data or pretrained models are copied
into the image. No ROS, GPU or NVIDIA runtime is required for this scope.
This is not the legacy ROS2 replay workflow under `linshi`.

## Build And Mounts

Prerequisites: Linux x86_64, Docker with BuildKit, this repository and the
OpenVINS source working tree (default `~/open_vins`). No host Conda or ROS
installation is needed. Execute commands from the roboego-hand-vis root.

```bash
docker build -t roboego-vio-env:dev docker/vio-env
export INPUT_DIR=/absolute/path/to/raw-session
export OUTPUT_DIR=/absolute/path/to/vio-results
export BUILD_DIR=/absolute/path/to/vio-build
bash docker/vio-env/run.sh doctor
bash docker/vio-env/run.sh build
bash docker/vio-env/run.sh test
```

Expected raw session layout:

```text
videos/left.mp4
videos/right.mp4
timestamps/left_timestamps.txt
timestamps/right_timestamps.txt
imu/imu0.csv
calibrations/camera_imu-camchain-imucam.yaml
calibrations/camera_imu-imu.yaml
```

Calibration files must contain real YAML mappings for the recording device,
not placeholders or calibration borrowed from another device. With valid input:

```bash
bash docker/vio-env/run.sh prepare
```

The local `sample` currently has placeholder camera-IMU calibration files, so
its `prepare` command is expected to fail. Select a session with real calibration
for a successful preparation run; see `VALIDATION.md`.

Building downloads Ubuntu packages. Runtime and compilation through `run.sh`
disable networking. Package versions are recorded in
`/usr/local/share/vio-packages.txt`; apt dependencies are not snapshot-locked,
so preserve the built image for exact environment reuse.

| Container path | Contents | Access |
| --- | --- | --- |
| `/home/user/roboego-hand-vis` | This working tree | Read-only |
| `/home/user/open_vins` | OpenVINS working tree | Read-only |
| `/home/user/data/input` | Raw session | Read-only |
| `/home/user/data/output` | Reports, trajectory, logs | Writable |
| `/home/user/build` | Container-compiled binaries and ccache | Writable |

Host overrides: `PROJECT_ROOT`, `OPENVINS_ROOT` (default `$HOME/open_vins`),
`INPUT_DIR` (required), `OUTPUT_DIR` (default `tmp/vio_container/output`),
`BUILD_DIR` (default `tmp/vio_container/build`), `VIO_ENV_IMAGE`, `BUILD_JOBS`
(default 2). Paths containing commas are rejected. Runs use the host UID/GID.
Use a new `BUILD_DIR` when changing compiler/environment or OpenVINS repository.

## Compile Before Running

```bash
# Validate the core C++ dependencies with the available upstream source:
bash docker/vio-env/run.sh build ov_msckf_lib
bash docker/vio-env/run.sh python3 -c \
  'import ctypes; ctypes.CDLL("/home/user/build/openvins/libov_msckf_lib.so"); print("Core library loaded")'

# Compile the official library and this repository's offline adapter:
bash docker/vio-env/run.sh build
bash docker/vio-env/run.sh run --duration_s 20
```

Compilation is out-of-source, inside the offline container. It never reuses
host-built binaries or writes to the mounted source tree. After C++ changes,
run `build` again; Python changes take effect on the next invocation. The default
build target is `roboego_openvins_offline`. Do not use a stale executable after
an unsuccessful rebuild. No initialization or tracking parameters are changed.

Each invocation records command arguments, UTC time, source commits/dirty status,
stdout/stderr and exit status under `OUTPUT_DIR/logs`. The launcher records the
resolved image ID. These logs help reproduce failures but do not constitute a
snapshot of uncommitted source changes. Archive the exact working trees with a
failure report. Generated config/manifest and outputs are under `OUTPUT_DIR/run`.

## Offline Adapter

The current stage expects a custom `roboego_openvins_offline` executable with
`--config`, `--left_video`, `--right_video`, `--stereo_pairs`, `--imu`,
`--output_dir`, `--start_s`, and optional `--duration_s` arguments.

The upstream OpenVINS checkout does not supply this executable. Use the local
OpenVINS working tree/fork containing `ov_offline`. Native replay, Python export
and their tests now live there, linked against official `ov_msckf_lib`.
The target builds directly from `ov_msckf` with `ENABLE_ROS=OFF`; there is no
second adapter build in this repository. OpenVINS can run independently using
`python3 -m ov_offline.run`; see its `ov_offline/README.md`.
It replays stereo frames and IMU without ROS, waits synchronously for initialization,
and exports initialized poses. The estimator itself remains official OpenVINS.
The integration is based on OpenVINS commit
`69488123ed9362dd44b6f28e7f4680abbff1442b` plus the local offline-module changes.

The public orchestration CLI is unchanged:

```bash
python -m roboego_hand_vis.vio.run \
  --session /absolute/path/to/session --input_mode rectified
```

The original project code and tests are unchanged. Its `offline_entry.py` calls
the compatible `roboego_openvins_offline` command generated by the OpenVINS build.
That launcher owns execution/export via `ov_offline` and delegates estimation to
the sibling `openvins_offline_native` executable. Its `--output_dir` is the final
trajectory directory, matching the original project contract.
The checked-in project config retains its original machine-specific paths;
outside Docker, override `--openvins_root` and `--openvins_binary` as needed.
The Docker entry sets these paths automatically. Rectified mode requires valid
rectification metadata and paths accessible in the runtime environment.
The Docker equivalent is `bash docker/vio-env/run.sh run --input_mode rectified`
with INPUT_DIR and OUTPUT_DIR set on the host.

The launcher executes the mounted `entry.sh`, so these integration changes work
with the existing environment image after `build`; no image rebuild is needed.
Use a fresh OUTPUT_DIR for each run. Existing trajectories are not overwritten.
Outputs include `trajectory.jsonl`, `trajectory.npz`, `states.csv`, per-frame
`frame_status.csv`, `run_report.json` and `run/openvins.log`. Covariance is not
exported. No initialized trajectory means a nonzero exit, not silent success.

## Acceptance Boundaries

Expected checks: doctor exits zero; the existing VIO test passes; the core
library compiles and loads. A successful prepare creates `run/run_manifest.json`,
`run/config` and input reports under OUTPUT_DIR. Full execution must additionally
produce a nonempty, initialized trajectory and its report under `run/trajectory`.
Check timestamps, coverage and failure logs, not just process exit status.

Doctor, core compilation, tests and session preparation validate only
the environment and input preparation. Real-data replay now emits initialized
trajectories for four local sessions, but the cashier session diverges badly.
`status: complete` and exit zero describe execution, not trajectory quality.
Full acceptance still requires resolving this divergence and checking output
quality. No small-motion fix is claimed. See `VALIDATION.md` for measurements.
The hand module's 32-frame depth sample is not a VIO initialization
acceptance dataset; use original stereo video, timestamps, IMU and calibration.

Synthetic stationary replay reproduces the default initializer waiting for an
acceleration jerk. In a controlled test, setting OpenVINS `try_zupt: true` and
`zupt_only_at_beginning: true` enables stationary initialization. This is not a
validated fix for the reported real-world small-motion failure; production
defaults are unchanged. The four local processed sessions originally contained
placeholder camera-IMU files. At the user's direction, these were backed up and
replaced with the calibration from `session_20260804_223748`; all four now pass
input preparation. See `VALIDATION.md` for replay results and remaining limits.

Start with raw input. Rectified mode uses absolute paths stored in the dataset
manifest; those paths must be valid inside the container. Do not blindly reuse
a manifest containing host-only paths.
