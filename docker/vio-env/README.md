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

# Requires the currently missing offline runner target, described below:
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

## Known Source Gap

The current stage expects a custom `roboego_openvins_offline` executable with
`--config`, `--left_video`, `--right_video`, `--stereo_pairs`, `--imu`,
`--output_dir`, `--start_s`, and optional `--duration_s` arguments.

The local `/home/user/open_vins` checkout at
`69488123ed9362dd44b6f28e7f4680abbff1442b` has neither that source nor that CMake
target. The stage's checked-in recommendation points to
`/home/liuyou/Documents/open_vins`, a different machine's path. The container
overrides this path, but cannot supply missing source. The user expects both
machines to use official OpenVINS. It is therefore not established that a
modified repository exists: an adapter may have been omitted, or migration from
the old ROS2 entrypoint may be incomplete. Locate the adapter or explicitly
choose and implement an entrypoint integration before full acceptance.
No replacement runner or algorithm was invented as part of packaging.

## Acceptance Boundaries

Expected checks: doctor exits zero; the existing VIO test passes; the core
library compiles and loads. A successful prepare creates `run/run_manifest.json`,
`run/config` and input reports under OUTPUT_DIR. Full execution must additionally
produce a nonempty, initialized trajectory and its report under `run/trajectory`.
Check timestamps, coverage and failure logs, not just process exit status.
The missing offline target below is not an environment installation failure.

Doctor, core compilation, existing tests and session preparation validate only
the environment and input preparation. Full acceptance still requires the
custom runner, a normal session with an initialized nonempty trajectory, and a
small-motion failure sample for later diagnosis. A process exiting successfully
does not itself prove successful VIO initialization. No small-motion fix is
claimed. The hand module's 32-frame depth sample is not a VIO initialization
acceptance dataset; use original stereo video, timestamps, IMU and calibration.

Start with raw input. Rectified mode uses absolute paths stored in the dataset
manifest; those paths must be valid inside the container. Do not blindly reuse
a manifest containing host-only paths.
