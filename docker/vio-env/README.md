# ROS-Free VIO Setup And Usage

## Normal Use: Bundled Runtime

Use `roboego-vio-runtime:dev` for normal inference. It includes the OpenVINS
binary, Python adapters and dependencies. Recipients do NOT need ROS, a GPU,
model weights, OpenVINS source or a compiler. Only Docker, input data and the
small `run-runtime.sh` launcher are needed. Full details: [RUNTIME.md](RUNTIME.md).

### Install Or Build

Complete [host Docker installation](../README.md#host-installation); the NVIDIA
steps are for hand estimation only. Run commands from this repository root.
Choose ONE path:

```bash
# A: Import a delivered runtime image. No compilation is needed.
docker load -i /absolute/path/to/vio-runtime.tar
```

```bash
# B: Build locally using the validated OpenVINS fork/snapshot.
export OPENVINS_ROOT="$HOME/open_vins"
test -f "$OPENVINS_ROOT/ov_offline/frame_export.py"
docker build -t roboego-vio-env:dev docker/vio-env
bash docker/vio-env/build-runtime.sh
```

The dependency build may download packages; the runtime build compiles C++
inside Docker with networking disabled. The cold-start release uses OpenVINS
`fix/vio-cold-start` at `aeaad0ec5a5b37fc475058e693a3b08d02da6168`; official
`master` alone has no compatible offline entry. Preserve tested source revisions.

```bash
docker run --rm --network none roboego-vio-runtime:dev doctor
docker run --rm --network none roboego-vio-runtime:dev test
```

Doctor validates imports/native loading, not trajectory quality. The tests use
synthetic fixtures; a private real-data test can be skipped without its dataset.
Override `VIO_RUNTIME_IMAGE` to use a different tag or immutable image ID.

### Input Layout

All runs require stereo video, camera timestamps, IMU measurements and valid
calibration for the actual device. Depth, hand results and Qwen are not needed.

```text
session/
  videos/left.mp4, right.mp4                       for raw mode
  videos/left_rectified.mp4, right_rectified.mp4   for rectified mode
  timestamps/left_timestamps.txt, right_timestamps.txt
  imu/imu0.csv
  calibrations/camera_imu-camchain-imucam.yaml
  calibrations/camera_imu-imu.yaml
  calibrations/rectify_params.yaml                additionally for rectified
  calibrations/camera-camchain.yaml               use when supplied
  dataset_manifest/dataset_manifest.json          additionally for rectified
```

Placeholder calibration is invalid. Copying calibration from an unrelated
device is not a fix. Video frame counts must match timestamp counts, and IMU
timestamps/units must match the dataset convention. See preparation errors
before changing estimator parameters.

### Run A Raw Session

```bash
cd ~/roboego-hand-vis
export INPUT_DIR=/absolute/path/to/session
export OUTPUT_DIR="$HOME/ego_data/vio_raw_$(date +%Y%m%d_%H%M%S)"
unset INPUT_CONTAINER_PATH
bash docker/vio-env/run-runtime.sh run --input_mode raw
python3 -m json.tool "$OUTPUT_DIR/report.json"
```

### Run A Rectified Session

The reader follows absolute video paths recorded in the dataset manifest. For
the local processed sessions, mount the input at their original `root_dir`:

```bash
cd ~/roboego-hand-vis
export INPUT_DIR="$HOME/ego_data/processed_data/session_20260827_113215_收银_20260903162506"
export OUTPUT_DIR="$HOME/ego_data/vio_cashier_$(date +%Y%m%d_%H%M%S)"
export INPUT_CONTAINER_PATH="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["root_dir"])' \
  "$INPUT_DIR/dataset_manifest/dataset_manifest.json")"

bash docker/vio-env/run-runtime.sh run --input_mode rectified
python3 -m json.tool "$OUTPUT_DIR/report.json"
```

For another dataset, change INPUT_DIR and confirm that manifest video paths are
under its `root_dir`, with matching relative files. The original NAS directory
does not need to exist on the host: INPUT_CONTAINER_PATH is a container mount
destination. It must not collide with `/opt` or `/data/output`. This does not
rewrite the source manifest. Host input is read-only and output is writable.

Append `--start_s 10 --duration_s 8` to select a time window, or `--debug` to
retain intermediates. Use a NEW OUTPUT_DIR for each run. Window replay can use
pre-window IMU history and is not the same as cutting out a strict cold-start
dataset. With no duration argument the whole selected session is replayed.

### Results And Frame Status

Successful default output contains only `trajectory.npz` and `report.json`.
`--debug` additionally retains `debug/` with configuration, CSV, legacy JSONL
and logs. Failure returns nonzero and keeps `report.json` plus `error.log`;
an exported frame-aligned NPZ is also kept even if every pose is unavailable.
Read report status, not just file existence. Data rejected before export may
not produce NPZ. A directory from `prepare` is not an empty inference output.

NPZ `openvins.frame_aligned.v2` keeps every left-video frame. `frame_index`
is `0..N-1`, and each row has timestamps, right-frame index, IMU/camera 4x4
transforms and ONE `status` field. Missing transforms are all NaN, never zero
or copied poses. No redundant `valid`, `processed`, `initialized` arrays are
saved. Schema/quality metadata are scalar fields; report contains aggregate counts.
Native initialization flags remain available in debug `frame_status.csv`.

| Status | Meaning |
| --- | --- |
| `valid` | Finite pose at this frame's timestamp; accuracy NOT certified |
| `initializing` | Processed, but initialization not complete |
| `imu_unavailable` | IMU does not bracket the aligned camera time |
| `state_not_advanced` | Initialized estimator did not advance to this frame |
| `invalid_state` | Non-finite estimator state |
| `not_processed` | Requested frame not reached, e.g. after a runtime failure |
| `not_requested` | Known stereo pair outside the selected replay interval |
| `no_stereo_pair` | Left-video frame absent from stereo-pair input |

Timestamps are seconds; translations are meters. `T_world_camera` maps the left
camera into the local VIO world, not GPS coordinates. Missing stereo pairs have
right index -1 and unknown timestamps NaN. Even windowed replay retains the
full input video's N rows. Legacy JSONL is intentionally valid-pose-only.
Inspect NPZ using the image's NumPy (no host NumPy installation needed):

```bash
bash docker/vio-env/run-runtime.sh python3 -c '
import numpy as np
with np.load("/data/output/trajectory.npz", allow_pickle=False) as data:
    valid = data["status"] == "valid"
    print("schema:", data["schema_version"])
    print("frames:", len(data["frame_index"]), "poses:", int(valid.sum()))
    print("statuses:", np.unique(data["status"], return_counts=True))
    positions = data["T_world_camera"][valid, :3, 3]
'
```

Earlier frame-aligned v1 files had three boolean arrays. New v2 files use
`status == "valid"`; old outputs are not rewritten automatically. Frame-level
availability does not certify drift/accuracy (`quality_status: not_validated`).

### Updating And Troubleshooting

| Symptom / change | Action |
| --- | --- |
| Cannot open manifest video path | Verify INPUT_CONTAINER_PATH and relative video files |
| Output must be empty | Use a fresh OUTPUT_DIR, including after `prepare` or failure |
| Calibration incomplete / frame-timestamp mismatch | Fix source data or supply correct device calibration; do not bypass checks |
| `not_initialized` | Inspect statuses and debug log; do not count NaN rows as poses |
| Complete but trajectory drifts | Initialization/output availability is not quality acceptance; preserve failing data for diagnosis |
| Host code changed but runtime unchanged | Run `build-runtime.sh`; bundled runtime never mounts host source |
| Docker permission denied | Complete host Docker access setup; avoid changing file ownership via sudo launcher runs |

The full-session cashier timing divergence is fixed in the current profile;
short-window cold-start drift and a feature-starved shelf initialization case
remain under investigation. Neither status fields nor online timing estimation
solve all quality problems. No production thresholds are silently relaxed.

The host launcher creates/removes a container for each invocation; no persistent
service or shell entry is needed. To transfer the current image:

```bash
docker save -o /absolute/path/to/vio-runtime.tar roboego-vio-runtime:dev
# On the destination:
docker load -i /absolute/path/to/vio-runtime.tar
```

Deliver `run-runtime.sh` separately, or use the direct `docker run` example in
[RUNTIME.md](RUNTIME.md). Old TAR archives do not update when an image is rebuilt.

## Development Environment Only

The remaining sections describe `roboego-vio-env:dev`, not the normal bundled
runtime. Use it only when compiling/testing a mounted OpenVINS working tree.

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
trajectories for four local sessions. The original fixed-time-offset cashier
baseline diverged badly; current online estimation fixes that full-session case.
`status: complete` and exit zero describe execution, not trajectory quality.
Full acceptance still requires checking output quality, including independently
started short windows. No general small-motion fix is claimed. See `VALIDATION.md`.
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
## Session Timing Update

The local offline adapter now estimates the camera/IMU time offset online by
default. This resolves the reproduced cashier divergence with the supplied
calibration in raw replay; see the newest `VALIDATION.md` section for coverage.
For development comparisons, set `OPENVINS_TIME_OFFSET_MODE=config` before
`run.sh` to honor the original estimator YAML setting. The underlying project
Python files and source calibration remain unchanged. Earlier validation notes
in this document refer to the fixed-offset baseline.

## Cold-Start Release

The full-session pre-release audit covers all four local sessions in raw and
rectified mode, with old-image comparisons and both private cold regressions.
See [VALIDATION.md](VALIDATION.md) for coverage, startup tradeoffs and remaining
accuracy limits. Release image tag: `roboego-vio-runtime:20260930-cold-start`.
Build from committed sources into that tag, verify, then promote `:dev` while
retaining the old image as `roboego-vio-runtime:baseline-20260929`.
The earlier `:cold-start` tag is an uncommitted candidate, not the release.

The `fix/vio-cold-start` OpenVINS branch adds shortage-triggered stereo feature
replenishment and a confidence-checked timing bootstrap before initialization.
No initialization/geometric/innovation acceptance thresholds are relaxed. The
bootstrap uses at most six seconds of images inside the requested window and
only the supplied IMU. This is offline look-ahead, not real-time initialization.
It estimates a local correction from visual/gyro rotation agreement; it does
not hard-code the cashier's offset or rewrite calibration files. An ambiguous,
poor or unexcited fit leaves the configured offset unchanged and records why.
That fallback is not a certificate of correct timing or pose accuracy.

Build a versioned image without replacing `:dev` before verification:

```bash
OPENVINS_ROOT=/home/user/open_vins \
VIO_RUNTIME_IMAGE=roboego-vio-runtime:20260930-cold-start \
  bash docker/vio-env/build-runtime.sh
docker run --rm --network none roboego-vio-runtime:20260930-cold-start doctor
docker run --rm --network none roboego-vio-runtime:20260930-cold-start test
```

Use the same data-only mounts and `run` command as above with this versioned
tag. The new delivery option `--timing-bootstrap auto|off` defaults to `auto`.
`off` retains online offset estimation while disabling the startup bootstrap;
`--time-offset-mode config` always bypasses bootstrap and honors the YAML. Read
`offline.timing_bootstrap` in `report.json` for the actual decision and offsets.
Changing alignment can legitimately change IMU coverage and initialization
frame counts. Successful default output remains exactly `trajectory.npz` and
`report.json`, with all input frames, status-only validity, and NaN invalid
matrices. `--debug` retains intermediate artifacts. `valid` is not an accuracy
guarantee, and remaining upstream timing provenance is unresolved.
