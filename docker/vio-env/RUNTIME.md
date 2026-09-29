# Bundled VIO Runtime

`roboego-vio-runtime:dev` is a separate, self-contained test image. It bundles
the ROS-free executable, shared library, OpenVINS Python adapter, project VIO
orchestration code and dependencies. It does not mount either code repository
or a host build directory at runtime. The development image/launcher remain
unchanged. No ROS, GPU or network is required at runtime.

## Build

From the roboego-hand-vis root, with Docker BuildKit and the local OpenVINS
checkout containing the offline module:

```bash
# Only necessary if the dependency image is not already present:
docker build -t roboego-vio-env:dev docker/vio-env
bash docker/vio-env/build-runtime.sh
```

Overrides: `OPENVINS_ROOT`, `VIO_ENV_IMAGE`, `VIO_RUNTIME_IMAGE`. Compilation
happens inside the image build, with networking disabled, rather than copying
host binaries. The final image retains the dependency image's tools for now;
size minimization is not part of this first test image. Source revisions are
recorded in image labels; a dirty revision denotes local uncommitted changes.

## Test Without Code Mounts

```bash
docker run --rm --network none roboego-vio-runtime:dev doctor
docker run --rm --network none roboego-vio-runtime:dev test

INPUT_DIR=/absolute/path/to/session \
OUTPUT_DIR=/absolute/path/to/new-results \
  bash docker/vio-env/run-runtime.sh run --input_mode raw --duration_s 20
```

The launcher mounts only the session (read-only) and results (writable), records
the image ID in the report, disables networking, and uses the host UID/GID.
Successful runs publish exactly `OUTPUT_DIR/trajectory.npz` and
`OUTPUT_DIR/report.json`, without extra `run/trajectory` directories.
Always use a fresh output directory. Override `VIO_RUNTIME_IMAGE` to select an
immutable tag or image ID. Source changes do not affect a built runtime image;
rebuild it to incorporate changes.

Add `--debug` to retain intermediate inputs, configuration, CSV/JSONL trajectories
and logs under `OUTPUT_DIR/debug/`. Default runs use a temporary `.work` directory
on the output filesystem and remove it after publishing results. Debug manifests
record the original temporary paths and are diagnostic snapshots, not portable
replay manifests.

On failure, `report.json` and `error.log` are retained even without `--debug`,
and the exit code remains nonzero. If frame-aligned export succeeded,
`trajectory.npz` is also retained with explicit invalid/unprocessed rows, even
when no frame initialized. Its presence does not indicate success. Invalid
inputs that prevent export may leave only report/log. Existing nonempty output
directories are rejected without modification.
An interrupted/killed container may leave `.work`; use a new directory rather
than treating such an incomplete run as successful.

`report.json` includes execution status, input arguments, code revisions, image
ID (when launched by the host script), pose count and trajectory statistics.
`quality_status: not_validated` explicitly means execution success is not an
accuracy certification. The delivery report schema is `roboego.vio.delivery.v2`.
`frame_alignment` contains input frame count, valid/invalid pose counts and
status counts. NPZ schema `openvins.frame_aligned.v2` provides one row per left
video frame; the existing matrix and timestamp arrays now include invalid rows.
Always filter poses by `data['status'] == 'valid'`. Invalid matrices are NaN.
NPZ does not store redundant `valid`, `processed`, or `initialized` arrays.
Detailed initialization/state-advance flags remain in debug `frame_status.csv`.
Old v1 files are not automatically rewritten; rerun with the current image.
`status` distinguishes `initializing`, `imu_unavailable`, `state_not_advanced`,
`invalid_state`, `not_processed`, `not_requested`, `no_stereo_pair`, and `valid`.
The last means a finite advanced estimate, not validated positioning accuracy.
Windowed runs retain the full video's indices/row count and mark excluded known
pairs `not_requested`. Frames missing stereo pairs have right index -1 and
unknown timestamps NaN. Legacy debug JSONL remains valid-pose-only.

```python
import numpy as np
with np.load('trajectory.npz', allow_pickle=False) as data:
    print('Video frames:', len(data['frame_index']))
    valid = data['status'] == 'valid'
    print('Available poses:', int(valid.sum()))
    print(np.unique(data['status'], return_counts=True))
    poses = data['T_world_camera'][valid]
```
The separate `prepare` action intentionally retains preparation intermediates;
the compact contract applies to `run`.

## Rectified Sessions

Use `--input_mode rectified` when the session contains rectified videos,
rectification calibration and its dataset manifest. The unchanged project
reader uses absolute video paths in that manifest. If these point to a different
machine, mount the session at its original directory via `INPUT_CONTAINER_PATH`:

```bash
INPUT_DIR=/actual/local/session \
INPUT_CONTAINER_PATH=/original/session/path/from/dataset_manifest \
OUTPUT_DIR=/actual/local/new-results \
  bash docker/vio-env/run-runtime.sh run --input_mode rectified --duration_s 20
```

This only changes the mount destination; it does not rewrite source metadata.
Referenced videos must be inside that session at the corresponding relative
paths. The image cannot supply missing input files or repair inconsistent
calibration. Without an override, the input is mounted at `/data/input`.

For the local 714-frame shelf session, the tested 20-second command is:

```bash
INPUT_DIR=/home/user/ego_data/processed_data/session_20260826_100651_整理货架_20260828102934 \
INPUT_CONTAINER_PATH=/mnt/nas/synnas/ego/baai_ego_task/cp_0001/session_20260826_100651/divide/session_20260826_100651_整理货架_20260828102934 \
OUTPUT_DIR="$HOME/ego_data/vio_runtime_manual_$(date +%Y%m%d_%H%M%S)" \
  bash docker/vio-env/run-runtime.sh run --input_mode rectified --duration_s 20
```

## Original Python Entry

The module entry is retained, with image-local OpenVINS paths configured in the
bundled YAML only (the host repository configuration is not edited):

```bash
docker run --rm --network none \
  --mount type=bind,src=/actual/session,dst=/data/input,readonly \
  --mount type=bind,src=/actual/new-results,dst=/data/output \
  roboego-vio-runtime:dev python3 -m roboego_hand_vis.vio.run \
  --session /data/input --output_dir /data/output --input_mode raw
```

## Transfer And Acceptance

The image entrypoint recognizes this Python module command and routes it through
the same output wrapper, including `--debug`. Bypassing the entrypoint to invoke
the internal Python module directly retains the original low-level output layout.

```bash
docker save -o /path/to/vio-runtime.tar roboego-vio-runtime:dev
# On the destination machine:
docker load -i /path/to/vio-runtime.tar
```

No OpenVINS repository or rebuild is needed on the destination. Docker and the
input data are still required. The small `run-runtime.sh` launcher can be copied
separately, or use `docker run` directly. Preserve corresponding source revisions
and license obligations when distributing an image.

Successful execution is not proof of accurate VIO; inspect the trajectory before
accepting results. See `VALIDATION.md` for measured test coverage.

## Camera-IMU Timing

Offline replay now defaults to `--time-offset-mode estimate`: OpenVINS estimates
the camera/IMU time offset online, starting from the supplied calibration. This
addresses the reproduced cashier divergence caused by treating an inappropriate
session offset as fixed. It does not hard-code the correction measured on that
recording, modify source calibration files, or change the project's Python code.

For a controlled comparison, append `--time-offset-mode config` to honor the
estimator YAML's `calib_cam_timeoffset` setting. The wrapper passes this mode to
the OpenVINS launcher through its environment, not through the unchanged project
CLI. Reports include the requested mode, whether estimation was enabled, and
initial/final offsets. Online estimation still requires informative observations;
it is not a guarantee against every synchronization or low-motion failure.
