# Local Validation - 2026-09-28

## Full-Session Release Gate - 2026-09-30

OpenVINS repair commit: `aeaad0ec5a5b37fc475058e693a3b08d02da6168`.
All four extracted sessions under `/home/user/ego_data/processed_data` were
replayed in raw and rectified modes against the pinned previous runtime. The
sibling ZIP is a duplicate, not a fifth session. No algorithm changes were made
after the candidate tested below; final cleanup adds tests/audit/docs only.

Eight corrected candidate runs passed frame coverage, v2 status/NaN/rigid-pose
contract checks and gross-trajectory symptom checks. All have continuous valid
tails to the last frame; maximum translation increment is below 5.91 cm and
maximum finite-difference speed is below 1.771 m/s. No prolonged zero-exported-
visual-count stretch occurred (maximum 11 frames; a proxy, not an update log).

| Session suffix | Raw available/path m | Rectified available/path m |
| --- | --- | --- |
| Shelf 20260827190358 | 2071 / 17.408840 | 2075 / 18.683074 |
| Shelf 20260828102934 | 652 / 3.786154 | 649 / 4.309932 |
| Packing 20260831144710 | 2737 / 23.088412 | 2737 / 25.665842 |
| Cashier 20260903162506 | 1272 / 7.422920 | 1287 / 8.594048 |

Explicit tradeoff: cashier raw initialization is 26 frames (0.867 s) later than
baseline; rectified is three frames later. All video rows remain present.
Common-frame comparisons avoid counting this startup difference as a quality
improvement. Neither the old nor new trajectory is ground truth.

Rectified manifests retain original-machine `/mnt/nas/...` paths. The initial
attempt failed before estimation; reruns use explicit validated read-only mount
aliases, not rewritten source data. Exact commands and reports are in
`/home/user/ego_data/vio_release_audit_20260930/{baseline_mounted,candidate}`.
The OpenVINS `ov_offline/diagnostics/RELEASE_AUDIT_20260930.md` contains the full
comparison and reproducible audit command. Temporary probes are archived under
the audit root, not shipped as production code.

Pre-build verification: **39 passed, 1 skipped**, including both private cold
cases and delivery/audit tests; native timing unit tests also pass. The remaining
skip is the older optional cashier-manifest test. Build a clean-commit image as
`roboego-vio-runtime:20260930-cold-start` and repeat doctor, bundled tests and the
full audit before promoting `:dev`. Final image/commit IDs and results are
recorded alongside the audit artifacts after building. No remote push.

## Cold-Start Candidate - 2026-09-30

Candidate: `roboego-vio-runtime:cold-start`, image
`sha256:636fc8e9b200267534d11c219708bc38069c35d029255ae4cb40ba0aad102482`.
`:dev` remains pinned to `4474a8157e80`; the existing exported TAR is unchanged.
Sources are uncommitted on `fix/vio-cold-start` in both repositories, based on
OpenVINS `7e036d9` and project `1d47c92`. Project edits are only under `docker/`.

- Build includes native timing unit tests; all passed. Bundled doctor passed.
- Bundled tests with the two private cold cases enabled: **35 passed, 1 skipped**.
  Only private data was mounted, not source or compiled artifacts. The skip is
  the older optional cashier manifest test, not either cold-window regression.
- Shelf strict cold: 0 -> 60 available poses out of 120 requested frames.
- Cashier strict cold: automatic initial offset -7.776 ms, 176 available poses,
  0.829028 m path. In the boundary-controlled comparison, exactly the same 179
  frames have path 18.546230 -> 0.847708 m (full-run reference: 0.824331 m).
- Negative control in the same new image, bootstrap disabled: cashier test
  intentionally fails again at 18.546230 m. Online estimation remains enabled.
- Two cold windows, corresponding historical-IMU windows and four full
  rectified recordings completed regression. Bootstrap is a bounded offline
  look-ahead with rejection checks, not a hard-coded lag or relaxed VIO gate.
- Public raw `run` smoke checks: cashier full returns exactly NPZ + report;
  shelf 19..23 s with `--debug` adds only the requested debug directory. Every
  frame remains present, status is the only validity array, invalid matrices
  are entirely NaN, and available matrices are finite. The latter window retains
  historical IMU and is not described as a strict cold start.

Detailed diagnosis, commands, bounds and limitations:
`/home/user/open_vins/ov_offline/diagnostics/COLD_START_FIX_20260930.md`.
Artifacts: `/home/user/ego_data/vio_cashier_cold_20260930/`, including
`bootstrap_final`, `bootstrap_other_full`, `runtime_cashier_raw_full` and
`runtime_shelf_raw_window`. Run with `--timing-bootstrap off` to disable only
startup alignment, or `--time-offset-mode config` to honor the YAML timing
policy. Timing decisions are in `report.json` under `offline.timing_bootstrap`.

No ground truth is available. These are reproduced-failure regressions, not
trajectory accuracy acceptance; upstream recovery/cutting timing provenance
remains unresolved. Rejected bootstrap retains the configured lag, not a
guarantee that the lag or subsequent estimates are correct.

## Status-Only Delivery - 2026-09-29

Current image: `sha256:f98e46bf20659cc23874ea16c2d936bf554addcf0b0699424b45f625d0afa239`.
NPZ schema `openvins.frame_aligned.v2` removes the redundant `valid`, `processed`
and `initialized` arrays. Use `status == "valid"`; other values explain missing
poses, whose matrices remain NaN. Native diagnostic flags remain in debug CSV.
The earlier v1 validation below is historical; old result files/TARs are not
rewritten automatically.

Rebuilt-image doctor and 29 tests passed; one private-data test skipped.
Real-data acceptance under `/home/user/ego_data/vio_status_only_20260929/`:

- Default compact cashier run: 1360 rows, 1290 valid, 68 initializing, 2
  IMU-unavailable. Only `trajectory.npz` and `report.json` published.
- Frozen 120-frame shelf cold replay: exit 3 / not_initialized; 120 rows,
  119 initializing, 1 IMU-unavailable, all pose matrices NaN.
- Both NPZ files have only metadata, original frame/time/pose fields and status;
  none contain the three removed arrays.

Documentation bash syntax/local-link checks passed (32 shell blocks). Hand
image was not rebuilt: both hand doctors, mount regression and five hand output
unit tests passed. No full hand inference was rerun for this output-only change.

## Frame-Aligned Delivery - 2026-09-29

Image: `sha256:e46629a28aee8bee41ad8f1ef544ca2df09cdf37013c208ef04495b9e77b8395`.
Output report schema is `roboego.vio.delivery.v2`; NPZ schema is
`openvins.frame_aligned.v1`. Missing matrices are NaN and must be filtered by
`valid`. Legacy JSONL remains valid-pose-only. No estimator tuning was changed.

- Image tests: 29 passed, 1 private-data test skipped. Covers explicit IMU
  boundary records, initialization, non-advanced/invalid states, interrupted
  tails, absent stereo pairs, replay windows, and default failure publication.
- Full rectified cashier run: 1360 NPZ rows, 1290 available poses, 68 initializing
  frames and 2 IMU-unavailable frames. All 70 missing matrices are entirely NaN.
  Available frame IDs match the prior run exactly; maximum absolute matrix-entry
  difference was 1.85e-6 (numerical replay variation, not pose filling).
- 19..23 s shelf replay using production pre-roll: 714 rows for the full input
  video, 594 not-requested frames, 46 initializing frames and 74 available poses.
- Frozen strict cold-start shelf case was also materialized as lossless
  120-frame stereo videos with remapped frame indices and original timestamps.
  Native replay returns 3 / not_initialized, but NPZ retains all 120 rows:
  119 initializing, 1 IMU-unavailable, zero valid poses, all matrices NaN.

Artifacts: `/home/user/ego_data/vio_frame_aligned_20260929/` (`cashier`, `shelf19`,
`cold_fixture/result/trajectory`). The lossless fixture script is retained there.
This is an output-contract fix, not a solution for feature-starved initialization
or the separately observed short-window cashier drift. `valid` means an
available finite advanced state; positioning accuracy remains unvalidated.

## Cashier Timing Fix - 2026-09-29

Image: `sha256:65c4724c2b1cb5e477a3e64bcc0685b7f0a014727b7b61a74e50c8507f38c512`.
Offline estimation now enables the official camera/IMU time-offset state rather
than freezing the supplied -47.7755 ms offset. Source calibration files and
project VIO Python code were not changed. No output poses were clipped, frozen
or discarded to make a trajectory appear bounded.

Evidence: the first 10 seconds of cashier data repeatedly fail the private
regression's generous 10 m work-area bound (about 35.3 m displacement). Visual
rotation versus gyro alignment favors approximately -10 ms; the same check on
a shelf session favors approximately -40 ms, near its calibration value.
Changing only the cashier test config to -10 ms reduced full raw path length to
7.61 m. Enabling the official online state instead, without hard-coding a value,
converged to -6.8 ms and produced 8.78 m cumulative raw path length, with a maximum
0.80 m distance from the initial position and 1298 continuous pose samples.

Full rectified replay in the bundled image was compared on the same recording:

| Mode | Poses | Duration | Path length | Max distance from initial position |
| --- | ---: | ---: | ---: | ---: |
| Fixed YAML offset (old image) | 1290 | 42.97 s | 3323.93 m | 3228.84 m |
| Online offset estimate | 1290 | 42.97 s | 8.92 m | 0.56 m |

The rectified offset converged to -7.23 ms; maximum output timestamp gap remained
0.03339 s. This identifies an inappropriate fixed effective timing offset for
this recording, not its exact upstream origin (capture/cutting clock handling
versus calibration session). Online estimation fixes the reproduced divergence;
there is no ground truth pose-accuracy certification.

The other three full raw sessions retain the same pose counts (2071, 652, 2737),
with finite, continuous outputs and no comparable divergence. Their final time
offsets are -39.33, -38.98 and -43.78 ms. The private regression fails in `config`
mode and passes in `estimate` mode. Image suite: 20 passed, 1 private-fixture skip;
an additional Docker mode-forwarding test passes in the source suite (6 Docker
wrapper tests). Development suite: 15 passed, 1 skip; private replay: 1 passed.

All data/probes: `/home/user/ego_data/vio_cashier_fix_20260929/`.
User-facing fixed rectified results: `rectified_after/trajectory.npz` and
`rectified_after/report.json`; intermediate evidence is under its `debug/`.
Use `--time-offset-mode config` for an explicit fixed-configuration comparison.
The normal `run` command now defaults to `estimate`. Quality remains
`not_validated`; arbitrary sensor miscalibration or tracking failures are not
claimed solved by this change.

## Compact Outputs - 2026-09-29

Runtime image rebuilt as
`sha256:09b26b0de6cde2188fb69048e467307de271bd8fde4e09a1ead4470d4875df12`.
The delivery wrapper is entirely under `docker/vio-env`; project VIO source
and OpenVINS estimator behavior are unchanged.

- Image test suite: 18 passed, including output layout, debug retention,
  failure logging, missing-result handling and refusal to overwrite.
- Real 20-second raw replay: 539 poses; only `trajectory.npz` and `report.json`
  at the output root. Another run with `--debug` additionally retains `debug/`.
- A 0.1-second replay fails initialization, exits nonzero and publishes only
  `report.json` and `error.log`; no final trajectory is published.
- Re-running into the successful output directory is rejected.
- Results: `/home/user/ego_data/vio_compact_20260929/{normal,debug,failure}`.
- Reports distinguish execution completion from `quality_status: not_validated`.

Earlier runtime validation below describes the previous image/output layout.

## Bundled Runtime - 2026-09-29

Built `roboego-vio-runtime:dev` from the existing dependency image, compiling
OpenVINS from source in a network-disabled build step. The runtime bundles code,
native executable, shared library and Python compatibility launcher. No host
source or build directory is mounted for these checks:

- Image ID: `sha256:d30ab73d37239a865285661765e7ca72117341c1dccae81ebf510a977973c037`.
- OpenVINS revision: `e34182f543628567d3a2403b79ba717207138c64`.
- Project revision: `e36a1d769f89ff1c1458e78557bb2e6449254edb` plus Docker-only work.
- Doctor passed; bundled test suite: 13 passed, with networking disabled.
- 20-second raw replay of shelf session ending `20260828102934`: 539 poses,
  17.93 seconds of output, 3.46 m path length.
- Same session, rectified mode: 536 poses, 17.83 seconds of output, 3.82 m path
  length. The manifest's original NAS directory was used as the input mount
  destination; no input metadata was rewritten. Initialization logged covariance
  rank-deficiency warnings before succeeding; this is not an accuracy guarantee.
- Results: `/home/user/ego_data/vio_runtime_test_20260929/{raw,rectified}`.

The image-local recommended config uses bundled OpenVINS paths; the host project
config/code are unchanged. This does not fix the known cashier divergence.
Development image and source-mounted launcher are retained independently.

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
