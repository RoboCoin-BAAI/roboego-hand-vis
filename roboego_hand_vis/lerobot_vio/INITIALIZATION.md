# Stereo initialization policy

L1/run.py, L3/run.py and scripts/L3/rectify_vio.py support these policies:

| --init-mode | Behavior |
| --- | --- |
| static-zupt | Static initialization with ZUPT; dynamic initialization disabled |
| dynamic | Dynamic initialization only; static initialization and ZUPT disabled |
| static-then-dynamic | Static + ZUPT first, retry dynamic on failure (default) |
| dynamic-then-static | Dynamic first, retry static + ZUPT on failure |

Example:

```sh
python -m roboego_hand_vis.lerobot_vio.L3.run --source SESSION --init-mode static-then-dynamic
```

Combination policies rerun the same episode/time range with a fresh estimator, rather
than switching an already initialized estimator in place. A successful first attempt
stops the policy. Backend failure, unsuccessful initialization, no valid poses, or
failed trajectory quality triggers the second attempt. Invalid inputs and publication
errors remain errors without retry. If both fail, the episode remains failed.
`--prepare-only` prepares only the first mode and does not retry.

Static + ZUPT uses init_dyn_use=false and try_zupt=true. Dynamic uses
init_dyn_use=true, init_imu_thresh=0 (disables the backend static branch), and
try_zupt=false; the visual motion gate and other dynamic thresholds remain intact.
ZUPT is gated by the backend's stationarity checks and is not guaranteed to apply.

Dynamic reports use vio/ (vio_raw/ for raw inputs); static + ZUPT reports use
vio_zupt/ (vio_raw_zupt/ for raw inputs). fallback_selection.json in vio/ or vio_raw/
records the current policy, attempts, and selected mode (null when no attempt passes).
Each attempt saves initialization_settings.json under its .work directory and includes
settings in its report.json. Rectified runs retain the existing derived-feature
publication behavior; failed-quality artifacts retain their failure status.

Only stereo inputs are supported. The standalone monocular script has been removed.
Legacy --init-mode auto/static and --zupt/--zupt-fallback remain accepted for older
commands; auto allows both backend initializers and is not dynamic-only. Do not
combine legacy ZUPT flags with the four policy modes above.


L1/L3 pipelines explicitly select `--init-mode static-then-dynamic`.
Before writing runtime imu_clean.csv, VIO validates all input measurements,
stably sorts by timestamp and retains the first sample at each duplicate time.
Original Parquet files remain unchanged. IMU diagnostics include normalization
sample counts, removed duplicates and original backward steps.

Video timestamps need not be strictly increasing. Runtime stereo pairing keeps original frame order and indices, skipping repeated/backward timestamps until time advances. Source videos/Parquet remain unchanged; skipped indices are recorded in .work/<run>/camera_timestamp_handling.json. Skipped frames have no estimated pose and still count toward the trajectory valid-pose ratio.


Before rectified stereo inference, L1/L3 share a first-frame quality gate.
SIFT ratio-test matches must be mutual, with >=30 matches spread over >=4 cells
of a 4x3 grid. Median absolute vertical error must be <=1.5 pixels and P95 <=3
pixels at the rectified image resolution. These are initial heuristic thresholds,
not calibrated guarantees. No vertical-error filtering is applied before measuring.
Reports are cached at annotation/stereo_quality/<chunk>_<episode>.json and reused
when video paths, sizes, mtimes and policy match. Warning/inconclusive results
block depth/VIO with a nonzero exit, leaving rectification and semantic annotation
unaffected. Only frame 0 is assessed; later frame quality is not certified.

First-frame quality thresholds now default to median <=20 px, P95 <=40 px. Thresholds are fixed in code and cannot be overridden by environment variables. Policy changes invalidate the cached gate result. These relaxed thresholds are not accuracy guarantees.
