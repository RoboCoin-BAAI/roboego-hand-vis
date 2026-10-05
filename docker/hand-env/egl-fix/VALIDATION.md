# EGL Backend Validation - 2026-10-05

## Decision

Promoted `roboego-hand-env:dev` to
`sha256:ccd9778457140d26c38c780bfe3ad3f407c23fe919fe90fdbd746c240d8d0f8d`.
Rollback image is `roboego-hand-env:pre-egl-c8a998c05a1e` (original c8a998c05a1e).
Only NVIDIA EGL vendor registration/selection changed in this image; no Ray
packages, model weights, Python package versions or algorithm parameters changed.
The full source Dockerfile also contains the fix. Ray render image remains
`sha256:40aca6e01bda26c4f713f6eb8ce47c9e140d6900a3ecdcfdd52454cb9ccf593f`.

## Measurements

Same full session, 2137 frames, 30 FPS, same final Parquet/depth/calibration,
same isolated instrumented EgoHandKit source (base 34b3b1c), same replay code.
All frames retained. MANO fit: 200 steps, batch 64, RMSE threshold 0.03 m.
Renderer still created/deleted per hand-bearing frame; no Actor or reuse.
Both runs explicitly recorded the actual GL renderer. CUDA was used for fitting
in both: "CPU" below refers only to the software OpenGL rendering backend.

Run roots under `/home/user/ego_data/hand_opt_runs/`:

- GPU: `render_egl_gpu_20261004_2355`
- CPU control: `render_egl_cpu_control_20261005_0010`
- Saved original output: `ray_video_20261004_182731/baseline/result/debug/run/mint/input_left_rectified_canonical/final`

Each run contains `replay/render_profile.json`, `profile/render_replay.json`,
resource samples and the complete generated output. GPU ran first, CPU second.

| Wall-time segment | Mesa llvmpipe (s) | NVIDIA RTX 4070 (s) |
| --- | ---: | ---: |
| MANO fit and filter | 28.810 | 26.474 |
| Renderer creation | 16.234 | 12.106 |
| Drawing and readback | 411.812 | 82.494 |
| Renderer deletion | 217.236 | 211.956 |
| Renderer total (includes nested timings) | 665.630 | 327.183 |
| Overlay | 74.251 | 74.644 |
| JPEG encode/write | 8.683 | 8.656 |
| JPEG reload for video | 27.516 | 48.487 |
| Video encode/write | 19.425 | 20.003 |
| Full render pipeline (fit through video output) | 866.348 | 546.870 |
| Replay with extraction/loading, excluding verification | 883.909 | 563.469 |

Observed rendering-pipeline speedup: 1.584x; wall-time reduction: 36.88%.
Drawing/readback alone was approximately 4.99x faster. Do not add parent and
child timers. The full MINT + HaMeR pipeline was NOT rerun; these are not measured
end-to-end inference speedups. This is an EGL environment fix, not a Ray gain.

Resource samples restricted to the render pipeline epoch interval:

| Sample summary | CPU control | GPU candidate |
| --- | ---: | ---: |
| Samples | 866 | 547 |
| Process-tree CPU mean, 100% per core | 200.77% | 234.19% |
| Process-tree RSS peak | 15.408 GiB | 15.246 GiB |
| Device GPU utilization mean | 30.93% | 29.52% |
| Device VRAM peak | 6603 MiB | 6701 MiB |

Device GPU metrics include desktop activity and are not a renderer utilization
measurement. At preflight no compute jobs were listed; desktop graphics remained
active. During the control the sampled compute process was this run. OS page
cache was not controlled; there is only one new run per backend. IO variability
is visible in JPEG reload time. No confidence interval or P95 claim is made.

## Correctness and Visual Review

- Both runs produced 2137 JPEGs and 3095 rendered hands. MANO fit diagnostics
  match the saved original exactly, excluding the source pathname.
- Both MP4s were decoded with ffprobe: 2137 frames, 30 FPS, 1920x1056 (unchanged
  encoder alignment from the 1920x1050 input). No decode errors were reported.
- Strict equality remains false. CPU control differs on 9 frames, max channel
  delta 2; GPU differs on all 1906 hand-bearing frames, max delta 15/255.
- GPU mean absolute channel delta against the original is 0.0035812/255 over
  all pixels. Of 4,308,192,000 pixel locations, 15,453,438 have any channel delta,
  1,565,027 exceed 2, 23,041 exceed 5 and 19 exceed 10.
- Inspected crops from the three frames with highest mean error (1642, 2023,
  69), plus the full frame-1642 comparison. No visible hand displacement or
  silhouette regression was observed in those samples. This is sampled visual
  inspection, not a human review of every frame or a statement about ground-truth
  hand-estimation quality. Differences are consistent with backend rasterization
  differences, but no low-level causal proof of every changed pixel is claimed.
- Evidence is retained in the GPU run's `quality_inspection/comparison.json`
  and PNG comparison sheets. No acceptance threshold was retroactively changed;
  reports keep `comparison_mismatch` and exit code 2. Promotion accepts these
  recorded visualization differences under the user's delegated decision.
- Candidate `doctor` passed for both original MINT and EgoHandKit, including
  CUDA operations, MINT decode/assets, and a nonblank EGL draw.

## Next Experiment

Keep NVIDIA EGL constant. Deletion still costs about 212 seconds across 1906
contexts, making renderer lifetime reuse a concrete next hypothesis. Compare
ordinary single-process reuse and one Ray Actor on the same prepared inputs
before trying concurrency. No Actor changes were made in this validation step.
