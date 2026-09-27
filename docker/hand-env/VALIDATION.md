# Local Validation - 2026-09-26

## Documented Profile Retest - 2026-09-27

The full `documented-pipeline.sh` profile passed offline with the existing image
and RTX 4070. No algorithm settings were reduced; HaMeR used its default batch
size 48. Input was the existing 100-frame sample (30 FPS, 1920x1080) from
`session_20260826_121434_商品打包_20260831144710`, with matching depth and intrinsics.

Host output:
`/home/user/ego_data/hand_test100_rK14fe/documented100_retest_20260927_142448`.
Commands and exit status are retained in `pipeline.log`; exit status was zero.

- MINT: window 32, four windows, 100 frames, forward time 17.08 seconds
  (not end-to-end time); checkpoint missing/unexpected tensor counts both zero.
- Canonical: 100 left, 46 right observations. Depth and MINT 3D consistency
  gates each kept 146/146 hands.
- Final Parquet independently read: 100 sequential frame rows; left HaMeR 100,
  right HaMeR 46 and absent 54. No MINT fallback in final source fields.
- Final render independently decoded: all 100 frames nonblank (minimum pixel
  standard deviation 53.31). Encoder resized height 1080 to 1088 for alignment.
- No OOM, no image rebuild, no depth archiving, no overwrite of earlier results.
  This validates the documented short-sample chain, not long-video accuracy.

The remainder records the earlier, reduced 32-frame smoke test.

Image: `roboego-hand-env:dev`

Docker image ID:
`sha256:c8a998c05a1ed0aa10513a6a4e3d5d7e649bbab3234390d3350176ca2ab65ded`

Docker-reported image size: 14,583,445,066 bytes. No image archive has been
exported. All runtime commands used `run.sh` with `--network none` and read-only
repository/model mounts; builds used network access. No algorithm code changed.

## Environment

- Host: RTX 4070, 12 GB VRAM, NVIDIA driver 550.163.01, Docker 29.0.4.
- MINT: Python 3.10, torch 2.8.0+cu128. GPU math, CUDA NMS, decord decoding,
  model imports and both MANO assets passed.
- EgoHandKit: Python 3.10, torch 2.7.1+cu126. GPU math, CUDA NMS, imports
  and nonblank 64x64 EGL rendering passed.
- Asset staging tests: 2 passed. Shell syntax checks passed.
- MINT wheel-tag exception is documented in README; dependency conflicts are
  not suppressed. Package resolutions are recorded in each env's installed.txt.

## End-to-End Sample

Used the README's three commands with 32 rectified RGB frames, 30 FPS and
matching uint16 millimeter depth. Output root:
`tmp/container_validation/container_output`.

- MINT: checkpoint missing/unexpected tensors both zero; 32 predicted frames.
  Model forward time 4.71 seconds, excluding initialization and other stages.
- Canonical export: 32 frames, 32 left and 32 right observations.
- HaMeR: depth gate kept 64/64; 3D consistency kept 43/64 (left 32, right 11).
- Parquet: 32 sequential frame rows. Left source: HaMeR 32. Right source:
  HaMeR 11, MINT fallback 21. The 64 rendered hands are not 64 accepted HaMeR
  reconstructions.
- Final video: decoded all 32 frames; 1920x1056, 30 FPS. ImageIO resized the
  original 1920x1050 for codec block alignment. Every decoded frame was nonblank
  (minimum pixel standard deviation 77.70); the first frame was visually checked
  for rendered meshes.

Final artifacts are under
`egohandkit/sample32_raw_left_rectified_canonical/final/`:
`hand_tracking.parquet`, `render.mp4`, `summary.json`, `parquet_mesh_fit.json`.

The initial final-render attempt failed because camera calibration metadata was
missing. Added sample-only compatibility metadata with the actual existing
FoundationStereo intrinsics under the filename required by EgoHandKit; rerun
passed. No calibration was guessed and no algorithm implementation was changed.

## Code And Remaining Limits

- roboego-hand-vis baseline: `b6328a9`; this change adds Docker scripts/tests/docs.
- MINT baseline: `1bd21e4`, with pre-existing local modifications to `.gitignore`,
  `docs/inference.md`, inference engine/runner/video and visualization renderer;
  converter script and tests are untracked. Preserve that working tree, not
  only the commit, for reproducibility.
- EgoHandKit baseline: `6a83293`, clean working tree after verification.
- Existing wrapper argument mismatch remains; acceptance invokes repos directly.
- Harmless runtime warnings remain for Ultralytics cache fallback and disabled
  PyTorch kernel caching when the host UID differs from image UID 1000.
- This validates container operation on a short sample, not pose/depth accuracy,
  long-video memory usage, other backends, or deployment on the target 24 GB GPU.
- Review model redistribution licenses before delivery. Transitive dependencies
  are recorded but not yet supplied as fully locked rebuild requirements.
