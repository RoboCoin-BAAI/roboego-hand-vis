# Offline Hand Runtime

Scope: MINT headless inference, canonical observation export, HaMeR, depth
gating, Parquet export and EGL rendering. No algorithm repositories are copied
into the image. Source repositories are mounted at runtime. Qwen,
FoundationStereo, Omega, ViTPose and other HMR backends are not part of v1.

## Layout

```text
/home/user/envs/mint          Python 3.10, PyTorch 2.8.0 + CUDA 12.8 wheels
/home/user/envs/egohandkit    Python 3.10, PyTorch 2.7.1 + CUDA 12.6 wheels
/home/user/model             Model assets and SHA-256 manifest
/home/user/wuji-ego-mint      Read-only source mount
/home/user/EgoHandKit        Read-only source mount
/home/user/roboego-hand-vis   Read-only source mount
/home/user/data/input        Read-only data mount
/home/user/data/output       Writable outputs
```

The host provides Docker and NVIDIA Container Toolkit with a compatible GPU
driver. CUDA user-space runtimes come from the pinned PyTorch wheels; no host
Conda environment or host CUDA toolkit is mounted. The deployment target is an
NVIDIA GPU with at least 24 GB VRAM. Test the actual target driver/GPU before
declaring MINT acceptance; local 12 GB hardware is not equivalent.

## Build

Prerequisites: Linux x86_64, Docker with BuildKit and volume-subpath support
(tested with Docker 29.0.4), NVIDIA Container Toolkit, and a compatible NVIDIA
driver. `docker info` and `nvidia-smi` must work. The host's Python/Conda packages
are not used for inference. Keep the MINT and EgoHandKit working trees, including
local changes, available at the paths below or override their roots.

Required source assets (not supplied by Git):

```text
~/wuji-ego-mint/checkpoints/model.safetensors
~/wuji-ego-mint/assets/mano/mano_left/MANO_LEFT.pkl
~/wuji-ego-mint/assets/mano/mano_right/MANO_RIGHT.pkl
~/EgoHandKit/_DATA/data/mano/MANO_RIGHT.pkl
~/EgoHandKit/_DATA/data/mano_mean_params.npz
~/EgoHandKit/_DATA/hamer_ckpts/model_config.yaml
~/EgoHandKit/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
```

The MINT working tree must also contain `scripts/export_egohandkit_cache.py`,
identical to EgoHandKit's `scripts/export_egohandkit_cache_mint.py`. The launcher
does not synchronize or modify these read-only source mounts.

From the roboego-hand-vis repository root:

```bash
python3 docker/hand-env/prepare_assets.py
docker build -t roboego-hand-env:dev docker/hand-env
```

`prepare_assets.py` resolves symlinks, checks every required source, stages real
files in ignored `.build/models`, and records sizes and SHA-256 hashes. Arguments
`--mint-root`, `--ego-root`, and `--output` override staging locations. Use the
default output with this Dockerfile. Building requires network access for the
Ubuntu base, apt, PyPI/PyTorch wheels and the pinned chumpy Git revision.

For a Linux host with a local HTTP proxy, pass it only at build time:

```bash
docker build --network host \
  --build-arg HTTP_PROXY="$HTTP_PROXY" \
  --build-arg HTTPS_PROXY="$HTTPS_PROXY" \
  -t roboego-hand-env:dev docker/hand-env
```

The proxy is not a runtime dependency. Do not bake proxy credentials into the
Dockerfile or pass the host's `ALL_PROXY` into the image.

**Do not publish this image by default.** MANO and pretrained checkpoints have
their own licenses. Local availability does not establish redistribution rights.
The image is intended for private testing until asset permissions are reviewed.

Only direct dependencies are pinned initially; each built environment includes
`installed.txt` recording the resolved transitive versions. Preserve the image
digest and those files. A clean rebuild is not claimed to be bit-for-bit
reproducible until transitive dependencies are also locked. The Ubuntu 22.04
base is pinned by digest through the public ECR mirror; `BASE_IMAGE` is an
overridable build argument.

The published decord 0.6.0 wheel has a stale internal `cp36` tag. The build
allows exactly that one `pip check` platform warning, still rejects dependency
conflicts, and imports decord. The doctor additionally decodes a generated
two-frame video; a successful build alone does not validate decord.

## Mounts And Offline Checks

```bash
export INPUT_DIR=/absolute/path/to/session
export OUTPUT_DIR=/absolute/path/to/separate/hand-results
bash docker/hand-env/run.sh doctor
```

The launcher defaults to sibling `$HOME/wuji-ego-mint`, `$HOME/EgoHandKit`, this
repository and `tmp/container_validation`. Override `MINT_ROOT`, `EGO_ROOT`,
`PROJECT_ROOT`, `INPUT_DIR`, `OUTPUT_DIR`, `HAND_ENV_IMAGE`, or `HAND_GPUS`.
Input and output paths containing commas are unsupported by Docker's mount syntax.

All runs disable container networking. An image-specific Docker model volume is
seeded once from the bundled assets, then mounted read-only at both `~/model`
and the model paths expected by the code. This costs one additional copy of the
models on the Docker host, avoids changing the mounted source, and prevents
host-specific model symlinks from leaking into the runtime. A new image ID uses
a new volume. Unique temporary directories under host `TMPDIR` (default `/tmp`)
isolate EgoHandKit's decoded frame cache. The host launcher removes its own cache
directory when the container exits; it is not retained in the result directory,
even with `--debug`.

The doctor checks both import stacks, CUDA matrix multiplication, torchvision
CUDA NMS, both MANO assets in MINT, and a nonblank headless EGL render. It does
not prove checkpoint inference or algorithm accuracy.

## Documented Algorithm Profile

Use `documented-pipeline.sh` for the document's full algorithm settings. It runs
MINT, canonical export and EgoHandKit with the flags in the local
`EgoHandKit/scripts/run_without_front.sh`. It keeps the default HaMeR batch size
(currently 48), rather than the earlier smoke test's override of 2. If memory
fails, report it before changing the acceptance configuration. Algorithm commands
are kept in the internal `pipeline-stages.sh`; the public entry wraps it with
compact output publishing, without changing model settings.

For a complete session with 30 FPS rectified RGB, registered left-camera depth
PNGs, and at most 10000 frames, use the exported directories above:

```bash
RUN_NAME="documented_$(date +%Y%m%d_%H%M%S)"
bash docker/hand-env/run.sh bash \
  /home/user/roboego-hand-vis/docker/hand-env/documented-pipeline.sh \
  /home/user/data/input/videos/left_rectified.mp4 \
  /home/user/data/input/depth \
  "/home/user/data/output/$RUN_NAME" 10000
```

No image rebuild is needed: the script is mounted from this repository. It
requires an empty/new output directory, 30 FPS video, and matching exporter copies in
EgoHandKit/MINT. It does not copy into read-only repositories. The frame limit
defaults to 10000; shorter tests must trim RGB and matching depth first.
Depth archiving is deliberately not run here.

Successful output is exactly:

```text
NEW_OUTPUT_ROOT/
  hand_tracking.parquet
  render.mp4
  report.json
```

Add `--debug` after the optional frame limit to retain MINT predictions, canonical
observations, EgoHandKit intermediate results and logs under `debug/`:

```bash
bash docker/hand-env/run.sh bash \
  /home/user/roboego-hand-vis/docker/hand-env/documented-pipeline.sh \
  /home/user/data/input/videos/left_rectified.mp4 \
  /home/user/data/input/depth \
  /home/user/data/output/my_debug_run 10000 --debug
```

Without debug, `.work` intermediates are deleted after final artifacts have
been validated/published. Parquet row count and render frame count must match
the input video. `report.json` contains frame/hand counts, model source revisions,
image ID, elapsed time and execution status, with `quality_status: not_validated`.
It does not certify hand-pose accuracy or distinguish every fallback hand by
itself; detailed intermediate results remain available in debug mode.

Failures return nonzero and retain `report.json` plus `error.log` even without
debug. Partial final artifacts are not published. Existing nonempty result
directories are rejected. Hard-killed runs can leave `.work`; use a fresh output
directory instead of treating them as complete. Debug snapshots may contain
temporary paths and are not self-contained replay datasets. Low-level `mint`,
`export`, and `egohandkit` commands retain their original output behavior.

This profile passed a 100-frame end-to-end offline test on 2026-09-27; see
VALIDATION.md. Earlier 32-frame results below used a reduced smoke configuration.

## Prepare And Run 100 Frames

This example expects `INPUT_DIR` to point to a processed session containing
`videos/left_rectified.mp4`, `depth/fast_foundation/depth_uint16_png`, and
`depth/fast_foundation/fast_foundation_stereo_video_meta.json`. Original data is
read-only. Use the same terminal for all commands. Stop if any step fails.

```bash
export INPUT_DIR=/absolute/path/to/processed-session
export OUTPUT_DIR="$(mktemp -d /tmp/hand-test100-XXXXXX)"

bash docker/hand-env/run.sh bash -euc '
mkdir -p /home/user/data/output/sample100/depth/depth_uint16_png
for i in $(seq 0 99); do
  name=$(printf "frame_%06d_depth_mm.png" "$i")
  cp "/home/user/data/input/depth/fast_foundation/depth_uint16_png/$name" \
     /home/user/data/output/sample100/depth/depth_uint16_png/
done
cp /home/user/data/input/depth/fast_foundation/fast_foundation_stereo_video_meta.json \
   /home/user/data/output/sample100/depth/
ffmpeg -hide_banner -v error -n \
  -i /home/user/data/input/videos/left_rectified.mp4 \
  -map 0:v:0 -frames:v 100 -an -vsync 0 -c:v libx264 -preset fast -crf 18 \
  /home/user/data/output/sample100/left_rectified.mp4
'

RUN_NAME="documented100_$(date +%Y%m%d_%H%M%S)"
bash docker/hand-env/run.sh bash \
  /home/user/roboego-hand-vis/docker/hand-env/documented-pipeline.sh \
  /home/user/data/output/sample100/left_rectified.mp4 \
  /home/user/data/output/sample100/depth \
  "/home/user/data/output/$RUN_NAME" 100

find "$OUTPUT_DIR/$RUN_NAME" -type f \
  \( -name render.mp4 -o -name hand_tracking.parquet \)
```

For the exact 100-frame test, verify MINT/export report 100 frames, the final
Parquet has 100 sequential frame rows, and the render decodes to 100 frames.
Review the video visually; missing hands and MINT fallback must be distinguished
from successful HaMeR estimates. In debug mode, consult `debug/run/pipeline.log`
for actual commands and exit status; default mode retains the summary in
`report.json`. The video encoder may pad/resize height to a multiple of
16; the test's 1920x1080 RGB produced a 1920x1088 rendered video.

All paths passed after `run.sh` are container paths. `export INPUT_DIR` and
`export OUTPUT_DIR` set host mount sources; they are not substituted inside the
container command. No shell login or long-lived container is needed.

Do not run depth archiving during this test: its recommended configuration can
delete the depth PNGs required for rerunning hand estimation. Parameter changes
belong in `pipeline-stages.sh`; preserve this validated baseline when making
an experimental copy. No algorithm simplification or automatic tuning occurs.

## Historical 32-Frame Smoke Test (Not Document-Equivalent)

Run stages separately first. Use the *rectified* 32-frame RGB paired with the
prepared depth, not the original raw video. Preserve native 30 FPS: MINT's
default 15 FPS would produce only half the observations.

Registered camera intrinsics are also required for final Parquet mesh rendering.
The current EgoHandKit reader only accepts
`depth32/fast_foundation_stereo_video_meta.json` (or the same under
`fast_foundation/`). This sample includes a compatibility file with the original
FoundationStereo intrinsics and explicit provenance. Its filename does not mean
Fast-FoundationStereo produced the depth. Do not substitute arbitrary intrinsics.

```bash
bash docker/hand-env/run.sh mint infer \
  --input /home/user/data/input/sample32_raw/videos/left_rectified.mp4 \
  --checkpoint /home/user/model/mint/model.safetensors \
  --output /home/user/data/output/mint \
  --target-fps 30 --max-frames 32 --window-batch-size 1 --no-render

bash docker/hand-env/run.sh export \
  --source /home/user/data/input/sample32_raw/videos/left_rectified.mp4 \
  --raw /home/user/data/output/mint/prediction.npz \
  --out /home/user/data/output/observations.pkl

bash docker/hand-env/run.sh egohandkit \
  --input /home/user/data/input/sample32_raw/videos/left_rectified.mp4 \
  --frontend canonical --observations /home/user/data/output/observations.pkl \
  --backend hamer --depth_dir /home/user/data/input/depth32 \
  --output_root /home/user/data/output/egohandkit --batch_size 2 \
  --depth_gate --motion_gate --mint_3d_consistency_gate \
  --hand_tracking_parquet --render_hand_tracking_parquet --no_omega_world
```

The canonical export script currently lives in the MINT working tree; include
it in the mounted code even if it has not been committed upstream. The main
roboego-hand-vis wrapper is not used here: its `--prediction` option and default
observations path do not match the current EgoHandKit interface.

Check prediction/canonical frame counts, final Parquet, decoded render frame
count and nonblank images. Inspect per-hand source fields: the existing pipeline
can fall back to MINT when HaMeR is rejected. Do not count fallback as successful
HaMeR recovery or fabricate missing observations. Record mounted repository commits and
dirty status with results. Algorithm changes alone do not require rebuilding;
new imports, native extensions, checkpoint layouts or runtime version changes do.
