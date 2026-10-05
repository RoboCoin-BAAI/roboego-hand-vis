# EGL-Only Candidate

This patch image preserves the original entrypoint, model assets, Python
environments, thread settings and pipeline. Only the NVIDIA EGL vendor
registration and vendor-selection environment variable change. No Ray packages
are added. The full hand Dockerfile also includes this fix for future rebuilds.

The old image is retained as `roboego-hand-env:pre-egl-c8a998c05a1e`.
`roboego-hand-env:dev` was promoted to the EGL-fixed candidate on 2026-10-05
after full CPU/GPU rendering replays and output checks. See [VALIDATION.md](VALIDATION.md).
The Ray render image is already separately fixed.

```bash
docker build --network none \
  --build-arg HAND_BASE=roboego-hand-env:pre-egl-c8a998c05a1e \
  -t roboego-hand-env:egl-candidate docker/hand-env/egl-fix
```

Candidate ID: `sha256:ccd9778457140d26c38c780bfe3ad3f407c23fe919fe90fdbd746c240d8d0f8d`.
Both MINT and EgoHandKit `pip freeze` lists match the original image exactly.
The replay probe reports Mesa llvmpipe for the original image, NVIDIA RTX 4070
for this candidate. The probe is outside the timed render pipeline and adds one
small warm-up context; it is now present in both replay variants.

Run the existing full render replay command with
`HAND_ENV_IMAGE=roboego-hand-env:dev` and a fresh output directory. The temporary
`egl-candidate` alias can be removed after promotion; it is not needed at runtime.
It retains per-frame creation/deletion, fit settings, all frames and final video.
The report now records `egl.renderer`, `egl.vendor`, `egl.version` and the
selected EGL vendor file. Do not attribute any backend-switch gain to Ray.
For a strictly matched pair, rerun with the preserved original image too;
the earlier replay is a historical reference without a recorded EGL probe.

Both versions write to separate outputs. GPU/software rasterization may differ;
do not silently relax pixel comparisons or treat `comparison_mismatch` as a
crash. Review fitting results, completeness, pixel differences and appearance.
The original strict-comparison reports remain unchanged, including exit code 2.

Rollback (only when intentionally restoring the software-rendering image):

```bash
docker tag roboego-hand-env:pre-egl-c8a998c05a1e roboego-hand-env:dev
```
