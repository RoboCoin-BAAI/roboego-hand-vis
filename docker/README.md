# Docker Setup And Usage

Run commands from the delivered `roboego-hand-vis` repository root.
There are two runtime workflows, plus an optional VIO development image:

| Workflow | Image | Host code needed at runtime | GPU |
| --- | --- | --- | --- |
| MINT + EgoHandKit | `roboego-hand-env:dev` | This repository + validated MINT and EgoHandKit snapshots | NVIDIA required |
| VIO (normal use) | `roboego-vio-runtime:dev` | None; only launcher, input and output directories | Not required |
| VIO development | `roboego-vio-env:dev` | This repository + OpenVINS fork + build directory | Not required |

- [Hand: setup, build/import, full run, 100-frame test, outputs](hand-env/README.md)
- [VIO: setup, build/import, raw/rectified runs, NPZ format](vio-env/README.md)
- [VIO runtime details and original Python entry](vio-env/RUNTIME.md)

## Host Installation

Use Linux x86_64. On Ubuntu, install Docker Engine and Buildx using the
[official Docker installation guide](https://docs.docker.com/engine/install/ubuntu/).
These launchers expect `docker` to work as the current user. Follow Docker's
[post-install instructions](https://docs.docker.com/engine/install/linux-postinstall/)
if necessary; Docker group membership grants root-level privileges. Do not
blindly run launchers with sudo: HOME, mounts and output ownership can change.

```bash
docker version
docker info
docker buildx version
```

For hand estimation only, install an NVIDIA driver appropriate for the GPU and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
After installing the toolkit, configure Docker (restarting Docker can interrupt
existing workloads, so schedule this with the machine owner):

```bash
nvidia-smi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The hand launcher uses Docker volume subpaths (locally tested with Docker
29.0.4). Use a compatible Engine. The target hand GPU has 24 GB VRAM; local
12 GB smoke tests do not certify every full-resolution workload on 12 GB.
No host Conda environment, CUDA toolkit or ROS installation is required for
inference. Build helpers use host Python 3; algorithm dependencies live
inside images. Host installation and initial builds may need network access;
inference launchers disable networking.

## Import Or Build

If you received image TAR files, import them. Do not rebuild dependencies or
download model weights again:

```bash
docker load -i /absolute/path/to/hand-env.tar
docker load -i /absolute/path/to/vio-runtime.tar
docker image inspect roboego-hand-env:dev --format '{{.Id}}'
docker image inspect roboego-vio-runtime:dev --format '{{.Id}}'
```

Use `HAND_ENV_IMAGE` / `VIO_RUNTIME_IMAGE` if delivered tags differ.
There is no assumed public registry. To build instead, follow the two module
READMEs: hand assets must be staged first; VIO needs the OpenVINS fork containing
`ov_offline`, not an unmodified official checkout.

Hand weights are bundled. Mounted source snapshots are NOT in the TAR; deliver
them separately, including local modifications and the matching canonical
exporter. VIO runtime contains code and compiled binaries, so its recipient
does not clone or compile OpenVINS.

## Runtime Rules

- `INPUT_DIR` and `OUTPUT_DIR` are absolute host paths. Arguments passed into
  the container use the documented container paths, not arbitrary host paths.
- Use `export` in the same terminal, or assignments immediately before a command.
- Each launcher creates a temporary container with `--rm`; no service needs
  starting and no container needs entering first. Host outputs persist.
- Keep output separate from input and use a fresh result directory per run.
- Inputs/source are read-only; outputs/caches are writable; runs are offline.
- Hand source edits apply on the next run; dependency/model changes require a
  rebuild. Bundled VIO code changes always require a runtime-image rebuild.
- Default outputs are compact. `--debug` retains intermediates on disk, not an
  additional in-memory copy. Hard termination can leave temporary files.
- Success means execution completed, not guaranteed hand/VIO accuracy.

## Export For Another Machine

```bash
docker save -o /absolute/path/to/hand-env.tar roboego-hand-env:dev
docker save -o /absolute/path/to/vio-runtime.tar roboego-vio-runtime:dev
sha256sum /absolute/path/to/hand-env.tar /absolute/path/to/vio-runtime.tar
```

Transfer archives, checksum record, instructions, launchers and required hand
source snapshots privately. After rebuilding, export a NEW TAR: old archives
do not update. Model licenses (including MANO) must permit redistribution; local
access alone does not grant that permission. Data, outputs and model volumes
are not included by `docker save`.

To transfer the optional VIO development image:

```bash
docker save -o /absolute/path/to/vio-env.tar roboego-vio-env:dev
```

That image is not the standalone runtime: it still needs mounted code and
compilation. Preserve image IDs and matching source revisions.
