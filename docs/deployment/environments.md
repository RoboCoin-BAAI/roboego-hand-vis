# Pipeline Environment Setup

This project keeps the Kedro orchestration code in `roboego-hand-vis`. Heavy
models can still live in separate environments, but each pipeline stage should
be launched through this repository.

## Recommended Machines

- `base_prepare`: CPU machine is enough for planning; stereo preparation also
  needs local video I/O and RoboEgoPipe.
- `phase1_depth`: GPU machine recommended. RTX 4090/A800 class is preferred for
  Fast-FoundationStereo.
- `egohand`: GPU machine with the HaMeR environment and MANO assets.
- `objects`: GPU machine with SAM3 installed and the SAM3 checkpoint.

## Base Environment: `ego`

Use this for Kedro, data quality, semantic post-processing, materialization,
depth cache archival, and most lightweight nodes.

```bash
cd /path/to/roboego-hand-vis
source /path/to/conda/etc/profile.d/conda.sh
conda create -n ego python=3.11 -y
conda activate ego
pip install -e ".[dev]"
```

System packages normally required by OpenCV/video tools:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs libgl1 libglib2.0-0
```

For Qwen annotation:

```bash
export DASHSCOPE_API_KEY=...
```

## Depth/SAM3 Environment: `py312`

Use this for Fast-FoundationStereo and SAM3 subprocesses.

```bash
source /path/to/conda/etc/profile.d/conda.sh
conda create -n py312 python=3.12 -y
conda activate py312
cd /path/to/roboego-hand-vis
pip install -e ".[dev]"
```

External repositories/checkpoints used by depth and SAM3:

```text
/path/to/Fast-FoundationStereo
/path/to/do-as-i-do/sam3/sam3.pt
```

SAM3 runner entrypoint is now local to this repository:

```text
scripts/object/run_sam3_video.py
```

The runner imports the SAM3 backend in this order:

```text
SAM3_PKG_DIR
<roboego-hand-vis>/external/sam3
/path/to/do-as-i-do/reconstruction/modules/sam3
```

Recommended explicit setting during migration:

```bash
export SAM3_PKG_DIR=/path/to/do-as-i-do/reconstruction/modules/sam3
```

This keeps the script dependency inside `roboego-hand-vis`; only the SAM3
backend module and checkpoint remain external until we vendor or package SAM3
directly.

## HaMeR Environment: `hamer`

Use this only for hand mesh inference.

```text
/path/to/hamer
/path/to/hamer/_DATA/data/mano
```

The Kedro stage launches HaMeR as a subprocess, usually with:

```text
python
```

## Optional Restoration Environment

Use this only when running video deblurring before depth estimation.

```text
/path/to/RealVDeblur
```

Typical large model assets are kept outside this repository.

## Required Model Assets

Keep model weights outside git. The default config currently points to these
local paths:

```text
Fast-FoundationStereo:
  /path/to/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth

Legacy FoundationStereo:
  /path/to/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth

HaMeR MANO assets:
  /path/to/hamer/_DATA/data/mano

SAM3:
  /path/to/do-as-i-do/sam3/sam3.pt

RealVDeblur:
  /path/to/RealVDeblur/checkpoints/Wan2.1-T2V-1.3B
  /path/to/RealVDeblur/checkpoints/realvdeblur_dmd.safetensors
```

If a machine uses different paths, override them in `conf/datasets/raw.yml` or
pass stage params from the launcher.

## Pipeline Dependency Matrix

| Stage or workflow | Environment | External repo/model |
| --- | --- | --- |
| `raw_quality` | `ego` | none |
| `video_simple_quality` | `ego` | none |
| `qwen_annotation` | `ego` | Qwen API key |
| `post_annotation_actions` | `ego` | Qwen API key for resolver |
| `processing_plan` | `ego` | none |
| `hand_windows` | `ego` | none |
| `base_prepare` | `ego` | Qwen API key, `/path/to/RoboEgoPipe` |
| `hand_visibility_gate` | `ego` | none |
| `materialize_session` | `ego` | none |
| `materialize_hand_windows` | `ego` | none |
| `materialize_segment_hand_windows` | `ego` | none |
| `prepare_stereo` | `ego` | `/path/to/RoboEgoPipe` |
| `stereo_summary_quality` | `ego` | none |
| `stereo_quality` | `ego` | none |
| `fast_foundation_depth` | `ego` launcher + `py312` subprocess | `/path/to/Fast-FoundationStereo` |
| `depth_numeric_quality` | `ego` | none |
| `archive_depth_cache` | `ego` | none |
| `phase1_depth` | `ego` launcher + `py312` subprocess | RoboEgoPipe, Fast-FoundationStereo |
| `hamer` | `ego` launcher + `hamer` subprocess | `/path/to/hamer`, MANO assets |
| `hand_filter` | `ego` | none |
| `joints21_pose_gate` | `ego` | none |
| `depth_anchored_joints21` | `ego` | depth PNG cache |
| `sam3_objects` | `ego` launcher + `py312` subprocess | SAM3 backend, `sam3.pt` checkpoint |

## Smoke Commands

Planning machine:

```bash
cd /path/to/roboego-hand-vis
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/session \
  --stage base_prepare \
  --dry-run
```

Depth machine:

```bash
cd /path/to/roboego-hand-vis
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/session \
  --stage phase1_depth \
  --dry-run
```

SAM3 object segmentation:

```bash
cd /path/to/roboego-hand-vis
export SAM3_PKG_DIR=/path/to/do-as-i-do/reconstruction/modules/sam3
python scripts/run_pipeline_stage.py \
  --dataset raw \
  --session /path/to/session \
  --stage sam3_objects \
  --hand-window handwin_0000 \
  --dry-run
```
