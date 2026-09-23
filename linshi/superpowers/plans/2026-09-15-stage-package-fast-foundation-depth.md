# Fast Foundation Depth Stage Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Fast-FoundationStereo depth stage into a self-contained stage package while preserving the existing script entrypoint.

**Architecture:** Create `roboego_hand_vis/stages/fast_foundation_depth/` as the home for stage-owned compute, CLI, and future visualization code. The legacy `scripts/run_fast_foundation_stereo_video.py` becomes a thin compatibility wrapper that imports the stage package and delegates execution.

**Tech Stack:** Python, pytest, OpenCV, NumPy, Torch, OmegaConf, existing depth manifest helpers.

---

### Task 1: Establish Stage Package Boundary

**Files:**
- Create: `roboego_hand_vis/stages/__init__.py`
- Create: `roboego_hand_vis/stages/fast_foundation_depth/__init__.py`
- Create: `roboego_hand_vis/stages/fast_foundation_depth/compute.py`
- Create: `roboego_hand_vis/stages/fast_foundation_depth/cli.py`
- Create: `roboego_hand_vis/stages/fast_foundation_depth/visualization.py`
- Modify: `scripts/run_fast_foundation_stereo_video.py`
- Modify: `tests/depth/test_fast_foundation_stereo_video.py`

- [x] **Step 1: Write the failing import-boundary test**

```python
from roboego_hand_vis.stages.fast_foundation_depth.compute import disparity_to_depth_original_size, to_jsonable
from roboego_hand_vis.stages.fast_foundation_depth.visualization import make_depth_preview_panel_rgb
from roboego_hand_vis.stages.fast_foundation_depth.cli import parse_args
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/depth/test_fast_foundation_stereo_video.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'roboego_hand_vis.stages'`.

- [x] **Step 3: Move the implementation**

Move pure helpers and the stage runner from `scripts/run_fast_foundation_stereo_video.py` into the new package:
- `compute.py`: config dataclass, JSON conversion, model loading, frame IO, disparity-to-depth conversion, `run_fast_foundation_depth`.
- `visualization.py`: depth visualization and preview panel helpers.
- `cli.py`: `parse_args`, `config_from_args`, and `main`.
- legacy script: import `main`, `parse_args`, and compatibility helper names from the package.

- [x] **Step 4: Run focused tests**

Run: `pytest tests/depth/test_fast_foundation_stereo_video.py -q`
Expected: PASS.

- [x] **Step 5: Run related depth tests**

Run: `pytest tests/depth/test_fast_foundation_stereo_video.py tests/pipelines/test_depth_foundation_node.py -q`
Expected: PASS.
