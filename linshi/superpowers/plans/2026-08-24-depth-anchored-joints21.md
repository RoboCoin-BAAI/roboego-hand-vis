# Depth Anchored Joints21 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export final hand 21 joints by trusting HaMeR mesh pose/projection and using stereo depth only as a weak wrist/palm Z anchor.

**Architecture:** Add a focused egohand module that converts final HaMeR meshes to 21 joints, samples depth around wrist/palm projections, and shifts the whole hand only along camera Z. Add a CLI script that writes JSON/NPZ plus lightweight RGB skeleton contact sheets.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing RoboEgo hand utilities.

---

### Task 1: Core Depth Anchor Logic

**Files:**
- Create: `roboego_hand_vis/egohand/depth_anchored_joints21.py`
- Test: `tests/egohand/test_depth_anchored_joints21.py`

- [ ] Write failing tests for depth window median, Z-only translation, and missing-depth fallback.
- [ ] Implement minimal helpers for projected anchor sampling and Z-only joint shifting.
- [ ] Run `pytest tests/egohand/test_depth_anchored_joints21.py -q`.

### Task 2: CLI Export Script

**Files:**
- Create: `scripts/hand/export_depth_anchored_joints21.py`
- Test: extend `tests/egohand/test_depth_anchored_joints21.py`

- [ ] Add a CLI that reads `final_hand_candidates_gated.json`, `dataset_manifest.json`, and depth `.npy` files.
- [ ] Write `joints21_depth_anchored.json`, `joints21_depth_anchored.npz`, `summary.json`, and sampled RGB overlay images.
- [ ] Run the CLI on the current hand window and inspect the contact sheet.

### Task 3: Verification

**Files:**
- Verify only.

- [ ] Run focused tests and `py_compile`.
- [ ] Report exact output paths and summary counts.
