# Pipeline Stage Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin launcher that runs a Kedro node from a dataset config, session directory, and stage name.

**Architecture:** Keep Kedro nodes unchanged. Add a focused launcher module that loads `conf/datasets/<name>.yml`, derives common session/segment/hand-window paths, builds `kedro run --pipelines ... --nodes ... --params ...`, and optionally executes it.

**Tech Stack:** Python standard library, PyYAML, Kedro CLI, pytest.

---

### Task 1: Dataset Config

**Files:**
- Create: `conf/datasets/wumei.yml`

- [ ] Add dataset defaults for runtimes, stage mapping, and path templates.
- [ ] Include at least the current hand-window stages: stereo quality, fast depth, HaMeR, hand filter, joints21 gate, depth anchoring, final result, trajectory polish, SAM3.

### Task 2: Launcher Module

**Files:**
- Create: `roboego_hand_vis/pipeline_launcher.py`
- Test: `tests/test_pipeline_launcher.py`

- [ ] Write tests for loading `wumei.yml`.
- [ ] Write tests for command generation for `depth_anchored_joints21`.
- [ ] Write tests for `--dry-run` printing the generated command without executing.
- [ ] Implement path discovery for `--hand-window` and `--all-hand-windows`.

### Task 3: CLI Wrapper

**Files:**
- Create: `scripts/run_pipeline_stage.py`
- Modify: `README.md`

- [ ] Add CLI args: `--dataset`, `--config`, `--session`, `--stage`, `--hand-window`, `--all-hand-windows`, `--dry-run`, `--print-command`.
- [ ] Document short examples replacing long Kedro commands.

### Task 4: Verification

**Files:**
- Test: `tests/test_pipeline_launcher.py`

- [ ] Run `PYTHONPATH=. /home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/test_pipeline_launcher.py -q`.
- [ ] Run full test suite.
- [ ] Run `git diff --check`.
