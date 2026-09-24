# Detect Hand Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `roboego_hand_vis/detect_hand` from a standalone script into a standard stage with `run.py`, `cli.py`, `recommended.yaml`, and `stage.yaml`.

**Architecture:** Keep YOLO detection in `compute.py`. Add `cli.py` for YAML parsing, path defaults, and JSON summary output; add `run.py` as the stable module/script entrypoint; add metadata YAML files that match the newer stage layout.

**Tech Stack:** Python argparse/dataclasses, PyYAML, pytest monkeypatch.

---

### Task 1: CLI Configuration

**Files:**
- Create: `tests/detect_hand/test_cli.py`
- Create: `roboego_hand_vis/detect_hand/cli.py`

- [ ] Write a failing test that default YAML config resolves `detector.pt` relative to `detect_hand`.
- [ ] Write a failing test that `run_detect_hand_stage()` calls `process_video()` and returns a compact summary.
- [ ] Run `PYTHONPATH=. python -m pytest tests/detect_hand/test_cli.py -q` and confirm failure.
- [ ] Implement `DetectHandConfig`, `parse_args`, `load_yaml`, `config_from_yaml`, and `run_detect_hand_stage`.
- [ ] Re-run the test and confirm pass.

### Task 2: Stage Files

**Files:**
- Create: `roboego_hand_vis/detect_hand/run.py`
- Create: `roboego_hand_vis/detect_hand/recommended.yaml`
- Create: `roboego_hand_vis/detect_hand/stage.yaml`
- Create: `roboego_hand_vis/detect_hand/__init__.py`

- [ ] Add `run.py` that supports both package and direct stage execution.
- [ ] Add `recommended.yaml` with defaults for model, sample interval, confidence, device, image size, and batch size.
- [ ] Add `stage.yaml` with entrypoint, inputs, outputs, runtime, and dependencies.
- [ ] Run `python -m py_compile roboego_hand_vis/detect_hand/cli.py roboego_hand_vis/detect_hand/run.py roboego_hand_vis/detect_hand/compute.py`.
- [ ] Run `python -m roboego_hand_vis.detect_hand.run --help`.
