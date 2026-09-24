# Video Simple Quality Stereo Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `video_simple_quality` support a default binocular session mode with a top-level pass/fail conclusion.

**Architecture:** Keep the existing single-video analyzer unchanged. Extend the CLI layer to accept either `--video` for legacy single-video reports or `--session` for default left/right session reports, then aggregate per-stream reports into one JSON.

**Tech Stack:** Python argparse, dataclasses, JSON, pytest monkeypatch.

---

### Task 1: Add Session Mode Tests

**Files:**
- Create: `tests/video_simple_quality/test_cli.py`
- Modify: `roboego_hand_vis/video_simple_quality/cli.py`

- [ ] Write a failing test that `--session` defaults to `videos/left.mkv` and `videos/right.mkv`, writes `annotation/video_simple_quality.json`, and reports `passed=True` when both streams pass.
- [ ] Write a failing test that any stream with a `drop_before_qwen` interval makes the aggregate `passed=False`.
- [ ] Run `PYTHONPATH=. python -m pytest tests/video_simple_quality/test_cli.py -q` and confirm failure.
- [ ] Implement the smallest CLI/dataclass changes to pass.
- [ ] Re-run the same test command and confirm pass.

### Task 2: Update Stage Metadata And Defaults

**Files:**
- Modify: `roboego_hand_vis/video_simple_quality/recommended.yaml`
- Modify: `roboego_hand_vis/video_simple_quality/stage.yaml`

- [ ] Change recommended IO from a single left mp4 to session-level left/right mkv defaults.
- [ ] Document that output is an aggregate report with `passed`.
- [ ] Run `python -m py_compile roboego_hand_vis/video_simple_quality/cli.py`.
