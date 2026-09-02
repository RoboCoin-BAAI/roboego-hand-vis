# Kedro Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal Kedro project skeleton for the existing RoboEgo processing codebase.

**Architecture:** Kedro owns pipeline registration, configuration layout, and stable stage boundaries. Existing scripts remain callable while pipeline nodes are migrated incrementally.

**Tech Stack:** Python 3.11, Kedro 1.5.0, pytest, existing `roboego_hand_vis` package.

---

### Task 1: Registry And Pipeline Packages

**Files:**
- Create: `roboego_hand_vis/pipeline_registry.py`
- Create: `roboego_hand_vis/pipelines/*`
- Test: `tests/pipelines/test_kedro_registry.py`

- [ ] **Step 1: Write failing tests for registered pipeline names.**
- [ ] **Step 2: Run the registry test and confirm import failure.**
- [ ] **Step 3: Add empty Kedro pipeline packages for datasets, quality, semantic, depth, egohand, objects, export, and experiments.**
- [ ] **Step 4: Register all pipelines and an empty `__default__` pipeline.**
- [ ] **Step 5: Run the registry test and confirm it passes.**

### Task 2: Core Runtime And Reports

**Files:**
- Create: `roboego_hand_vis/core/runtime.py`
- Create: `roboego_hand_vis/core/reports.py`
- Create: `roboego_hand_vis/core/artifacts.py`
- Create: `roboego_hand_vis/core/paths.py`
- Test: `tests/core/test_runtime_reports.py`

- [ ] **Step 1: Write failing tests for command construction and report writing.**
- [ ] **Step 2: Run the core test and confirm import failure.**
- [ ] **Step 3: Add minimal runtime, report, artifact, and path helpers.**
- [ ] **Step 4: Run the core test and confirm it passes.**

### Task 3: Kedro Project Metadata And Config

**Files:**
- Modify: `pyproject.toml`
- Create: `conf/base/catalog.yml`
- Create: `conf/base/parameters.yml`
- Create: `conf/base/parameters_runtimes.yml`
- Create: `conf/base/parameters_paths.yml`
- Create: `conf/local/.gitkeep`

- [ ] **Step 1: Add `[tool.kedro]` metadata with `source_dir = "."`.**
- [ ] **Step 2: Add base config files for catalog, runtime names, and common paths.**
- [ ] **Step 3: Run `kedro registry list` in the `ego` environment.**
- [ ] **Step 4: Run `kedro run --pipeline depth` in the `ego` environment.**
