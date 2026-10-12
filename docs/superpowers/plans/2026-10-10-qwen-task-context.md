# Qwen Task Context Implementation Plan

**Goal:** Automatically use session task metadata to provide an English scene, a detailed overall task, and a closed object vocabulary for Qwen annotation.

**Architecture:** Read `任务详情.txt` and one `*_config.yaml` beside the session's `videos` directory (or beside a standalone video). Prefer YAML `lib_name` for object identity and merge repeated instances. Translate the extracted semantic information once through the configured Qwen endpoint, cache it, inject it into every batch, and validate all structured object names against the translated vocabulary. Preserve visual evidence requirements and the existing frame schema; store the overall task separately as report metadata.

**Tech stack:** Python, PyYAML, existing urllib Qwen client, pytest.

The user approved automatic task context and explicitly requires English scene/task text and using only supplied objects. Continue implementation in the existing workspace containing their database integration.

## Tasks

- [x] Add failing tests in `tests/qwen_annotation/test_task_context.py` for multiline TXT extraction, YAML object deduplication, partial/missing metadata, ambiguous/invalid metadata, translation validation, and closed object vocabulary.
- [x] Implement `roboego_hand_vis/qwen_annotation/task_context.py`: metadata discovery/extraction, strict English translation schema, context prompt and frame validation. TXT task content supplies full instructions; YAML notes are retained as additional task detail. YAML objects are authoritative, with TXT props used when YAML objects are unavailable.
- [x] Extend `frame_annotation.py` with an optional response parser so the existing retry/HTTP handling supports translation and context-aware frame validation without changing legacy callers. Make retry instructions schema-neutral.
- [x] Add failing integration tests using temporary sessions and mocked HTTP/frame extraction for translation caching, batch injection, persisted metadata, recovery validation, and changed/legacy context rejection.
- [x] Integrate context preparation before frame extraction in `compute.py`. Bind cache to source content, prompt and model; reject incompatible existing artifacts before writing or making requests. Carry metadata into final/batch/checkpoint/chunk outputs and validate recovered frames against the vocabulary.
- [x] Document automatic discovery, English output, strict vocabulary behavior, one-time translation cost, errors and fresh-output requirements in the Qwen README.
- [x] Run `/home/liuyou/miniconda3/envs/ego/bin/python -m pytest tests/qwen_annotation -q`, inspect a local extraction of the supplied sample, request an independent code review, and resolve material findings.

## Behavior details

- No metadata: preserve legacy behavior. One metadata file: use its available task/scene/object fields, represent unavailable scene or task as `unknown`; do not invent objects.
- Malformed YAML, multiple matching YAML files, or metadata with no usable object list: fail with a clear error instead of silently falling back to unconstrained annotation.
- Translation returns `scene`, `task_description`, and `objects` containing `source_name` and English `name`. Require exactly one mapping per canonical source object and unique English names. Include supplied identifying color/material in names where useful.
- Only allowed names may appear in interaction/attention/task object arrays and attribute names. Activity must reuse interaction names; prompt also restricts task-object references in free text. Unmatched objects are omitted with uncertainty in notes; privacy/background observations remain possible.
- Task instructions describe intent, never proof that an action occurred. Per-frame scene remains a visual description; metadata.scene records the translated collection scene.
- Cache translation usage separately from frame usage. Store source file hashes and source/translated context for reproducibility. Context changes or legacy output lacking context require a fresh output directory; never delete old results automatically.

## Verification

- Qwen and downstream post-annotation suites: 79 passed. HTTP responses and frame extraction are mocked in integration tests; no live inference or database mutation was performed.
- Supplied real session metadata extracts the complete two-step task and exactly two canonical objects (黄色收纳盒 and 砂纸); renderer settings and collector identity are excluded.
- Independent review findings fixed: colon-containing TXT task steps are preserved, translation JSON example is valid, and required per-frame scene/activity plus nonempty notes reject Chinese descriptions.
