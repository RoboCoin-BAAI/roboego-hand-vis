"""Translate and bind session context to annotation artifacts before a run."""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from .frame_annotation import call_qwen_frame_annotations
from roboego_hand_vis.qwen_annotation.task_context import CONTEXT_RULES, TRANSLATION_PROMPT, load_task_context, parse_task_translation

if TYPE_CHECKING:
    from .compute import QwenAnnotationConfig


def prepare_task_metadata(config: QwenAnnotationConfig) -> dict[str, Any] | None:
    source = load_task_context(config.video)
    cache_path = config.raw_output_dir / "task_context.json"
    json_artifacts = [path for path in (config.output_json, config.checkpoint_json) if path.exists()]
    json_artifacts += sorted(config.batch_output_dir.glob("batch_*.json"))
    json_artifacts += sorted(config.chunk_output_dir.glob("chunk_*.json"))
    raw_artifacts = list(config.raw_output_dir.glob("batch_*.raw.txt"))
    mismatch = "Task context differs from existing artifacts; use a fresh output directory (including raw/batch/checkpoint paths)."
    if source is None:
        if cache_path.exists() or any(
            json.loads(path.read_text(encoding="utf-8")).get("task_metadata") is not None
            for path in json_artifacts
        ):
            raise ValueError(mismatch)
        return None

    signature = {
        "version": 1, "source": source, "video": str(config.video.resolve()),
        "prompt_path": str(config.prompt.resolve()), "prompt": config.prompt.read_text(encoding="utf-8"),
        "translation_prompt": TRANSLATION_PROMPT, "context_rules": CONTEXT_RULES,
        "model": config.model, "base_url": config.base_url, "chat_template_kwargs": config.chat_template_kwargs,
        "fps": config.fps, "start_s": config.start_s, "duration_s": config.duration_s,
        "max_width": config.max_width, "batch_size": config.batch_size,
        "windows": config.windows_json.read_text(encoding="utf-8") if config.windows_json else None,
    }
    signature['image_mode'] = config.image_mode
    signature['grid_columns'] = config.grid_columns
    signature['grid_cell_width'] = config.grid_cell_width
    if config.overlap_frames:
        signature["overlap_frames"] = config.overlap_frames
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if cache_path.exists():
        try:
            metadata = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or metadata.get("fingerprint") != fingerprint:
                raise ValueError(mismatch)
            parsed = parse_task_translation(json.dumps(metadata), source)
            if parsed["status"] != "ok":
                raise ValueError(mismatch)
        except (ValueError, TypeError) as exc:
            raise ValueError(mismatch) from exc
        for path in json_artifacts:
            if json.loads(path.read_text(encoding="utf-8")).get("task_metadata") != metadata:
                raise ValueError(f"{path}: {mismatch}")
        return metadata
    if json_artifacts or raw_artifacts:
        raise ValueError(mismatch)

    semantic_source = {key: value for key, value in source.items() if key != "sources"}
    result = call_qwen_frame_annotations(
        messages=[{"role": "system", "content": TRANSLATION_PROMPT},
                  {"role": "user", "content": json.dumps(semantic_source, ensure_ascii=False)}],
        model=config.model, api_key_env=config.api_key_env, base_url=config.base_url,
        chat_template_kwargs=config.chat_template_kwargs, timeout_s=config.timeout_s,
        max_retries=config.max_retries, retry_sleep_s=config.retry_sleep_s,
        response_parser=lambda text: parse_task_translation(text, source),
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"Task context translation failed: {result.get('status')}: {result.get('error') or result.get('raw_response', '')}")
    metadata = {
        **result["context"], "fingerprint": fingerprint, "source": source,
        "translation_usage": result.get("usage", {}),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(cache_path)
    return metadata
