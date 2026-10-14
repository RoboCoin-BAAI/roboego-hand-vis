"""Prevent checkpoints from silently mixing different sampling or prompt settings."""

import hashlib
import json
from pathlib import Path


def bind_run_settings(config):
    stat = config.video.stat()
    settings = {
        'version': 1, 'video': str(config.video), 'video_size': stat.st_size, 'video_mtime_ns': stat.st_mtime_ns,
        'prompt_sha256': hashlib.sha256(config.prompt.read_bytes()).hexdigest(), 'model': config.model,
        'endpoint_sha256': hashlib.sha256(config.base_url.encode()).hexdigest(),
        'chat_template_kwargs': config.chat_template_kwargs,
        'fps': config.fps, 'start_s': config.start_s, 'duration_s': config.duration_s,
        'max_width': config.max_width, 'batch_size': config.batch_size, 'overlap_frames': config.overlap_frames,
        'image_mode': config.image_mode, 'grid_columns': config.grid_columns,
        'grid_cell_width': config.grid_cell_width,
        'windows_sha256': hashlib.sha256(config.windows_json.read_bytes()).hexdigest() if config.windows_json else None,
    }
    path = config.raw_output_dir / 'run_settings.json'
    error = 'Annotation sampling/settings differ from existing artifacts; use a fresh output directory (including raw/batch/checkpoint paths).'
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(error)
        return
    artifacts = [p for p in (config.output_json, config.checkpoint_json) if p.exists()]
    artifacts += list(config.batch_output_dir.glob('batch_*.json'))
    artifacts += list(config.raw_output_dir.glob('batch_*.raw.txt'))
    artifacts += list(config.chunk_output_dir.glob('chunk_*.json'))
    # Legacy outputs have no prompt/content signature, even if their path is unchanged.
    # Keep them readable, but never silently adopt them into a different run.
    if artifacts:
        raise ValueError(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(settings, sort_keys=True, indent=2))
    temporary.replace(path)
