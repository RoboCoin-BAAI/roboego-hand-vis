#!/usr/bin/env python3
"""Compatibility wrapper for the chunked Stage 1 subtask runner."""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.post_annotation_actions.model_subtasks import (  # noqa: E402
    frame_chunks,
    main,
    merge_chunk_subtasks,
)


__all__ = ["frame_chunks", "merge_chunk_subtasks"]


if __name__ == "__main__":
    main()
