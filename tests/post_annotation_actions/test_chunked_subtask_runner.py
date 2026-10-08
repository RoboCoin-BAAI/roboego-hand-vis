from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_stage1_subtasks_chunked.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_stage1_subtasks_chunked", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_chunks_cover_all_frames_without_overlap() -> None:
    module = _load_script_module()
    frames = [{"t": idx} for idx in range(5)]

    chunks = module.frame_chunks(frames, chunk_size=2, overlap=0)

    assert [[frame["t"] for frame in chunk] for chunk in chunks] == [[0, 1], [2, 3], [4]]


def test_frame_chunks_support_small_overlap() -> None:
    module = _load_script_module()
    frames = [{"t": idx} for idx in range(5)]

    chunks = module.frame_chunks(frames, chunk_size=3, overlap=1)

    assert [[frame["t"] for frame in chunk] for chunk in chunks] == [[0, 1, 2], [2, 3, 4]]


def test_merge_chunk_subtasks_renumbers_ids_and_skips_exact_duplicates() -> None:
    module = _load_script_module()

    merged = module.merge_chunk_subtasks(
        [
            {
                "chunk_index": 0,
                "subtasks": [
                    {"id": 10, "action": "idle", "description": "rest", "source_frames": [0]},
                    {"id": 11, "action": "approach", "description": "approach rack", "source_frames": [1]},
                ],
            },
            {
                "chunk_index": 1,
                "subtasks": [
                    {"id": 1, "action": "approach", "description": "approach rack", "source_frames": [1]},
                    {"id": 2, "action": "secure", "description": "secure rack", "source_frames": [2]},
                ],
            },
        ]
    )

    assert [subtask["id"] for subtask in merged] == [1, 2, 3]
    assert [subtask["description"] for subtask in merged] == ["rest", "approach rack", "secure rack"]
    assert all("source_chunk_index" not in subtask for subtask in merged)
