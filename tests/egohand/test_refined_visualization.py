import json
from pathlib import Path

import cv2
import numpy as np

from scripts.hand.run_hamer_ego_filter import (
    _prepare_refined_visualization_output_dir,
    _write_refined_selection_preview,
)


def test_prepare_refined_visualization_output_dir_removes_stale_generated_frames(tmp_path: Path):
    output_dir = tmp_path / "refined_annotated"
    output_dir.mkdir()
    stale_frame = output_dir / "frame_000286_refined.jpg"
    stale_frame.write_bytes(b"old")
    unrelated = output_dir / "manual_note.txt"
    unrelated.write_text("keep", encoding="utf-8")
    contact_sheet = tmp_path / "refined_contact_sheet.jpg"
    contact_sheet.write_bytes(b"old-sheet")

    _prepare_refined_visualization_output_dir(output_dir, contact_sheet)

    assert not stale_frame.exists()
    assert unrelated.exists()
    assert not contact_sheet.exists()
    assert output_dir.exists()


def test_write_refined_selection_preview_writes_video_without_frame_images(tmp_path: Path):
    video_path = tmp_path / "left.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (128, 96))
    assert writer.isOpened()
    for idx in range(3):
        image = np.full((96, 128, 3), 30 + idx * 40, dtype=np.uint8)
        writer.write(image)
    writer.release()

    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps({"videos": {"left": {"path": str(video_path), "fps": 5.0}}}),
        encoding="utf-8",
    )
    refined_json = tmp_path / "ego_hand_candidates_refined.json"
    refined_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_idx": idx,
                        "refined_candidates": [
                            {
                                "candidate_id": idx,
                                "bbox_xyxy": [10, 12, 30, 36],
                                "refine_role": "ego_left",
                                "refine_quality": "reliable",
                                "ego_prior_score": 4.0,
                            }
                        ],
                        "refine": {"selected_candidate_ids": {"left": idx}},
                    }
                    for idx in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = _write_refined_selection_preview(
        manifest=manifest,
        refined_json=refined_json,
        preview_path=tmp_path / "selection_preview.mp4",
        contact_sheet_path=tmp_path / "selection_contact_sheet.jpg",
        frames_dir=tmp_path / "refined_annotated",
        write_frames=False,
        preview_max_width=64,
    )

    assert Path(outputs["preview_video"]).is_file()
    assert Path(outputs["contact_sheet"]).is_file()
    assert not list((tmp_path / "refined_annotated").glob("frame_*_refined.jpg"))
    cap = cv2.VideoCapture(outputs["preview_video"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    cap.release()
