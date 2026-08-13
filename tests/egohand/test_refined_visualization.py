from pathlib import Path

from scripts.hand.run_hamer_ego_filter import _prepare_refined_visualization_output_dir


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
