import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from roboego_hand_vis.quality.video_segments import export_selected_video_segments


def write_numbered_video(path: Path, *, frame_count: int, fps: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 12))
    assert writer.isOpened()
    for idx in range(frame_count):
        frame = np.full((12, 16, 3), idx, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def video_codec_info(path: Path) -> dict[str, str]:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        text=True,
    )
    return dict(line.split("=", 1) for line in output.strip().splitlines())


def test_export_selected_video_segments_cuts_streams_and_writes_segment_metadata(tmp_path: Path):
    session = tmp_path / "session"
    videos = session / "videos"
    videos.mkdir(parents=True)
    write_numbered_video(videos / "left.mp4", frame_count=30)
    write_numbered_video(videos / "right.mp4", frame_count=30)
    (session / "meta.json").write_text(json.dumps({"fps": 5}), encoding="utf-8")
    plan = tmp_path / "processing_plan.json"
    plan.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "seg_0000",
                        "start": 1,
                        "end": 3,
                        "duration": 3,
                        "semantic_label": "pick_place",
                        "actions": ["pick", "place"],
                        "objects": ["product"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = export_selected_video_segments(
        session_dir=session,
        processing_plan_json=plan,
        output_dir=tmp_path / "selected",
        camera_streams=("left", "right"),
        overwrite=True,
    )

    segment_dir = tmp_path / "selected" / "seg_0000_t000001_000003_pick_place"
    assert summary["segment_count"] == 1
    assert segment_dir.exists()
    assert video_frame_count(segment_dir / "left.mp4") == 15
    assert video_frame_count(segment_dir / "right.mp4") == 15
    segment = json.loads((segment_dir / "segment.json").read_text(encoding="utf-8"))
    assert segment["start"] == 1
    assert segment["end"] == 3
    assert segment["source_start_frame"] == 5
    assert segment["segment_frame_count"] == 15
    assert segment["videos"]["left"].endswith("seg_0000_t000001_000003_pick_place/left.mp4")


def test_export_selected_video_segments_defaults_to_browser_compatible_h264(tmp_path: Path):
    session = tmp_path / "session"
    videos = session / "videos"
    videos.mkdir(parents=True)
    write_numbered_video(videos / "left.mp4", frame_count=10)
    (session / "meta.json").write_text(json.dumps({"fps": 5}), encoding="utf-8")
    plan = tmp_path / "processing_plan.json"
    plan.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "seg_0000",
                        "start": 0,
                        "end": 1,
                        "semantic_label": "pick",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    export_selected_video_segments(
        session_dir=session,
        processing_plan_json=plan,
        output_dir=tmp_path / "selected",
        camera_streams=("left",),
        overwrite=True,
    )

    info = video_codec_info(tmp_path / "selected" / "seg_0000_t000000_000001_pick" / "left.mp4")
    assert info["codec_name"] == "h264"
    assert info["codec_tag_string"] == "avc1"
    assert info["pix_fmt"] == "yuv420p"
