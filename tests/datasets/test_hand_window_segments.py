import json
from pathlib import Path

import cv2
import numpy as np

from roboego_hand_vis.datasets.hand_window_segments import export_hand_window_segments


def write_video(path: Path, *, frame_count: int, fps: float = 5.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (16, 12))
    assert writer.isOpened()
    for idx in range(frame_count):
        writer.write(np.full((12, 16, 3), idx, dtype=np.uint8))
    writer.release()


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def test_export_hand_window_segments_cuts_rectified_videos_and_manifest(tmp_path: Path):
    segments_root = tmp_path / "segments"
    segment = segments_root / "seg_0000_t000006_000015_place_reach"
    write_video(segment / "rectified/videos/left_rectified.mp4", frame_count=20)
    write_video(segment / "rectified/videos/right_rectified.mp4", frame_count=20)
    (segment / "timestamps").mkdir(parents=True)
    (segment / "timestamps/left_timestamps.txt").write_text(
        "\n".join(f"{idx} {1000.0 + idx * 0.2:.1f}" for idx in range(20)) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "sequence_name": segment.name,
        "dataset_name": "wumei01",
        "root_dir": str(segment / "rectified"),
        "videos": {
            "left": {"path": str(segment / "rectified/videos/left_rectified.mp4"), "frame_count": 20},
            "right": {"path": str(segment / "rectified/videos/right_rectified.mp4"), "frame_count": 20},
        },
        "outputs": {"manifest_path": str(segment / "rectified/dataset_manifest/dataset_manifest.json")},
        "intrinsics": {
            "left": {"fx": 1.0, "fy": 1.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12},
            "right": {"fx": 1.0, "fy": 1.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12},
        },
        "extrinsics": {"baseline_m": 0.1},
        "camera_model": {"type": "pinhole", "is_rectified": True, "is_undistorted": True},
        "metadata": {},
    }
    manifest_path = segment / "rectified/dataset_manifest/dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    windows = tmp_path / "hand_windows.json"
    windows.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "handwin_0000",
                        "source_segment_id": "seg_0000",
                        "start": 8,
                        "end": 9,
                        "segment_frame_range": {"start": 5, "end": 12, "num_frames": 8},
                        "actions": ["hold"],
                        "objects": ["product"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = export_hand_window_segments(
        segments_root=segments_root,
        hand_windows_json=windows,
        overwrite=True,
        codec="mp4v",
    )

    window_dir = segment / "hand_windows/handwin_0000_t000008_000009"
    assert result["window_count"] == 1
    assert frame_count(window_dir / "videos/left_rectified.mp4") == 8
    assert frame_count(window_dir / "videos/right_rectified.mp4") == 8
    assert json.loads((window_dir / "annotation/window.json").read_text(encoding="utf-8"))["window_id"] == "handwin_0000"
    out_manifest = json.loads((window_dir / "dataset_manifest/dataset_manifest.json").read_text(encoding="utf-8"))
    assert out_manifest["sequence_name"] == "seg_0000_t000006_000015_place_reach_handwin_0000"
    assert out_manifest["root_dir"] == str(window_dir)
    assert out_manifest["videos"]["left"]["path"] == str(window_dir / "videos/left_rectified.mp4")
    assert out_manifest["metadata"]["hand_window"]["segment_frame_range"] == {"start": 5, "end": 12, "num_frames": 8}
    assert (window_dir / "timestamps/left_timestamps.txt").read_text(encoding="utf-8").splitlines()[0] == "0 1001.0"
