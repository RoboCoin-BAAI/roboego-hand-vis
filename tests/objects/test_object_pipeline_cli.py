from pathlib import Path

import cv2
import numpy as np

from scripts.object.run_object_annotation import run_annotation


def write_tiny_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    for _ in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[28:40, 20:38] = [20, 230, 255]
        writer.write(frame)
    writer.release()


def test_run_annotation_offline_writes_core_outputs(tmp_path: Path):
    video = tmp_path / "video.mp4"
    write_tiny_video(video)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    for idx in range(3):
        np.save(
            depth_dir / f"frame_{idx:06d}_foundation_depth_m.npy",
            np.ones((48, 64), dtype=np.float32) * 0.5,
        )
    out = tmp_path / "out"
    run_annotation(
        video_path=video,
        depth_dir=depth_dir,
        output_dir=out,
        intrinsics={"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0},
        hand_tracks_path=None,
        hand_joints_npz=None,
        backend="color_yellow",
        object_id="yellow_object_0",
        num_frames=3,
        no_qwen=True,
        save_rrd=False,
    )
    assert (out / "object_tracks.json").exists()
    assert (out / "contact_annotations.json").exists()
    assert (out / "semantic_annotations.json").exists()
