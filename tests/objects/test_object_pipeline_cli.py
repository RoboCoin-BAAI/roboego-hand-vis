import json
from pathlib import Path

import cv2
import numpy as np

from scripts.object.run_object_annotation import run_annotation


FRAME_SIZE = (96, 72)
INTRINSICS = {"fx": 50.0, "fy": 50.0, "cx": 48.0, "cy": 36.0}


def write_tiny_video(path: Path, *, yellow: bool = True, missing_yellow_idx: int | None = None) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, FRAME_SIZE)
    for idx in range(3):
        frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        if yellow and idx != missing_yellow_idx:
            frame[30:60, 24:64] = [20, 230, 255]
        writer.write(frame)
    writer.release()


def write_depth_frames(depth_dir: Path, *, missing_idx: int | None = None) -> None:
    depth_dir.mkdir()
    for idx in range(3):
        if idx == missing_idx:
            continue
        np.save(
            depth_dir / f"frame_{idx:06d}_foundation_depth_m.npy",
            np.ones((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.float32) * 0.5,
        )


def run_tiny_annotation(tmp_path: Path, *, missing_depth_idx: int | None = None, no_qwen: bool = True) -> Path:
    video = tmp_path / "video.mp4"
    write_tiny_video(video)
    depth_dir = tmp_path / "depth"
    write_depth_frames(depth_dir, missing_idx=missing_depth_idx)
    out = tmp_path / "out"
    run_annotation(
        video_path=video,
        depth_dir=depth_dir,
        output_dir=out,
        intrinsics=INTRINSICS,
        hand_tracks_path=None,
        hand_joints_npz=None,
        backend="color_yellow",
        object_id="yellow_object_0",
        num_frames=3,
        no_qwen=no_qwen,
        save_rrd=False,
    )
    return out


def test_run_annotation_offline_writes_detected_outputs(tmp_path: Path):
    out = run_tiny_annotation(tmp_path)

    tracks = json.loads((out / "object_tracks.json").read_text())
    frame = tracks["objects"]["yellow_object_0"]["frames"][0]
    assert frame["bbox_xywh"] is not None
    assert frame["crop_path"] is not None
    assert frame["mask_path"] is not None
    assert frame["depth_median_m"] == 0.5
    assert frame["depth_valid_ratio"] == 1.0
    assert (out / frame["crop_path"]).exists()
    assert (out / frame["mask_path"]).exists()

    contact = json.loads((out / "contact_annotations.json").read_text())
    assert contact["status"] == "hand_joints_npz_missing"
    assert {item["status"] for item in contact["frames"]} == {"hand_joints_npz_missing"}

    semantic = json.loads((out / "semantic_annotations.json").read_text())
    assert semantic["status"] == "skipped"


def test_run_annotation_records_missing_depth_and_continues(tmp_path: Path):
    out = run_tiny_annotation(tmp_path, missing_depth_idx=1)

    tracks = json.loads((out / "object_tracks.json").read_text())
    missing_frame = tracks["objects"]["yellow_object_0"]["frames"][1]
    assert missing_frame["depth_median_m"] is None
    assert missing_frame["depth_valid_ratio"] is None
    assert tracks["warnings"] == [{"frame_idx": 1, "status": "depth_file_missing"}]


def test_run_annotation_writes_api_failure_without_qwen_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    out = run_tiny_annotation(tmp_path, no_qwen=False)

    semantic = json.loads((out / "semantic_annotations.json").read_text())
    assert semantic["status"] == "api_failed"
    assert semantic["provider"] == "qwen"
    assert semantic["error_type"] == "RuntimeError"
    assert "DASHSCOPE_API_KEY" in semantic["error"]


def test_run_annotation_rerun_overlay_uses_gap_filled_observations(tmp_path: Path, monkeypatch):
    video = tmp_path / "video.mp4"
    write_tiny_video(video, missing_yellow_idx=1)
    depth_dir = tmp_path / "depth"
    write_depth_frames(depth_dir, missing_idx=1)
    out = tmp_path / "out"
    overlay_observations = []

    monkeypatch.setattr("scripts.object.run_object_annotation.rr.init", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.object.run_object_annotation.rr.save", lambda *args, **kwargs: None)
    time_calls = []
    monkeypatch.setattr(
        "scripts.object.run_object_annotation.rr.set_time",
        lambda timeline, *, sequence: time_calls.append((timeline, sequence)),
        raising=False,
    )
    monkeypatch.delattr("scripts.object.run_object_annotation.rr.set_time_sequence", raising=False)

    def record_overlay(image_rgb, obs, *, label, contact_text=None):
        overlay_observations.append(obs)
        return image_rgb

    monkeypatch.setattr("scripts.object.run_object_annotation.draw_object_bbox_overlay", record_overlay)
    monkeypatch.setattr("scripts.object.run_object_annotation.log_object_overlay", lambda *args, **kwargs: None)

    run_annotation(
        video_path=video,
        depth_dir=depth_dir,
        output_dir=out,
        intrinsics=INTRINSICS,
        hand_tracks_path=None,
        hand_joints_npz=None,
        backend="color_yellow",
        object_id="yellow_object_0",
        num_frames=3,
        no_qwen=True,
        save_rrd=True,
    )

    tracks = json.loads((out / "object_tracks.json").read_text())
    assert tracks["objects"]["yellow_object_0"]["frames"][1]["source"] == "interpolated_yellow_bbox"
    assert len(overlay_observations) == 3
    assert time_calls == [("frame", 0), ("frame", 1), ("frame", 2)]
    assert overlay_observations[1].source == "interpolated_yellow_bbox"
    assert overlay_observations[1].bbox_xywh is not None
