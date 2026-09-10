import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export.roboego_hand_parquet import discover_export_units, export_auto, export_session_hand_parquet


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_session(tmp_path: Path) -> Path:
    session = tmp_path / "session_0001_task"
    videos = session / "videos"
    videos.mkdir(parents=True)
    for name in ("left", "right", "bleft", "bright", "left_rectified", "right_rectified"):
        (videos / f"{name}.mp4").write_bytes(b"video")
    audio = session / "audio"
    audio.mkdir()
    (audio / "audio.wav").write_bytes(b"audio")
    archive = session / "depth/fast_foundation_archive"
    archive.mkdir(parents=True)
    (archive / "depth_hevc12.mkv").write_bytes(b"depth")
    (archive / "valid_mask_ffv1.mkv").write_bytes(b"mask")
    _write_json(
        archive / "depth_archive_meta.json",
        {
            "frame_count": 2,
            "hevc12": {"enabled": True, "codec": "libx265"},
            "valid_mask": {"enabled": True, "codec": "ffv1"},
        },
    )
    _write_json(
        session / "dataset_manifest/dataset_manifest.json",
        {
            "dataset_name": "wumei01",
            "sequence_name": "session_0001_task",
            "videos": {
                "left": {"path": str(videos / "left_rectified.mp4"), "fps": 30.0, "frame_count": 2},
                "right": {"path": str(videos / "right_rectified.mp4"), "fps": 30.0, "frame_count": 2},
            },
        },
    )
    joints = [[[float(j), float(j) + 0.1, float(j) + 0.2] for j in range(21)]]
    _write_json(
        session / "hand/final_hand_result/smoothed_joints21.json",
        {
            "coordinate_frame": "camera_opencv",
            "unit": "meter",
            "joint_names": [f"joint_{idx}" for idx in range(21)],
            "frames": [
                {
                    "frame_idx": 0,
                    "hands": {
                        "ego_left": {"valid": True, "joints21_camera_opencv": joints[0]},
                        "ego_right": {"valid": False, "reason": "missing_candidate"},
                    },
                },
                {
                    "frame_idx": 1,
                    "hands": {
                        "ego_left": {"valid": False, "reason": "missing_candidate"},
                        "ego_right": {"valid": True, "joints21_camera_opencv": joints[0]},
                    },
                },
            ],
        },
    )
    return session


def test_export_session_hand_parquet_records_window_global_frame_range(tmp_path: Path):
    session = _make_session(tmp_path)
    _write_json(
        session / "annotation/window.json",
        {
            "window_id": "handwin_0000",
            "semantic_label": "reach",
            "actions": ["reach"],
            "objects": ["product"],
            "source_session": "/data/parent_session",
            "source_start_frame": 240,
            "local_frame_range": {"start": 0, "end": 1, "num_frames": 2},
            "segment_frame_range": {"start": 240, "end": 241, "num_frames": 2},
            "core_segment_frame_range": {"start": 300, "end": 330, "num_frames": 31},
            "fps": 30.0,
        },
    )

    export_session_hand_parquet(session=session, output_root=session, episode_index=0)

    df = pd.read_parquet(session / "data/episode_000000.parquet")
    assert list(df["frame_index"]) == [0, 1]
    assert list(df["source_frame_index"]) == [240, 241]

    info = json.loads((session / "meta/info.json").read_text(encoding="utf-8"))
    assert info["window"] == {
        "window_id": "handwin_0000",
        "source_session": "parent_session",
        "source_frame_range": {"start": 240, "end": 241, "num_frames": 2},
        "local_frame_range": {"start": 0, "end": 1, "num_frames": 2},
    }


def test_export_auto_discovers_complete_hand_windows_from_parent_session(tmp_path: Path):
    parent = _make_session(tmp_path)
    window = parent / "hand_windows/handwin_0000_t000000_000001_reach"
    shutil.copytree(parent / "hand", window / "hand")
    shutil.copytree(parent / "videos", window / "videos")
    shutil.copytree(parent / "depth", window / "depth")
    shutil.copytree(parent / "dataset_manifest", window / "dataset_manifest")
    _write_json(
        window / "annotation/window.json",
        {
            "window_id": "handwin_0000",
            "semantic_label": "reach",
            "actions": ["reach"],
            "objects": ["product"],
            "source_session": str(parent),
            "source_start_frame": 10,
            "local_frame_range": {"start": 0, "end": 1, "num_frames": 2},
        },
    )

    units = discover_export_units(parent)
    results = export_auto(session=parent)

    assert units == [window.resolve()]
    assert results["mode"] == "hand_windows"
    assert results["export_count"] == 1
    assert (window / "data/episode_000000.parquet").is_file()
    info = json.loads((window / "meta/info.json").read_text(encoding="utf-8"))
    assert info["window"]["source_frame_range"] == {"start": 10, "end": 11, "num_frames": 2}
    assert info["window"]["source_session"] == "session_0001_task"


def test_discover_export_units_falls_back_to_session_when_no_complete_windows(tmp_path: Path):
    session = _make_session(tmp_path)
    incomplete = session / "hand_windows/handwin_0000"
    incomplete.mkdir(parents=True)

    assert discover_export_units(session) == [session.resolve()]


def test_export_window_uses_parent_audio_and_stage1_annotation(tmp_path: Path):
    parent = _make_session(tmp_path)
    window = parent / "hand_windows/handwin_0000_t000000_000001_reach"
    shutil.copytree(parent / "hand", window / "hand")
    shutil.copytree(parent / "videos", window / "videos")
    (window / "videos/bright.mp4").unlink()
    (window / "videos/bleft.mp4").unlink()
    shutil.copytree(parent / "depth", window / "depth")
    shutil.copytree(parent / "dataset_manifest", window / "dataset_manifest")
    _write_json(parent / "annotation/stage1_actions_left_v1_resolved.json", {"frames": []})
    _write_json(
        window / "annotation/window.json",
        {
            "window_id": "handwin_0000",
            "semantic_label": "reach",
            "actions": ["reach"],
            "objects": ["product"],
            "source_session": str(parent),
            "source_start_frame": 10,
            "local_frame_range": {"start": 0, "end": 1, "num_frames": 2},
        },
    )

    export_session_hand_parquet(session=window, output_root=window, episode_index=0)

    info = json.loads((window / "meta/info.json").read_text(encoding="utf-8"))
    assert info["source_session"] == "session_0001_task"
    assert info["audio"]["audio"]["path"] == "session_0001_task/audio/audio.wav"
    assert info["annotations"]["stage1_actions"] == "session_0001_task/annotation/stage1_actions_left_v1_resolved.json"
    assert "videos" not in info
    assert info["features"]["observation.images.bright"]["path"] == "session_0001_task/videos/bright.mp4"
    assert info["features"]["observation.images.left_rectified"]["path"] == (
        "session_0001_task/hand_windows/handwin_0000_t000000_000001_reach/videos/left_rectified.mp4"
    )


def test_export_session_hand_parquet_uses_annotation_task_label(tmp_path: Path):
    session = _make_session(tmp_path)
    _write_json(
        session / "annotation/hand_estimation_windows_left_v1.json",
        {
            "windows": [
                {
                    "actions": ["hold", "place", "reach"],
                    "objects": ["product"],
                    "semantic_label": "place_reach",
                }
            ]
        },
    )

    export_session_hand_parquet(session=session, output_root=session, episode_index=0)

    tasks = [json.loads(line) for line in (session / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    episodes = [json.loads(line) for line in (session / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert tasks == [{"task_index": 0, "task": "place_reach product: hold, place, reach"}]
    assert episodes[0]["tasks"] == ["place_reach product: hold, place, reach"]


def test_export_session_hand_parquet_writes_nan_invalid_hands(tmp_path: Path):
    session = _make_session(tmp_path)
    output = session

    result = export_session_hand_parquet(session=session, output_root=output, episode_index=0)

    parquet_path = Path(result["parquet_path"])
    assert parquet_path == session / "data/episode_000000.parquet"
    df = pd.read_parquet(parquet_path)
    assert list(df["frame_index"]) == [0, 1]
    assert list(df["observation.left_hand_joints_valid"]) == [True, False]
    assert list(df["observation.right_hand_joints_valid"]) == [False, True]
    assert df.loc[0, "observation.state.left.wrist.x"] == 0.0
    assert pd.isna(df.loc[0, "observation.state.right.wrist.x"])
    assert pd.isna(df.loc[1, "observation.state.left.wrist.x"])
    assert df.loc[1, "observation.state.right.wrist.x"] == 0.0


def test_export_session_hand_parquet_writes_external_media_metadata(tmp_path: Path):
    session = _make_session(tmp_path)
    output = session

    export_session_hand_parquet(session=session, output_root=output, episode_index=3)

    source_meta = json.loads((output / "meta/roboego_sources.json").read_text(encoding="utf-8"))
    assert "videos" not in source_meta
    assert source_meta["depth"]["hevc12"]["path"] == "session_0001_task/depth/fast_foundation_archive/depth_hevc12.mkv"
    assert source_meta["depth"]["hevc12"]["codec"] == "h265"
    assert source_meta["depth"]["hevc12"]["codec_name"] == "unknown"
    assert source_meta["depth"]["hevc12"]["codec_family"] == "h265/hevc"
    assert source_meta["depth"]["valid_mask"]["path"] == "session_0001_task/depth/fast_foundation_archive/valid_mask_ffv1.mkv"

    info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "roboego_hand_parquet_v1"
    assert info["features"]["observation.state"]["shape"] == [126]
    assert info["features"]["observation.images.left"]["dtype"] == "video"
    assert info["features"]["observation.images.left"]["path"] == "session_0001_task/videos/left.mp4"
    assert info["features"]["observation.images.left"]["info"]["video.codec"] == "unknown"
    assert "videos" not in info
    assert info["depth"]["hevc12"]["path"] == "session_0001_task/depth/fast_foundation_archive/depth_hevc12.mkv"
    assert info["depth"]["valid_mask"]["path"] == "session_0001_task/depth/fast_foundation_archive/valid_mask_ffv1.mkv"
    assert info["audio"]["audio"]["path"] == "session_0001_task/audio/audio.wav"

    episodes = [json.loads(line) for line in (output / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert episodes == [{"episode_index": 3, "length": 2, "tasks": ["session_0001_task"], "data_path": "data/episode_000003.parquet"}]
