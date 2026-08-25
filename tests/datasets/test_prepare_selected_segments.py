import json
from pathlib import Path

import pytest

from scripts.datasets import prepare_selected_segments


def test_prepare_segment_reads_selected_videos_from_videos_directory(monkeypatch, tmp_path: Path):
    session = tmp_path / "session"
    (session / "calibrations").mkdir(parents=True)
    (session / "calibrations/stereo.json").write_text("{}", encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    segment = tmp_path / "selected/seg_0000_t000001_000003_pick"
    (segment / "videos").mkdir(parents=True)
    (segment / "videos/left.mp4").write_bytes(b"left")
    (segment / "videos/right.mp4").write_bytes(b"right")

    def fake_run_rectify(**kwargs):
        output = kwargs["rectify_output_root"] / kwargs["rectify_input"].name
        (output / "videos").mkdir(parents=True)
        (output / "videos/left.mp4").write_bytes(b"left")
        (output / "videos/right.mp4").write_bytes(b"right")
        (output / "videos/left_rectified.mp4").write_bytes(b"left-rectified")
        (output / "videos/right_rectified.mp4").write_bytes(b"right-rectified")
        (output / "calibrations").mkdir(parents=True)
        (output / "timestamps").mkdir(parents=True)
        (output / "meta.json").write_text("{}", encoding="utf-8")

    def fake_write_manifest(**kwargs):
        manifest = kwargs["manifest_dir"] / "dataset_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(prepare_selected_segments, "_run_rectify", fake_run_rectify)
    monkeypatch.setattr(prepare_selected_segments, "_write_manifest", fake_write_manifest)

    result = prepare_selected_segments.prepare_segment(
        segment_dir=segment,
        session_dir=session,
        roboegopipe_root=tmp_path / "RoboEgoPipe",
        dataset_adapter="wumei01",
        model="equidistant",
        balance=0.0,
        width=1600,
        height=900,
        overwrite=False,
        dry_run=False,
    )

    assert result["status"] == "ok"
    assert (segment / "rectified/videos/left_rectified.mp4").exists()
    assert not (segment / "rectified/videos/left.mp4").exists()
    assert not (segment / "rectified/videos/right.mp4").exists()
    assert (segment / "rectified/dataset_manifest/dataset_manifest.json").exists()
    assert not (segment / "rectified/_work").exists()
    assert result["rectified_dir"].endswith("/rectified")


def test_prepare_segment_rejects_legacy_root_level_segment_videos(tmp_path: Path):
    segment = tmp_path / "selected/seg_0000_t000001_000003_pick"
    segment.mkdir(parents=True)
    (segment / "left.mp4").write_bytes(b"left")
    (segment / "right.mp4").write_bytes(b"right")

    with pytest.raises(FileNotFoundError, match="videos/left.mp4"):
        prepare_selected_segments._validate_segment(segment)
