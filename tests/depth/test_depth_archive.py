from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from roboego_hand_vis.depth.archive import (
    DepthArchiveConfig,
    archive_depth_cache,
    build_ffv1_command,
    build_hevc12_command,
    build_valid_mask_command,
    depth_m_to_uint16_mm,
    depth_uint16_mm_to_preview_bgr,
)


def test_depth_m_to_uint16_mm_clips_and_marks_invalid_values():
    depth = np.array(
        [
            [0.1234, 2.5, 2.501],
            [np.nan, np.inf, -0.1],
        ],
        dtype=np.float32,
    )

    encoded = depth_m_to_uint16_mm(depth, max_depth_m=2.5, invalid_value=0)

    assert encoded.dtype == np.uint16
    assert encoded.tolist() == [
        [123, 2500, 0],
        [0, 0, 0],
    ]


def test_depth_uint16_mm_to_preview_bgr_maps_invalid_to_black():
    depth_mm = np.array([[0, 500], [1500, 2500]], dtype=np.uint16)

    preview = depth_uint16_mm_to_preview_bgr(depth_mm, max_depth_m=2.5, invalid_value=0)

    assert preview.dtype == np.uint8
    assert preview.shape == (2, 2, 3)
    assert preview[0, 0].tolist() == [0, 0, 0]
    assert preview[0, 1].tolist() != preview[1, 1].tolist()


def test_depth_preview_can_stretch_contrast_by_percentile():
    depth_mm = np.array([[0, 700, 710, 720], [730, 740, 750, 4000]], dtype=np.uint16)

    fixed = depth_uint16_mm_to_preview_bgr(
        depth_mm,
        max_depth_m=4.0,
        invalid_value=0,
        contrast_percentile=None,
        preview_mode="full",
    )
    stretched = depth_uint16_mm_to_preview_bgr(
        depth_mm,
        max_depth_m=4.0,
        invalid_value=0,
        contrast_percentile=2.0,
        preview_mode="full",
    )

    fixed_unique = np.unique(fixed.reshape(-1, 3), axis=0)
    stretched_unique = np.unique(stretched.reshape(-1, 3), axis=0)

    assert len(stretched_unique) > len(fixed_unique)


def test_depth_preview_near_mode_emphasizes_015_to_3m_and_clamps_far():
    depth_mm = np.array([[0, 150, 900], [1600, 3000, 4000]], dtype=np.uint16)

    preview = depth_uint16_mm_to_preview_bgr(
        depth_mm,
        max_depth_m=4.0,
        invalid_value=0,
        preview_mode="near",
        near_min_m=0.15,
        near_max_m=3.0,
    )

    assert preview[0, 0].tolist() == [0, 0, 0]
    assert preview[0, 1].tolist() != preview[0, 2].tolist()
    assert preview[0, 2].tolist() != preview[1, 0].tolist()
    assert preview[1, 1].tolist() == preview[1, 2].tolist()


def test_archive_depth_cache_writes_uint16_pngs_and_metadata(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "archive"
    depth_dir.mkdir()
    np.save(depth_dir / "frame_000000_foundation_depth_m.npy", np.array([[0.5, np.nan], [1.25, 2.6]], dtype=np.float32))
    np.save(depth_dir / "frame_000001_foundation_depth_m.npy", np.full((2, 2), 1.0, dtype=np.float32))

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(max_depth_m=2.5, write_avi=False, write_preview=False),
    )

    png0 = cv2.imread(str(output_dir / "depth_uint16_png" / "frame_000000_depth_mm.png"), cv2.IMREAD_UNCHANGED)
    assert png0.dtype == np.uint16
    assert png0.tolist() == [[500, 0], [1250, 0]]
    assert report["frame_count"] == 2
    metadata = json.loads((output_dir / "depth_archive_meta.json").read_text(encoding="utf-8"))
    assert metadata["unit"] == "millimeter"
    assert metadata["invalid_value"] == 0
    assert metadata["outputs"]["png_dir"] == str(output_dir / "depth_uint16_png")


def test_archive_depth_cache_writes_color_preview_video(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "archive"
    depth_dir.mkdir()
    np.save(depth_dir / "frame_000000_foundation_depth_m.npy", np.full((8, 10), 0.5, dtype=np.float32))
    np.save(depth_dir / "frame_000001_foundation_depth_m.npy", np.full((8, 10), 1.5, dtype=np.float32))

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(max_depth_m=2.5, write_avi=False, write_preview=True, fps=5),
    )

    preview_path = output_dir / "depth_preview.mp4"
    cap = cv2.VideoCapture(str(preview_path))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    cap.release()
    assert report["outputs"]["preview_video"] == str(preview_path)


def test_archive_depth_cache_accepts_existing_uint16_png_depth(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    output_dir = tmp_path / "archive"
    png_dir.mkdir(parents=True)
    cv2.imwrite(str(png_dir / "frame_000000_depth_mm.png"), np.full((2, 2), 1000, dtype=np.uint16))

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(write_png=False, write_avi=False, write_preview=False),
    )

    assert report["source_depth_format"] == "uint16_png"
    assert report["frame_count"] == 1
    assert report["frames"][0]["source_png"] == str(png_dir / "frame_000000_depth_mm.png")
    assert not (output_dir / "depth_uint16_png").exists()
    assert report["outputs"]["png_dir"] == str(png_dir)


def test_build_ffv1_command_uses_gray16le_png_sequence(tmp_path: Path):
    png_dir = tmp_path / "depth_uint16_png"
    output_avi = tmp_path / "depth_uint16_ffv1.avi"

    command = build_ffv1_command(png_dir=png_dir, output_avi=output_avi, fps=30.0)

    assert command == [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(png_dir / "frame_%06d_depth_mm.png"),
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray16le",
        str(output_avi),
    ]


def test_archive_depth_cache_can_write_ffv1_mkv_container(monkeypatch, tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    output_dir = tmp_path / "archive"
    png_dir.mkdir(parents=True)
    cv2.imwrite(str(png_dir / "frame_000000_depth_mm.png"), np.full((2, 2), 1000, dtype=np.uint16))
    calls = []

    def fake_run(args, check):
        calls.append((args, check))
        (output_dir / "depth_ffv1.mkv").write_bytes(b"ffv1")

    monkeypatch.setattr("roboego_hand_vis.depth.archive.subprocess.run", fake_run)

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(write_png=False, write_avi=True, write_preview=False, ffv1_container="mkv"),
    )

    assert calls[0][0][-1] == str(output_dir / "depth_ffv1.mkv")
    assert report["outputs"]["avi_path"] == ""
    assert report["outputs"]["ffv1_video"] == str(output_dir / "depth_ffv1.mkv")
    assert report["ffv1"] == {
        "enabled": True,
        "codec": "ffv1",
        "container": "mkv",
        "pixel_format": "gray16le",
        "bit_depth": 16,
        "depth_unit": "millimeter",
        "invalid_value": 0,
        "lossless": True,
        "source_frame_pattern": "depth_uint16_png/frame_%06d_depth_mm.png",
        "frame_count": 1,
        "fps": 30.0,
        "width": 2,
        "height": 2,
    }
    assert report["compression"]["ffv1_bytes"] == 4


def test_build_hevc12_command_uses_gray12le_x265_rawvideo_pipe(tmp_path: Path):
    output_video = tmp_path / "depth_hevc12.mkv"

    command = build_hevc12_command(
        width=10,
        height=8,
        output_video=output_video,
        fps=30.0,
        crf=0,
        preset="slow",
        lossless=True,
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "10x8",
        "-pix_fmt",
        "gray12le",
        "-r",
        "30",
        "-i",
        "-",
        "-r",
        "30",
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        "90000",
        "-c:v",
        "libx265",
        "-crf",
        "0",
        "-preset",
        "slow",
        "-pix_fmt",
        "gray12le",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-x265-params",
        "lossless=1:bframes=0",
        str(output_video),
    ]


def test_archive_depth_cache_rejects_hevc12_when_max_depth_exceeds_12bit(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "archive"
    depth_dir.mkdir()
    np.save(depth_dir / "frame_000000_foundation_depth_m.npy", np.full((2, 2), 1.0, dtype=np.float32))

    with pytest.raises(ValueError, match="HEVC12 direct millimeter encoding requires max_depth_m <= 4.095"):
        archive_depth_cache(
            depth_dir=depth_dir,
            output_dir=output_dir,
            config=DepthArchiveConfig(write_avi=False, write_preview=False, write_hevc12=True, max_depth_m=4.2),
        )


def test_archive_depth_cache_writes_hevc12_from_uint16_png_and_metadata(monkeypatch, tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    output_dir = tmp_path / "archive"
    png_dir.mkdir(parents=True)
    cv2.imwrite(str(png_dir / "frame_000000_depth_mm.png"), np.array([[0, 500], [4000, 5000]], dtype=np.uint16))

    calls = []

    class FakeStdin:
        def __init__(self):
            self.data = bytearray()
            self.closed = False

        def write(self, data):
            self.data.extend(data)

        def close(self):
            self.closed = True

    class FakePopen:
        def __init__(self, args, stdin, stdout, stderr):
            self.args = args
            self.stdin = FakeStdin()
            calls.append(self)

        def wait(self):
            (output_dir / "depth_hevc12.mkv").write_bytes(b"hevc12")
            return 0

    monkeypatch.setattr("roboego_hand_vis.depth.archive.subprocess.Popen", FakePopen)

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(write_png=False, write_avi=False, write_preview=False, write_hevc12=True),
    )

    assert len(calls) == 1
    written = np.frombuffer(calls[0].stdin.data, dtype=np.uint16).reshape(2, 2)
    assert written.tolist() == [[0, 500], [4000, 4000]]
    assert calls[0].stdin.closed is True
    assert "gray12le" in calls[0].args
    assert report["outputs"]["hevc12_video"] == str(output_dir / "depth_hevc12.mkv")
    assert report["hevc12"]["bit_depth"] == 12
    assert report["hevc12"]["valid_range_mm"] == [1, 4000]
    metadata = json.loads((output_dir / "depth_archive_meta.json").read_text(encoding="utf-8"))
    assert metadata["outputs"]["hevc12_video"] == str(output_dir / "depth_hevc12.mkv")


def test_archive_depth_cache_writes_lossless_valid_mask_for_hevc12(monkeypatch, tmp_path: Path):
    depth_dir = tmp_path / "depth"
    png_dir = depth_dir / "depth_uint16_png"
    output_dir = tmp_path / "archive"
    png_dir.mkdir(parents=True)
    cv2.imwrite(str(png_dir / "frame_000000_depth_mm.png"), np.array([[0, 500], [4000, 0]], dtype=np.uint16))

    calls = []

    class FakeStdin:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)

        def close(self):
            pass

    class FakePopen:
        def __init__(self, args, stdin, stdout, stderr):
            self.args = args
            self.stdin = FakeStdin()
            calls.append(self)

        def wait(self):
            output_path = Path(self.args[-1])
            output_path.write_bytes(output_path.name.encode("utf-8"))
            return 0

    monkeypatch.setattr("roboego_hand_vis.depth.archive.subprocess.Popen", FakePopen)

    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(
            write_png=False,
            write_avi=False,
            write_preview=False,
            write_hevc12=True,
            hevc12_lossless=False,
            write_valid_mask=True,
        ),
    )

    assert [Path(call.args[-1]).name for call in calls] == ["depth_hevc12.mkv", "valid_mask_ffv1.mkv"]
    mask = np.frombuffer(calls[1].stdin.data, dtype=np.uint8).reshape(2, 2)
    assert mask.tolist() == [[0, 255], [255, 0]]
    assert report["outputs"]["valid_mask_video"] == str(output_dir / "valid_mask_ffv1.mkv")
    assert report["hevc12"]["requires_valid_mask"] is True
    assert report["valid_mask"] == {
        "enabled": True,
        "codec": "ffv1",
        "container": "mkv",
        "pixel_format": "gray",
        "bit_depth": 8,
        "lossless": True,
        "valid_value": 255,
        "invalid_value": 0,
        "restore_rule": "depth_mm[valid_mask == 0] = 0",
        "frame_count": 1,
        "fps": 30.0,
        "width": 2,
        "height": 2,
    }


def test_build_valid_mask_command_uses_gray_ffv1_mkv(tmp_path: Path):
    output_video = tmp_path / "valid_mask_ffv1.mkv"

    command = build_valid_mask_command(width=10, height=8, output_video=output_video, fps=30.0)

    assert command == [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "10x8",
        "-pix_fmt",
        "gray",
        "-r",
        "30",
        "-i",
        "-",
        "-r",
        "30",
        "-fps_mode",
        "cfr",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "gray",
        str(output_video),
    ]


def test_archive_depth_cache_can_dry_run_ffmpeg(monkeypatch, tmp_path: Path):
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "archive"
    depth_dir.mkdir()
    np.save(depth_dir / "frame_000000_foundation_depth_m.npy", np.full((2, 2), 1.0, dtype=np.float32))
    calls = []

    def fake_run(args, check):
        calls.append((args, check))

    monkeypatch.setattr("roboego_hand_vis.depth.archive.subprocess.run", fake_run)
    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(write_avi=True, dry_run_avi=True, write_preview=False),
    )

    assert calls == []
    assert report["outputs"]["avi_path"].endswith("depth_uint16_ffv1.avi")
    assert report["ffmpeg_command"][0] == "ffmpeg"


def test_archive_depth_cache_deletes_npy_and_png_when_requested(monkeypatch, tmp_path: Path):
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "archive"
    depth_dir.mkdir()
    npy_path = depth_dir / "frame_000000_foundation_depth_m.npy"
    np.save(npy_path, np.full((2, 2), 1.0, dtype=np.float32))

    def fake_run(args, check):
        (output_dir / "depth_uint16_ffv1.avi").write_bytes(b"avi")

    monkeypatch.setattr("roboego_hand_vis.depth.archive.subprocess.run", fake_run)
    report = archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(
            write_avi=True,
            write_preview=False,
            delete_npy=True,
            delete_png=True,
        ),
    )

    assert not npy_path.exists()
    assert not (output_dir / "depth_uint16_png").exists()
    assert report["deleted_source_npy"] is True
    assert report["deleted_uint16_png"] is True


def test_archive_depth_cache_cli_passes_arguments(monkeypatch, tmp_path: Path, capsys):
    calls = []

    def fake_archive_depth_cache(*, depth_dir, output_dir, config):
        calls.append((depth_dir, output_dir, config))
        return {"stage": "depth_archive_v1", "frames": [{"frame_idx": 0}], "frame_count": 1}

    monkeypatch.setattr("roboego_hand_vis.depth.archive.archive_depth_cache", fake_archive_depth_cache)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "archive_depth_cache.py",
            "--depth_dir",
            str(tmp_path / "depth"),
            "--output_dir",
            str(tmp_path / "archive"),
            "--max_depth_m",
            "2.5",
            "--fps",
            "30",
            "--dry_run_avi",
            "--delete_png",
        ],
    )

    runpy.run_path("scripts/depth/archive_depth_cache.py", run_name="__main__")

    assert calls[0][0] == tmp_path / "depth"
    assert calls[0][1] == tmp_path / "archive"
    assert calls[0][2].max_depth_m == 2.5
    assert calls[0][2].dry_run_avi is True
    assert calls[0][2].delete_png is True
    assert '"frames"' not in capsys.readouterr().out
