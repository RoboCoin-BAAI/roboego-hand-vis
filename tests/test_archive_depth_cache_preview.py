from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from roboego_hand_vis.archive_depth_cache.compute import DepthArchiveConfig, archive_depth_cache
from roboego_hand_vis.archive_depth_cache.compute import build_preview_h264_command


def test_build_preview_h264_command_uses_player_friendly_h264_mp4() -> None:
    command = build_preview_h264_command(
        width=3840,
        height=1156,
        output_video=Path("depth_preview.mp4"),
        fps=6,
        ffmpeg_bin="ffmpeg",
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "3840x1156",
        "-pix_fmt",
        "bgr24",
        "-r",
        "6",
        "-i",
        "-",
        "-r",
        "6",
        "-vsync",
        "cfr",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "depth_preview.mp4",
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe unavailable")
def test_archive_depth_cache_writes_all_preview_frames_with_h264(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    output_dir = tmp_path / "out"
    depth_dir.mkdir()
    for frame_idx in range(3):
        depth_m = np.linspace(0.2, 2.0, 64 * 96, dtype=np.float32).reshape(64, 96)
        np.save(depth_dir / f"frame_{frame_idx:06d}_foundation_depth_m.npy", depth_m)

    archive_depth_cache(
        depth_dir=depth_dir,
        output_dir=output_dir,
        config=DepthArchiveConfig(
            write_png=True,
            write_avi=False,
            write_preview=True,
            write_hevc12=False,
            write_valid_mask=False,
            fps=6,
        ),
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,codec_tag_string,pix_fmt,nb_read_frames",
            "-of",
            "default=nw=1",
            str(output_dir / "depth_preview.mp4"),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert "codec_name=h264" in probe.stdout
    assert "codec_tag_string=avc1" in probe.stdout
    assert "pix_fmt=yuv420p" in probe.stdout
    assert "nb_read_frames=3" in probe.stdout
