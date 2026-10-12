import json
from pathlib import Path

import pytest


def test_download_publishes_session_and_reuses_matching_receipt(tmp_path, ks3_tool):
    from roboego_hand_vis.download_data.downloader import KS3Downloader

    tool, config = ks3_tool
    downloader = KS3Downloader(tmp_path / "output", tool, config)
    result = downloader.download("ks3://bucket/data/session001")
    assert result.relative_path == "session001"
    assert result.path == tmp_path / "output/session001"
    assert (result.path / "videos/left.mkv").read_bytes() == b"video-data"
    assert result.reused is False
    again = downloader.download("ks3://bucket/data/session001/")
    assert again.reused is True


@pytest.mark.parametrize("mode", ["partial", "error", "empty"])
def test_unsuccessful_download_never_publishes_final_directory(tmp_path, ks3_tool, mode):
    from roboego_hand_vis.download_data.downloader import DownloadError, KS3Downloader

    tool, config = ks3_tool
    config.write_text(mode)
    downloader = KS3Downloader(tmp_path / "output", tool, config)
    with pytest.raises(DownloadError):
        downloader.download("ks3://bucket/data/session001")
    assert not (tmp_path / "output/session001").exists()


def test_existing_unverified_or_different_source_is_not_overwritten(tmp_path, ks3_tool):
    from roboego_hand_vis.download_data.downloader import DownloadError, KS3Downloader

    tool, config = ks3_tool
    downloader = KS3Downloader(tmp_path / "output", tool, config)
    result = downloader.download("ks3://bucket/data/session001")
    with pytest.raises(DownloadError):
        downloader.download("ks3://other-bucket/data/session001")
    (result.path / "videos/left.mkv").unlink()
    with pytest.raises(DownloadError):
        downloader.download("ks3://bucket/data/session001")


@pytest.mark.parametrize("source", ["local/file", "ks3://bucket", "ks3://bucket/../x", "ks3://bucket/data/."])
def test_invalid_prefixes_are_rejected_before_downloading(tmp_path, ks3_tool, source):
    from roboego_hand_vis.download_data.downloader import DownloadError, KS3Downloader

    tool, config = ks3_tool
    downloader = KS3Downloader(tmp_path / "output", tool, config)
    with pytest.raises(DownloadError):
        downloader.download(source)


def test_direct_cli_has_no_database_imports(tmp_path, ks3_tool):
    import subprocess
    import sys

    tool, config = ks3_tool
    script = '''
import builtins
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.startswith(('sqlalchemy', 'psycopg', 'roboego_hand_vis.database')):
        raise AssertionError('Direct downloads must not import database dependencies')
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from roboego_hand_vis.download_data.l1 import main
main()
'''
    command = [sys.executable, "-c", script, "--output-dir", str(tmp_path / "output"),
               "--ks3-tool", str(tool), "--ks3-config", str(config), "ks3://bucket/data/session001"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["completed_count"] == 1


def test_cli_rejects_paths_in_database_mode(tmp_path):
    import subprocess
    import sys

    result = subprocess.run([
        sys.executable, "-m", "roboego_hand_vis.download_data.l1", "--database",
        "--output-dir", str(tmp_path), "ks3://bucket/session",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "cannot" in result.stderr.lower()


def test_sigterm_stops_cli_and_its_download_subprocess(tmp_path, ks3_tool):
    import os
    import signal
    import subprocess
    import sys
    import time

    tool, config = ks3_tool
    config.write_text("sleep")
    command = [sys.executable, "-m", "roboego_hand_vis.download_data.l1",
               "--output-dir", str(tmp_path / "output"), "--ks3-tool", str(tool),
               "--ks3-config", str(config), "ks3://bucket/session001"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not config.with_suffix(".pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert config.with_suffix(".pid").exists()
        child_pid = int(config.with_suffix(".pid").read_text())
        process.terminate()
        process.communicate(timeout=10)
        assert process.returncode == 143
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        assert not (tmp_path / "output/session001").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
