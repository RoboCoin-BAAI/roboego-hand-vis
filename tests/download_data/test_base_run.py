import csv
from types import SimpleNamespace

import pytest

from roboego_hand_vis.download_data.base import run


def arguments(tmp_path):
    batch = tmp_path / "batch.csv"
    with batch.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["目标路径"])
        writer.writerows([["ks3://bucket/jianzhi/one/"], ["ks3://bucket/other/two/"]])
    tool = tmp_path / "ks3util"
    tool.touch()
    tool.chmod(0o755)
    config = tmp_path / "config"
    config.touch()
    return ["--batch-csv", str(batch), "--ks3-tool", str(tool),
            "--ks3-config", str(config), "--output-dir", str(tmp_path / "output")]


def test_jobs_paths_and_continue_after_failure(tmp_path, monkeypatch):
    calls = []

    def transfer(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=3 if len(calls) == 1 else 0)

    monkeypatch.setattr(run.subprocess, "run", transfer)
    assert run.main(arguments(tmp_path) + ["--jobs", "12"]) == 1
    assert len(calls) == 2
    for command, name in zip(calls, ["one", "two"]):
        assert "--jobs=12" in command
        assert command[command.index("-c") + 1] == str(tmp_path / "config")
        assert command[-1] == str(tmp_path / "output" / name)
        assert command[-2].endswith("/" + name + "/")


def test_invalid_jobs_does_not_start_download(tmp_path, monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("started download"))
    with pytest.raises(SystemExit) as error:
        run.main(arguments(tmp_path) + ["--jobs", "0"])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_old_entry_uses_shared_implementation():
    from roboego_hand_vis.download_data.L1 import run as old
    assert old.main is run.main
    assert old.read_batch is run.read_batch
