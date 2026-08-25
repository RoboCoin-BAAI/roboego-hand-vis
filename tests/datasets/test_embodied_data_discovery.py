import sys
import subprocess
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.embodied_data_discovery import (
    build_daily_prompt,
    find_previous_report,
    save_daily_report,
)


def test_find_previous_report_prefers_latest_before_today(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-23.md").write_text("older", encoding="utf-8")
    (reports / "2026-08-24.md").write_text("yesterday", encoding="utf-8")
    (reports / "latest.md").write_text("ignore alias", encoding="utf-8")
    (reports / "2026-08-25.prompt.md").write_text("ignore prompt", encoding="utf-8")

    previous = find_previous_report(reports, today=date(2026, 8, 25))

    assert previous == reports / "2026-08-24.md"


def test_build_daily_prompt_embeds_yesterday_report_and_required_sections(tmp_path: Path):
    yesterday = tmp_path / "2026-08-24.md"
    yesterday.write_text("# Yesterday\n- Old UMI dataset", encoding="utf-8")

    prompt = build_daily_prompt(today=date(2026, 8, 25), previous_report=yesterday)

    assert "Embodied Data Discovery Report - 2026-08-25" in prompt
    assert "<<<YESTERDAY_REPORT_START>>>" in prompt
    assert "# Yesterday\n- Old UMI dataset" in prompt
    assert "## New Since Yesterday" in prompt
    assert "Universal Manipulation Interface" in prompt
    assert "Name | Link | New/Updated | Real Robot | Ego | UMI" in prompt


def test_save_daily_report_writes_dated_report_and_latest_alias(tmp_path: Path):
    report = "# Embodied Data Discovery Report - 2026-08-25\n"

    report_path = save_daily_report(tmp_path, today=date(2026, 8, 25), report_text=report)

    assert report_path == tmp_path / "2026-08-25.md"
    assert report_path.read_text(encoding="utf-8") == report
    assert (tmp_path / "latest.md").read_text(encoding="utf-8") == report


def test_cli_writes_prompt_file(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "2026-08-24.md").write_text("# Yesterday", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/datasets/build_embodied_data_discovery_report.py"),
            "--date",
            "2026-08-25",
            "--reports_dir",
            str(reports),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    prompt_path = reports / "2026-08-25.prompt.md"
    assert str(prompt_path) in result.stdout
    assert "Embodied Data Discovery Report - 2026-08-25" in prompt_path.read_text(encoding="utf-8")


def test_cli_saves_report_from_stdin(tmp_path: Path):
    reports = tmp_path / "reports"
    report = "# Embodied Data Discovery Report - 2026-08-25\n"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/datasets/build_embodied_data_discovery_report.py"),
            "--date",
            "2026-08-25",
            "--reports_dir",
            str(reports),
            "--save_report_from_stdin",
        ],
        input=report,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert (reports / "2026-08-25.md").read_text(encoding="utf-8") == report
    assert (reports / "latest.md").read_text(encoding="utf-8") == report
