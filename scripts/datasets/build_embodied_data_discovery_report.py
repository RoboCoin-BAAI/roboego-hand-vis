#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.datasets.embodied_data_discovery import (
    DEFAULT_REPORTS_DIR,
    build_daily_prompt,
    find_previous_report,
    save_daily_prompt,
    save_daily_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the daily embodied-data discovery prompt, or save the "
            "finished report from stdin into the reports directory."
        )
    )
    parser.add_argument("--date", default=date.today().isoformat(), help="Report date in YYYY-MM-DD format.")
    parser.add_argument("--reports_dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--previous_report",
        type=Path,
        default=None,
        help="Optional report to compare against. Defaults to the newest dated report before --date.",
    )
    parser.add_argument(
        "--save_report_from_stdin",
        action="store_true",
        help="Read a completed Markdown report from stdin, then write <date>.md and latest.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_date = date.fromisoformat(args.date)
    reports_dir = args.reports_dir.expanduser()
    if args.save_report_from_stdin:
        report_text = sys.stdin.read()
        if not report_text.strip():
            raise SystemExit("No report text received on stdin.")
        report_path = save_daily_report(reports_dir, today=report_date, report_text=report_text)
        print(f"Wrote report: {report_path}")
        print(f"Updated latest: {reports_dir / 'latest.md'}")
        return

    previous_report = args.previous_report or find_previous_report(reports_dir, today=report_date)
    prompt_text = build_daily_prompt(today=report_date, previous_report=previous_report)
    prompt_path = save_daily_prompt(reports_dir, today=report_date, prompt_text=prompt_text)
    print(f"Wrote prompt: {prompt_path}")
    if previous_report:
        print(f"Previous report: {previous_report}")
    else:
        print("Previous report: none")


if __name__ == "__main__":
    main()
