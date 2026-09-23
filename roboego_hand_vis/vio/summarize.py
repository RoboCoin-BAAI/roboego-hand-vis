from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .timing import summarize_openvins_timing
except ImportError:
    from timing import summarize_openvins_timing


def main() -> None:
    parser = argparse.ArgumentParser(description="Add estimator timing statistics to an OpenVINS run report.")
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report["openvins_timing"] = summarize_openvins_timing(args.timing)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["openvins_timing"], indent=2))
