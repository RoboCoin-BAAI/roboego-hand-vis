from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR.parent.parent
for path in (PROJECT_ROOT, STAGE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
