from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a ROS-free OpenVINS offline binary from a prepared manifest.")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--duration_s", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(manifest["output_dir"]) / "trajectory"
    command = [
        str(args.binary.expanduser().resolve()),
        "--config",
        str(Path(manifest["estimator_config_path"]).expanduser().resolve()),
        "--left_video",
        str(Path(manifest["videos"]["left"]["path"]).expanduser().resolve()),
        "--right_video",
        str(Path(manifest["videos"]["right"]["path"]).expanduser().resolve()),
        "--stereo_pairs",
        str(Path(manifest["stereo_pairs_path"]).expanduser().resolve()),
        "--imu",
        str(Path(manifest["imu_clean_path"]).expanduser().resolve()),
        "--output_dir",
        str(output_dir),
        "--start_s",
        str(args.start_s),
    ]
    if args.duration_s is not None:
        command.extend(["--duration_s", str(args.duration_s)])
    # A successful exit without fresh artifacts must not reuse an earlier trajectory.
    for name in ('run_report.json', 'trajectory.jsonl'):
        (output_dir / name).unlink(missing_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
