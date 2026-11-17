"""Download dataset directories listed in a single-column KS3 batch CSV."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


def read_batch(batch_csv: Path) -> list[str]:
    """Read and validate a single-column KS3 directory CSV."""
    with batch_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if not rows or rows[0] != ["目标路径"]:
        raise ValueError("CSV must have a single '目标路径' header")

    paths = []
    destinations = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != 1:
            raise ValueError(f"CSV row {row_number} must contain exactly one path")

        source = row[0].strip()
        parsed = urlsplit(source)
        parts = parsed.path.strip("/").split("/")
        if (parsed.scheme != "ks3" or not parsed.netloc or not parsed.path.strip("/")
                or parsed.query or parsed.fragment or "\\" in source or "\x00" in source
                or any(part in ("", ".", "..") for part in parts)):
            raise ValueError(f"Invalid KS3 path on CSV row {row_number}: {source}")
        source = source.rstrip("/") + "/"
        name = parts[-1]
        if name in destinations:
            raise ValueError(
                f"Duplicate session directory on CSV row {row_number}: {name} "
                f"({destinations[name]} and {source})"
            )
        destinations[name] = source
        paths.append(source)

    if not paths:
        raise ValueError("CSV contains no paths to download")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download all KS3 dataset directories listed in a batch CSV"
    )
    parser.add_argument("--batch-csv", required=True, type=Path, help="Batch CSV file")
    parser.add_argument("--ks3-tool", required=True, type=Path, help="Path to ks3util")
    parser.add_argument("--ks3-config", required=True, type=Path, help="Path to ks3util-config")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local download directory")
    parser.add_argument("--jobs", type=int, default=64,
                        help="Concurrent transfers inside ks3util (default: 64)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    batch_csv = args.batch_csv.expanduser()
    tool = args.ks3_tool.expanduser().resolve()
    config = args.ks3_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        paths = read_batch(batch_csv)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.error(str(exc))

    if not tool.is_file():
        parser.error(f"KS3 tool does not exist: {tool}")
    if not config.is_file():
        parser.error(f"KS3 configuration does not exist: {config}")
    if os.name != "nt" and not tool.stat().st_mode & 0o111:
        parser.error(f"KS3 tool is not executable: {tool}")

    output_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for index, source in enumerate(paths, start=1):
        # A trailing-slash source copies its contents into the destination.
        # Give each session its own directory to prevent merging batch records.
        destination = output_dir / urlsplit(source).path.rstrip("/").rsplit("/", 1)[-1]
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(tool),
            "-c", str(config),
            "-r",
            "-u",
            "--read-timeout", "6",
            f"--jobs={args.jobs}",
            "cp",
            source,
            str(destination),
        ]
        print(f"[{index}/{len(paths)}] Downloading {source} -> {destination}", flush=True)
        try:
            result = subprocess.run(command, check=False)
            reason = f"exit {result.returncode}" if result.returncode else None
        except OSError as exc:
            reason = str(exc)
        if reason:
            failed += 1
            print(
                f"Download failed ({reason}): {source}",
                file=sys.stderr,
                flush=True,
            )

    succeeded = len(paths) - failed
    print(f"Finished: {succeeded} succeeded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
