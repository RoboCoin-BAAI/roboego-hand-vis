"""L1 downloads: explicit database mode or direct KS3 session paths."""

import argparse
import json
import signal
from pathlib import Path
import sys

from .downloader import DownloadError, KS3Downloader, session_source


def _terminate(signum, frame):
    # Unwind normal cleanup so the downloader stops its child before DB recovery.
    raise SystemExit(128 + signum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download L1 sessions from KS3 or the database queue")
    parser.add_argument("paths", nargs="*", help="Direct mode: KS3 session directory paths")
    parser.add_argument("--database", action="store_true", help="Claim pending l1data records using the configured database")
    parser.add_argument("--output-dir", type=Path, required=True, help="Shared download root")
    parser.add_argument("--limit", type=int, default=0, help="Database mode: maximum records, 0 = all pending")
    parser.add_argument("--jobs", type=int, default=8, help="Concurrent files per session")
    assets = Path(__file__).parent / "ks3"
    parser.add_argument("--ks3-tool", type=Path, default=assets / "ks3util-linux-amd64")
    parser.add_argument("--ks3-config", type=Path, default=assets / "ks3util-config")
    args = parser.parse_args()
    if args.database and args.paths:
        parser.error("Cannot supply KS3 paths in database mode")
    if not args.database and not args.paths:
        parser.error("Supply KS3 paths or use --database")
    if args.limit < 0 or (not args.database and args.limit):
        parser.error("--limit is nonnegative and available only in database mode")
    if args.database:
        from ..database.connection_config import database_url_or_error
        args.database_url = database_url_or_error(parser)
    try:
        for source in args.paths:
            session_source(source)
        downloader = KS3Downloader(args.output_dir, args.ks3_tool, args.ks3_config, args.jobs)
    except (ValueError, DownloadError, OSError) as exc:
        parser.error(str(exc))

    previous_handler = signal.signal(signal.SIGTERM, _terminate)
    try:
        records = _run(args, downloader)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    failed = sum(record["status"] == "failed" for record in records)
    print(json.dumps({"completed_count": len(records) - failed,
                      "failed_count": failed, "records": records}, ensure_ascii=False, indent=2))
    if failed:
        sys.exit(1)


def _run(args, downloader):
    if args.database:
        # Direct mode never imports SQLAlchemy, psycopg, or the database package.
        from ..database import DataService
        from .database_adapter import download_pending_l1

        service = DataService.from_url(args.database_url)
        try:
            records = download_pending_l1(service, downloader, args.limit)
        finally:
            service.database.close()
    else:
        records = []
        for source in args.paths:
            try:
                result = downloader.download(source)
                record = dict(ks3_path=source, status="completed",
                              local_path=result.relative_path, reused=result.reused)
            except DownloadError as exc:
                record = dict(ks3_path=source, status="failed", error=str(exc))
            records.append(record)

    return records


if __name__ == "__main__":
    main()
