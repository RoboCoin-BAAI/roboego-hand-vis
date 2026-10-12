"""Register KS3 records: python -m roboego_hand_vis.database.ingest."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit

from .data_service import DataService
from .model import get_table


def ingest_paths(service: DataService, table_name: str, paths: list[str]) -> list[dict]:
    """Validate all inputs, then independently commit each registered record.

    Invalid input causes no writes. A database failure after earlier commits may
    leave a partial batch; re-running is safe because paths are unique per table.
    No remote KS3 request is made; this records paths for subsequent downloads.
    """
    get_table(table_name)
    if not paths:
        raise ValueError("Provide at least one KS3 path")
    for path in paths:
        parsed = urlsplit(path)
        if (parsed.scheme != "ks3" or not parsed.netloc or not parsed.path.strip("/")
                or path != path.strip() or "\x00" in path):
            raise ValueError(f"Expected a full KS3 data path, got: {path!r}")

    records = []
    for path in paths:
        row, created = service.register_if_absent(table_name, path)
        records.append({
            "id": row["id"],
            "ks3_path": row["ks3_path"],
            "result": "created" if created else "existing",
            "download_status": row["download_status"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Register KS3 data without duplicate inserts")
    parser.add_argument("paths", nargs="*", help="Full ks3://bucket/path values")
    parser.add_argument("--table", type=str.lower, choices=["l1data", "l2data", "l3data"],
                        default="l1data", help="Target table (default: l1data)")
    parser.add_argument("--file", type=Path, help="UTF-8 text file: one KS3 path per line")
    args = parser.parse_args()
    paths = list(args.paths)
    if args.file:
        try:
            paths.extend(line.strip() for line in args.file.read_text(encoding="utf-8").splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
        except OSError as exc:
            parser.error(f"Unable to read input file: {exc}")
    if not paths:
        parser.error("Provide KS3 paths or --file")
    from .connection_config import database_url_or_error
    url = database_url_or_error(parser)
    service = DataService.from_url(url)
    try:
        try:
            records = ingest_paths(service, args.table, paths)
        except ValueError as exc:
            parser.error(str(exc))
        created = sum(row["result"] == "created" for row in records)
        print(json.dumps({
            "table": args.table,
            "created_count": created,
            "existing_count": len(records) - created,
            "records": records,
        }, ensure_ascii=False, indent=2))
    finally:
        service.database.close()


if __name__ == "__main__":
    main()
