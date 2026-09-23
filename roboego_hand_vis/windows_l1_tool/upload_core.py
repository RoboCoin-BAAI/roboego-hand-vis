from __future__ import annotations

import argparse
import getpass
import io
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import requests


MANIFEST_FILENAME = "egol1_upload_manifest.jsonl"
DEFAULT_PROJECT_ID = 252
DEFAULT_AK = "2Tlr5raEZMm"
DEFAULT_SK = "6FLD5DKwi2At/MjgUYWrbw=="


@dataclass(frozen=True)
class UploadCounts:
    success: int | None = None
    failed: int | None = None


@dataclass(frozen=True)
class UploadResult:
    returncode: int
    stdout: str
    stderr: str
    counts: UploadCounts


class TeeIO:
    def __init__(self, real_out):
        self._real = real_out
        self._buffer = io.StringIO()

    def write(self, value):
        self._buffer.write(value)
        self._real.write(value)
        self._real.flush()

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buffer.getvalue()


class FakeStdinAutoReply:
    def __init__(self, reply: str):
        self._reply = reply

    def readline(self):
        return self._reply + "\n"


def load_uploaded_folders(manifest_path: Path) -> set[str]:
    uploaded = set()
    if not manifest_path.exists():
        return uploaded
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("success") is True:
                uploaded.add(record["local_dir"])
    return uploaded


def append_manifest_record(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_files(directory: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in Path(directory).rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def parse_upload_counts(output: str) -> UploadCounts:
    success_match = re.search(r"成功:\s*(\d+)\s*个文件", output)
    failed_match = re.search(r"失败:\s*(\d+)\s*个文件", output)
    return UploadCounts(
        success=int(success_match.group(1)) if success_match else None,
        failed=int(failed_match.group(1)) if failed_match else None,
    )


def build_target_path(local_folder: Path) -> str:
    return local_folder.name.strip().strip("/\\")


def find_all_cp_cs_folders(root_dir: Path) -> list[Path]:
    pattern = re.compile(r"^(cp_|cs_)\d+")
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    return [item.resolve() for item in root_dir.rglob("*") if item.is_dir() and pattern.match(item.name)]


def run_robot_uploader(
    *,
    ak: str,
    sk: str,
    local_dir: Path,
    target_path: str,
    project_id: int,
    task_id: int | None = None,
    output_callback: Callable[[str], None] | None = None,
) -> UploadResult:
    import robot_data_uploader.uploader

    command = [
        "--ak",
        ak,
        "--sk",
        sk,
        "-d",
        str(local_dir),
        "-s",
        target_path,
        "--project-id",
        str(project_id),
    ]
    if task_id is not None:
        command.extend(["--task-id", str(task_id)])

    old_argv = sys.argv.copy()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin
    tee = TeeIO(sys.stdout)
    fake_stdin = FakeStdinAutoReply("2")

    try:
        sys.argv = ["robot-uploader"] + command
        sys.stdout = tee
        sys.stderr = tee
        sys.stdin = fake_stdin
        try:
            robot_data_uploader.uploader.main()
            ret_code = 0
        except SystemExit as exc:
            ret_code = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin

    combined_output = tee.getvalue()
    return UploadResult(
        returncode=ret_code,
        stdout=combined_output,
        stderr="",
        counts=parse_upload_counts(combined_output),
    )


def upload_folder(
    local_dir: Path,
    ak: str,
    sk: str,
    project_id: int,
    manifest_path: Path,
    dry_run: bool = False,
    task_id: int | None = None,
    output_callback: Callable[[str], None] | None = None,
) -> bool:
    def emit(message: str, end: str = "\n") -> None:
        text = f"{message}{end}"
        if output_callback is None:
            print(message, end=end, flush=True)
        else:
            output_callback(text)

    local_dir = local_dir.resolve()
    started_at = datetime.now().isoformat(timespec="seconds")
    target_path = build_target_path(local_dir)
    file_count, total_bytes = count_files(local_dir)
    emit(f"[INFO] cp_/cs_ Session folder: {local_dir}")
    emit(f"[INFO] Remote target path: {target_path}")
    emit(f"[INFO] Total files: {file_count}, size: {total_bytes} bytes")

    if dry_run:
        append_manifest_record(
            manifest_path,
            {
                "local_dir": str(local_dir),
                "target_path": target_path,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "success": True,
                "dry_run": True,
            },
        )
        emit("[DRY-RUN] Skip actual upload")
        return True

    try:
        result = run_robot_uploader(
            ak=ak,
            sk=sk,
            local_dir=local_dir,
            target_path=target_path,
            project_id=project_id,
            task_id=task_id,
            output_callback=output_callback,
        )
    except requests.exceptions.ConnectionError as exc:
        emit(f"[ERROR] KS3网络连接异常，远端断开连接: {exc}")
        result = UploadResult(returncode=-1, stdout="", stderr=str(exc), counts=UploadCounts())
    except Exception as exc:
        emit(f"[ERROR] 上传器发生未知异常: {exc}")
        result = UploadResult(returncode=-1, stdout="", stderr=str(exc), counts=UploadCounts())

    upload_ok = result.returncode == 0 and result.counts.failed in (0, None)
    append_manifest_record(
        manifest_path,
        {
            "local_dir": str(local_dir),
            "target_path": target_path,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "returncode": result.returncode,
            "upload_counts": asdict(result.counts),
            "success": upload_ok,
            "dry_run": False,
        },
    )
    emit("[SUCCESS] Upload finished, marked in manifest, will skip next time." if upload_ok else "[FAILED] Upload failed, will retry on next scan.")
    return upload_ok


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGOL1 cp_/cs_ session auto uploader.")
    parser.add_argument("--root", action="append", type=Path, help="Scan root folder, repeatable.")
    parser.add_argument("--ask-root", action="store_true", help="Interactively input scan root folders.")
    parser.add_argument("--ak", default=DEFAULT_AK, help="Access Key")
    parser.add_argument("--sk", default=DEFAULT_SK, help="Secret Key")
    parser.add_argument("--task-id", type=int, help="Task ID")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID, help="Project ID, default 252 EGOL1")
    parser.add_argument("--manifest", type=Path, help="Manifest file path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only, no real upload")
    parser.add_argument("--watch-interval", type=int, help="Scan every N seconds, no value for single run")
    return parser.parse_args(argv)


def resolve_scan_roots(args: argparse.Namespace) -> list[Path]:
    if args.root:
        return [Path(root) for root in args.root]
    if args.ask_root:
        folders = []
        print("Input scan root path, press enter on empty line to continue.")
        while True:
            value = input("Scan root: ").strip().strip('"')
            if not value:
                break
            folders.append(Path(value))
        return folders
    return []


def prompt_credentials(args: argparse.Namespace) -> tuple[str, str]:
    ak = args.ak
    sk = args.sk
    if args.ask_root:
        ak = input(f"Access Key [default {ak}]: ").strip() or ak
        sk = getpass.getpass("Secret Key [default]: ").strip() or sk
    if not ak or not sk:
        raise ValueError("AK and SK required.")
    return ak, sk


def scan_and_upload_once(args: argparse.Namespace) -> int:
    manifest_path = args.manifest or Path(MANIFEST_FILENAME)
    uploaded_set = load_uploaded_folders(manifest_path)
    scan_roots = resolve_scan_roots(args)
    if not scan_roots:
        print("[INFO] No scan root folders input.")
        return 0

    ak, sk = prompt_credentials(args)
    fail_count = 0
    sessions: list[Path] = []
    for scan_root in scan_roots:
        print(f"\n[SCAN] Search cp_/cs_ folders under: {scan_root}")
        found = find_all_cp_cs_folders(scan_root)
        print(f"[SCAN] Found {len(found)} cp_/cs_ session folders")
        sessions.extend(found)

    sessions = list(dict.fromkeys(sessions))
    print(f"\n[INFO] Total unique cp_/cs_ sessions found: {len(sessions)}")
    for target_dir in sessions:
        target_dir = target_dir.resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"[WARN] cp_/cs_ folder not exist: {target_dir}, skip")
            fail_count += 1
            continue
        if str(target_dir) in uploaded_set:
            print(f"[INFO] Already uploaded, skip cp_/cs_ session: {target_dir.name}")
            continue
        print(f"\n[DETECT] cp_/cs_ session detected, ready to process -> {target_dir.name}")
        ok = upload_folder(
            local_dir=target_dir,
            ak=ak,
            sk=sk,
            project_id=args.project_id,
            manifest_path=manifest_path,
            dry_run=args.dry_run,
            task_id=args.task_id,
        )
        if not ok:
            fail_count += 1
    return 1 if fail_count > 0 else 0


def upload_main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.watch_interval is None:
            return scan_and_upload_once(args)
        while True:
            scan_and_upload_once(args)
            print(f"\n[INFO] Sleep {args.watch_interval}s for next scan")
            time.sleep(args.watch_interval)
    except KeyboardInterrupt:
        print("\n[INFO] User exit.")
        return 0
