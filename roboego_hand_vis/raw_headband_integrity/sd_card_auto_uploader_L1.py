r"""EGOL1 folder uploader, project id=252
No file validation, no meta.json.
Auto scan: input any parent path, find all folders starting with cp_ / cs_ followed by digits as upload unit.
Each cp_0001-xxx / cs_0001-xxx folder is a single upload task.
Cloud target path = basename of cp_/cs_ folder.
Upload success mark the cp_/cs_ folder in manifest, skip on next scan.
Example:
Input root: data\baai_ego_task
Program auto find: data\baai_ego_task\cp_0001_sessionxxxxxx
Remote path: cp_0001_sessionxxxxxx
"""
from __future__ import annotations

import argparse
import getpass
import io
import json
import os
import re
import sys
import time
import requests
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


MANIFEST_FILENAME = "egol1_upload_manifest.jsonl"
DEFAULT_PROJECT_ID = 252  # 修复：EGOL1项目id为252
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

    def write(self, s):
        self._buffer.write(s)
        self._real.write(s)
        self._real.flush()

    def flush(self):
        self._real.flush()

    def getvalue(self):
        return self._buffer.getvalue()


class FakeStdinAutoReply:
    """Auto reply input, select 2 when dataset name conflict"""
    def __init__(self, reply: str):
        self._reply = reply

    def readline(self):
        return self._reply + "\n"


def load_uploaded_folders(manifest_path: Path) -> set[str]:
    """Load already successfully uploaded cp_/cs_ absolute folder paths"""
    uploaded = set()
    if not manifest_path.exists():
        return uploaded
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("success") is True:
                uploaded.add(record["local_dir"])
    return uploaded


def append_manifest_record(manifest_path: Path, record: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_files(directory: Path):
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
    # 云端路径 = cp_/cs_文件夹名称
    folder_name = local_folder.name.strip().strip("/\\")
    return folder_name


def find_all_cp_cs_folders(root_dir: Path) -> list[Path]:
    """递归扫描root目录，找出所有名称以 cp_ / cs_ 开头且下划线后紧跟数字的文件夹
    匹配示例：cp_0001_sessionxxxxxx, cs_0002_test
    不匹配：cpabc, cp-0001, cp_abc_session
    """
    pattern = re.compile(r'^(cp_|cs_)\d+')
    cp_cs_folders = []
    if not root_dir.exists() or not root_dir.is_dir():
        return cp_cs_folders
    for item in root_dir.rglob("*"):
        if item.is_dir() and pattern.match(item.name):
            cp_cs_folders.append(item.resolve())
    return cp_cs_folders


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
        "--ak", ak,
        "--sk", sk,
        "-d", str(local_dir),
        "-s", target_path,
        "--project-id", str(project_id),
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
        except SystemExit as e:
            ret_code = e.code if isinstance(e.code, int) else 1
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
    def emit(message: str, end: str = "\n"):
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
        record = {
            "local_dir": str(local_dir),
            "target_path": target_path,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "success": True,
            "dry_run": True,
        }
        append_manifest_record(manifest_path, record)
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
    except requests.exceptions.ConnectionError as e:
        emit(f"[ERROR] KS3网络连接异常，远端断开连接: {e}")
        result = UploadResult(returncode=-1, stdout="", stderr=str(e), counts=UploadCounts())
    except Exception as e:
        emit(f"[ERROR] 上传器发生未知异常: {e}")
        result = UploadResult(returncode=-1, stdout="", stderr=str(e), counts=UploadCounts())

    upload_ok = (result.returncode == 0 and result.counts.failed in (0, None))
    finished_at = datetime.now().isoformat(timespec="seconds")

    record = {
        "local_dir": str(local_dir),
        "target_path": target_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": result.returncode,
        "upload_counts": asdict(result.counts),
        "success": upload_ok,
        "dry_run": False,
    }
    append_manifest_record(manifest_path, record)

    if upload_ok:
        emit("[SUCCESS] Upload finished, marked in manifest, will skip next time.")
    else:
        emit("[FAILED] Upload failed, will retry on next scan.")
    return upload_ok


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EGOL1 cp_/cs_ session auto uploader. Auto find folders like cp_0001_sessionxxx.")
    parser.add_argument("--root", action="append", type=Path, help="Scan root folder, repeatable. Input any parent path.")
    parser.add_argument("--ask-root", action="store_true", help="Interactively input scan root folders.")
    parser.add_argument("--ak", default=DEFAULT_AK, help="Access Key")
    parser.add_argument("--sk", default=DEFAULT_SK, help="Secret Key")
    parser.add_argument("--task-id", type=int, help="Task ID")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID, help="Project ID, default 252 EGOL1")
    parser.add_argument("--manifest", type=Path, help="Manifest file path")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only, no real upload")
    parser.add_argument("--watch-interval", type=int, help="Scan every N seconds, no value for single run")
    return parser.parse_args(argv)


def _resolve_scan_roots(args: argparse.Namespace) -> list[Path]:
    if args.root:
        return [Path(r) for r in args.root]
    if args.ask_root:
        folders = []
        print("Input scan root path (any parent path), press enter on empty line to continue.")
        while True:
            val = input("Scan root: ").strip().strip('"')
            if not val:
                break
            folders.append(Path(val))
        return folders
    return []


def _prompt_credentials(args: argparse.Namespace):
    ak = args.ak
    sk = args.sk

    if args.ask_root:
        ak = input(f"Access Key [default {ak}]: ").strip() or ak
        sk = getpass.getpass(f"Secret Key [default]: ").strip() or sk

    if not ak or not sk:
        raise ValueError("AK and SK required.")
    return ak, sk


def scan_and_upload_once(args: argparse.Namespace) -> int:
    manifest_path = args.manifest or Path(MANIFEST_FILENAME)
    uploaded_set = load_uploaded_folders(manifest_path)
    scan_roots = _resolve_scan_roots(args)
    if not scan_roots:
        print("[INFO] No scan root folders input.")
        return 0

    ak, sk = _prompt_credentials(args)
    fail_count = 0
    all_cp_cs_sessions: list[Path] = []

    # 遍历所有输入根目录，收集全部cp_/cs_会话文件夹
    for scan_root in scan_roots:
        print(f"\n[SCAN] Search cp_/cs_ folders under: {scan_root}")
        cp_cs_list = find_all_cp_cs_folders(scan_root)
        print(f"[SCAN] Found {len(cp_cs_list)} cp_/cs_ session folders")
        all_cp_cs_sessions.extend(cp_cs_list)

    # 去重，防止多个scan_root重复命中同一个目录
    all_cp_cs_sessions = list(dict.fromkeys(all_cp_cs_sessions))
    print(f"\n[INFO] Total unique cp_/cs_ sessions found: {len(all_cp_cs_sessions)}")

    for target_dir in all_cp_cs_sessions:
        target_dir = target_dir.resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"[WARN] cp_/cs_ folder not exist: {target_dir}, skip")
            fail_count += 1
            continue
        if str(target_dir) in uploaded_set:
            print(f"[INFO] Already uploaded, skip cp_/cs_ session: {target_dir.name}")
            continue

        # ========== 上传前输出检测 ==========
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


def main(argv: Sequence[str] | None = None) -> int:
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


if __name__ == "__main__":
    sys.exit(main())