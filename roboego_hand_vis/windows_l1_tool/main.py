from __future__ import annotations

import getpass
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent

try:
    from roboego_hand_vis.windows_l1_tool.batch_quality import batch_quality_check, discover_cp_cs_sessions
    from roboego_hand_vis.windows_l1_tool.upload_core import (
        DEFAULT_AK,
        DEFAULT_PROJECT_ID,
        DEFAULT_SK,
        parse_args as parse_upload_args,
        scan_and_upload_once,
    )
    from roboego_hand_vis.windows_l1_tool.upload_orchestrator import upload_quality_passed_sessions
except ModuleNotFoundError:
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    from batch_quality import batch_quality_check, discover_cp_cs_sessions  # noqa: E402
    from upload_core import (  # noqa: E402
        DEFAULT_AK,
        DEFAULT_PROJECT_ID,
        DEFAULT_SK,
        parse_args as parse_upload_args,
        scan_and_upload_once,
    )
    from upload_orchestrator import upload_quality_passed_sessions  # noqa: E402


def _prompt_roots() -> list[Path]:
    roots: list[Path] = []
    print("请输入扫描根目录，可输入多个。空行结束。")
    while True:
        value = input("扫描根目录: ").strip().strip('"')
        if not value:
            break
        roots.append(Path(value))
    return roots


def _prompt_upload_credentials() -> tuple[str, str]:
    print("请输入上传认证信息。直接回车使用默认值。")
    ak = input(f"AK [{DEFAULT_AK}]: ").strip() or DEFAULT_AK
    sk = getpass.getpass("SK [默认值已内置，直接回车使用默认值]: ").strip() or DEFAULT_SK
    return ak, sk


def _build_upload_args(roots: list[Path], *, ak: str, sk: str, project_id: int):
    return parse_upload_args(
        [
            *(item for root in roots for item in ("--root", str(root))),
            "--ak",
            ak,
            "--sk",
            sk,
            "--project-id",
            str(project_id),
        ]
    )


def main() -> int:
    print("请选择操作：")
    print("1. 批量质检")
    print("2. 上传")
    choice = input("输入 1 或 2: ").strip()
    roots = _prompt_roots()
    if not roots:
        print("[INFO] 未输入扫描目录，退出。")
        return 0

    if choice == "1":
        summary = batch_quality_check(roots)
        print(
            f"[DONE] 质检完成: total={summary['total']}, passed={summary['passed']}, "
            f"failed={summary['failed']}, failed_json={summary['output_json']}"
        )
        return 1 if summary["failed"] else 0

    if choice == "2":
        print("请选择上传范围：")
        print("1. 只上传质检通过的")
        print("2. 默认逻辑上传")
        upload_choice = input("输入 1 或 2: ").strip()
        ak, sk = _prompt_upload_credentials()
        upload_args = _build_upload_args(roots, ak=ak, sk=sk, project_id=DEFAULT_PROJECT_ID)
        if upload_choice == "1":
            sessions = discover_cp_cs_sessions(roots)
            summary = upload_quality_passed_sessions(sessions, args=upload_args)
            print(
                f"[DONE] 上传完成: uploaded={summary['uploaded']}, skipped={summary['skipped']}, "
                f"upload_failed={summary['upload_failed']}, skipped_json={summary['skipped_json']}"
            )
            return 1 if summary["upload_failed"] else 0
        if upload_choice == "2":
            return scan_and_upload_once(upload_args)
        print("[ERROR] 无效上传选项。")
        return 2

    print("[ERROR] 无效操作选项。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
