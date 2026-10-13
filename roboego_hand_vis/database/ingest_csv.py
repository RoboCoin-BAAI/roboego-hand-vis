import argparse
import csv
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="读取 CSV 路径并批量入库")
    parser.add_argument("csv_path", help="CSV 文件路径")
    parser.add_argument("--column", default="path", help="路径列名，默认 path")
    parser.add_argument("--table", default="l1data", help="目标表名")
    parser.add_argument(
        "--prefix",
        default="ks3://baai-eai-datasets/data/",
        help="相对路径的前缀",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--dry-run", action="store_true", help="只预览路径，不执行入库"
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")

    paths = []
    seen = set()
    nonempty_count = 0

    with open(args.csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if args.column not in (reader.fieldnames or []):
            raise SystemExit(
                f"找不到列 {args.column!r}，实际列名：{reader.fieldnames}"
            )

        for row in reader:
            value = (row.get(args.column) or "").strip()
            if not value:
                continue

            nonempty_count += 1

            # 完整 KS3 路径直接使用，其他内容拼接前缀
            if value.startswith("ks3://"):
                path = value.rstrip("/")
            else:
                path = (
                    args.prefix.rstrip("/") + "/" + value.lstrip("/")
                ).rstrip("/")

            if path not in seen:
                seen.add(path)
                paths.append(path)

    print(f"非空路径数：{nonempty_count}", flush=True)
    print(f"去重后路径数：{len(paths)}", flush=True)
    print(f"重复路径数：{nonempty_count - len(paths)}", flush=True)

    if not paths:
        print("没有可入库的路径")
        return

    if args.dry_run:
        for path in paths:
            print(path)
        return

    for start in range(0, len(paths), args.batch_size):
        batch = paths[start:start + args.batch_size]
        print(
            f"\n入库：{start + 1}–{start + len(batch)} / {len(paths)}",
            flush=True,
        )

        subprocess.run(
            [
                sys.executable,
                "-m", "roboego_hand_vis.database.ingest",
                "--table", args.table,
                *batch,
            ],
            check=True,
        )

    print("\n全部入库命令执行成功", flush=True)


if __name__ == "__main__":
    main()