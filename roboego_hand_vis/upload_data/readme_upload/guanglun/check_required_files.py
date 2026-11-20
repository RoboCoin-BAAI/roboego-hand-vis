#!/usr/bin/env python3
"""检查光轮各 batch 样本的固定文件；标准输出仅打印缺失文件路径。"""

import argparse
import os
from pathlib import Path
import stat
import sys


PARQUETS = (
    "eye_gaze_predict",
    "master_index",
    "observation.bad_frame_pose_hand",
    "observation.bad_frame_pose_lower_body",
    "observation.bad_frame_pose_upper_body",
    "observation.body_pose",
    "observation.hand_left_points",
    "observation.hand_left_quaternions",
    "observation.hand_right_points",
    "observation.hand_right_quaternions",
    "observation.head_camera_pose",
    "observation.head_pose",
    "observation.images.depth_head_left_rectified",
    "observation.images.head_left_rectified",
    "observation.images.head_right_rectified",
    "observation.lower_body_pose",
    "observation.right_eye_camera_pose",
    "observation.upper_body_pose",
)
REQUIRED_FILES = (
    "annotation/episode_000000_structured.json",
    "calibrations/dataset_manifest.json",
    "calibrations/rectify_params.yaml",
    *(f"data/chunk-000/episode_000000/{name}.parquet" for name in PARQUETS),
    *(f"meta/{name}" for name in (
        "episodes.jsonl", "episodes_stats.jsonl", "info.json", "tasks.jsonl"
    )),
    *(f"videos/chunk-000/observation.images.{name}/episode_000000.mp4"
      for name in ("depth_head_left_rectified", "head_left_rectified", "head_right_rectified")),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "/share/project/liuyou/humancoin/data/guanglun_depth"))
    parser.add_argument("--start", type=int, default=1, help="起始 batch（含），默认 1")
    parser.add_argument("--end", type=int, default=39, help="结束 batch（含），默认 39")
    args = parser.parse_args()
    if not 1 <= args.start <= args.end:
        parser.error("需要 1 <= start <= end")

    samples = missing = incomplete = errors = 0
    for number in range(args.start, args.end + 1):
        batch = args.root / f"batch_{number:07d}"
        try:
            with os.scandir(batch) as entries:
                # 样本目录为 64 位十六进制 ID；跳过 logs 等辅助目录。
                directories = sorted(
                    entry.path for entry in entries
                    if len(entry.name) == 64
                    and all(c in "0123456789abcdefABCDEF" for c in entry.name)
                    and entry.is_dir()
                )
        except OSError as exc:
            print(f"[ERROR] {batch}: {exc}", file=sys.stderr)
            errors += 1
            continue
        if not directories:
            print(f"[ERROR] {batch}: 没有找到样本目录", file=sys.stderr)
            errors += 1
        for directory in directories:
            samples += 1
            sample_missing = 0
            for relative in REQUIRED_FILES:
                path = Path(directory) / relative
                try:
                    # 用 stat 区分文件缺失与权限、I/O 等读取错误。
                    exists = stat.S_ISREG(path.stat().st_mode)
                except FileNotFoundError:
                    exists = False
                except OSError as exc:
                    print(f"[ERROR] {path}: {exc}", file=sys.stderr)
                    errors += 1
                    continue
                if not exists:
                    print(path, flush=True)
                    sample_missing += 1
            missing += sample_missing
            incomplete += bool(sample_missing)
        print(f"[PROGRESS] {batch.name}: {len(directories)} 个样本", file=sys.stderr)

    print(f"[SUMMARY] 样本={samples}，缺文件样本={incomplete}，"
          f"缺失文件={missing}，检查错误={errors}", file=sys.stderr)
    return 2 if errors else (1 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())
