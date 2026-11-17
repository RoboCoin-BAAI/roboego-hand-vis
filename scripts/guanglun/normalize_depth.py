"""Normalize Lightwheel depth layout using source timestamps (requires pyarrow).

Preview:
  python normalize_depth.py --root /share/project/liuyou/humancoin/data/guanglun_depth
Apply:
  python normalize_depth.py --root /share/project/liuyou/humancoin/data/guanglun_depth --execute

Also accepts a single session directory. Videos are renamed, never re-encoded.
Old episode directories are removed after output verification, including sidecars.
Do not run against active depth jobs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


FEATURE = "observation.images.depth_head_left_rectified"
PARQUET = FEATURE + ".parquet"
SIDECARS = {"depth_hevc12.ffmpeg.log", "fast_foundation_stereo_video_meta.json",
            "lerobot_depth.json", "source_timestamps.parquet"}


def check_cleanup(source):
    if source.is_symlink() or not re.fullmatch(r"episode_\d+", source.name):
        raise ValueError("Unsafe episode directory: " + str(source))
    for path in source.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in SIDECARS | {"depth_hevc12.mkv"}:
            raise ValueError("Unexpected old episode content; refusing cleanup: " + str(path))


def cleanup(source, target, table, metadata):
    check_cleanup(source)
    if not pq.read_table(target / PARQUET).equals(table, check_metadata=True):
        raise ValueError("Output parquet verification failed")
    if json.loads((target / "lerobot_depth.json").read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output metadata verification failed")
    video = target / metadata["depth_storage"]["file"]
    if not video.is_file() or video.stat().st_size == 0 or (source / "depth_hevc12.mkv").exists():
        raise ValueError("Video move verification failed")
    # Delete only recognized sidecars, then remove the empty directory.
    for name in SIDECARS:
        (source / name).unlink(missing_ok=True)
    source.rmdir()
    print("REMOVED", source, flush=True)


def atomic_write(path, writer):
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def convert(source, execute):
    check_cleanup(source)
    episode = source.name
    episode_index = int(episode.removeprefix("episode_"))
    target = source.parent / FEATURE
    video = source / "depth_hevc12.mkv"
    output_video = target / (episode + ".mkv")
    metadata = json.loads((source / "lerobot_depth.json").read_text(encoding="utf-8"))
    timestamps = pq.read_table(source / "source_timestamps.parquet")
    storage = metadata["depth_storage"]
    frames = metadata["processed_frames"]
    count = storage["frames"]
    if not isinstance(count, int) or count <= 0 or len(frames) != count:
        raise ValueError("processed_frames length does not match depth_storage.frames")
    if any(type(i) is not int or i < 0 or i >= timestamps.num_rows for i in frames):
        raise ValueError("processed_frames contains invalid source row indices")
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ValueError("processed_frames must be strictly increasing")
    selected = timestamps.take(pa.array(frames, type=pa.int64()))
    times = selected["timestamp"].cast(pa.float64()).to_pylist()
    if any(t is None or not math.isfinite(t) for t in times):
        raise ValueError("Invalid source timestamp")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Source timestamps are not monotonic")
    if "episode_index" in selected.column_names:
        if any(i != episode_index for i in selected["episode_index"].to_pylist()):
            raise ValueError("Source episode_index does not match directory")
    table = pa.table({
        "timestamp": pa.array(times, type=pa.float64()),
        "episode_index": pa.array([episode_index] * count, type=pa.int64()),
    }).replace_schema_metadata({b"timestamp_domain": b"source_timestamp_seconds"})
    metadata["depth_storage"]["file"] = output_video.name
    metadata["timestamp_file"] = PARQUET
    metadata["timestamp_domain"] = "source_timestamp_seconds"
    metadata.pop("video_time_base", None)
    metadata["timestamp_lookup"] = (
        PARQUET + " row indexed by depth video display frame number; "
        "timestamp copied from source_timestamps.parquet at processed_frames source indices"
    )
    if video.exists() and output_video.exists():
        raise FileExistsError("Both source and destination videos exist; refusing overwrite")
    available = video if video.exists() else output_video
    if not available.is_file() or available.stat().st_size == 0:
        raise FileNotFoundError("Missing or empty depth video: " + str(available))
    output_parquet = target / PARQUET
    output_metadata = target / "lerobot_depth.json"
    if output_parquet.exists():
        existing = pq.read_table(output_parquet)
        if not existing.equals(table, check_metadata=True):
            raise FileExistsError("Destination parquet differs: " + str(output_parquet))
    if output_metadata.exists():
        existing = json.loads(output_metadata.read_text(encoding="utf-8"))
        if existing != metadata:
            raise FileExistsError("Destination metadata differs: " + str(output_metadata))
    if not video.exists() and output_parquet.exists() and output_metadata.exists():
        print("CLEANUP" if execute else "PLAN CLEANUP", source, flush=True)
        if execute:
            cleanup(source, target, table, metadata)
        return "cleaned" if execute else "planned"
    print("APPLY" if execute else "PLAN", source, "->", target,
          f"({count} frames; source timestamps)", flush=True)
    if execute:
        target.mkdir(exist_ok=True)
        if video.exists():
            video.rename(output_video)
        atomic_write(output_parquet, lambda p: pq.write_table(table, p))
        atomic_write(output_metadata, lambda p: p.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"))
        cleanup(source, target, table, metadata)
    return "converted" if execute else "planned"


def sessions(root, first, last):
    if (root / "depth").is_dir():
        yield root
        return
    batches = [root] if re.fullmatch(r"batch_\d+", root.name) else sorted(root.glob("batch_*"))
    for batch in batches:
        if not re.fullmatch(r"batch_\d+", batch.name) or not first <= int(batch.name[6:]) <= last:
            continue
        for session in sorted(batch.iterdir()):
            if session.is_dir() and (session / "depth").is_dir():
                yield session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-batch", type=int, default=1)
    parser.add_argument("--last-batch", type=int, default=9999999)
    parser.add_argument("--execute", action="store_true", help="Apply changes; default is preview")
    args = parser.parse_args()
    if not args.root.is_dir() or not 1 <= args.first_batch <= args.last_batch:
        parser.error("Invalid root or batch range")
    counts = dict(converted=0, planned=0, cleaned=0, skipped=0, failed=0)
    for session in sessions(args.root.resolve(), args.first_batch, args.last_batch):
        for chunk in sorted((session / "depth").glob("chunk-*")):
            episodes = [p for p in sorted(chunk.glob("episode_*"))
                        if p.is_dir() and re.fullmatch(r"episode_\d+", p.name)]
            # The reference layout has one shared metadata/parquet pair per chunk.
            if len(episodes) > 1:
                print("FAILED", chunk, "multiple episodes would share metadata/parquet", file=sys.stderr)
                counts["failed"] += 1
                continue
            for episode in episodes:
                try:
                    counts[convert(episode, args.execute)] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    print("FAILED", episode, str(exc), file=sys.stderr, flush=True)
    print(json.dumps(counts), flush=True)
    if not any(counts.values()):
        print("No depth episodes found", file=sys.stderr)
        return 1
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    sys.exit(main())
