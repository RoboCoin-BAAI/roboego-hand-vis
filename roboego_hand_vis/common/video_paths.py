from __future__ import annotations

from pathlib import Path


VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".m4v")


def normalize_video_ext(video_ext: str | None) -> str | None:
    if video_ext is None:
        return None
    value = str(video_ext).strip()
    if not value or value.lower() == "auto":
        return None
    return value if value.startswith(".") else f".{value}"


def find_named_video(
    videos_dir: str | Path,
    stem: str,
    *,
    preferred_ext: str | None = ".mp4",
    suffixes: tuple[str, ...] = VIDEO_SUFFIXES,
) -> Path:
    videos_dir = Path(videos_dir)
    preferred = normalize_video_ext(preferred_ext)
    if preferred is not None:
        preferred_path = videos_dir / f"{stem}{preferred}"
        if preferred_path.is_file():
            return preferred_path

    candidates = [
        videos_dir / f"{stem}{suffix}"
        for suffix in suffixes
        if (videos_dir / f"{stem}{suffix}").is_file()
    ]
    if preferred is not None:
        candidates = [path for path in candidates if path.suffix != preferred]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileExistsError(
            f"Multiple video files match {stem!r} under {videos_dir}: "
            + ", ".join(path.name for path in candidates)
        )

    tried = [f"{stem}{suffix}" for suffix in suffixes]
    if preferred is not None and preferred not in suffixes:
        tried.insert(0, f"{stem}{preferred}")
    raise FileNotFoundError(
        f"Missing video for {stem!r} under {videos_dir}; tried: {', '.join(tried)}"
    )
