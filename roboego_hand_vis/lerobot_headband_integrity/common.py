"""Shared dataset layout and report primitives; no LeRobot or GPU dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

CAMERAS = ("head_left", "head_right", "head_left_outer0", "head_right_outer0")
IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


def result(name, passed, *, level="fail", detail=None, **stats):
    return dict(name=name, passed=bool(passed), level=level, detail=detail, statistics=stats)


def finite(values):
    return all(v is not None and math.isfinite(float(v)) for v in values)


def valid_times(values):
    return bool(values) and finite(values) and all(b > a for a, b in zip(values, values[1:]))


@dataclass
class Context:
    root: Path
    ffprobe: str = "ffprobe"
    tolerance_s: float = 0.02
    duration_tolerance_s: float = 5.0
    imu_gap_frames: float = 10
    full_video_check: bool = False
    info: dict = field(init=False)
    episodes: list[Path] = field(init=False)
    _tables: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.root = self.root.resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text(encoding="utf-8"))
        self.episodes = sorted((self.root / "data").glob("chunk-*/episode_*"))
        if not self.episodes:
            raise ValueError("No data/chunk-*/episode_* directories found")

    def table(self, episode, feature):
        path = episode / f"{feature}.parquet"
        if path not in self._tables:
            import pyarrow.parquet as pq
            self._tables[path] = pq.read_table(path).to_pydict()
        return self._tables[path]

    def times(self, episode, camera):
        return self.table(episode, f"observation.images.{camera}")["timestamp"]

    def video(self, episode, camera):
        return self.root / "videos" / episode.parent.name / f"observation.images.{camera}" / f"{episode.name}.mp4"

    def fps(self, camera="head_left"):
        fps = float(self.info["features"][f"observation.images.{camera}"].get("fps", self.info["fps"]))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid declared camera FPS")
        return fps


def guarded(name, function):
    try:
        return function()
    except Exception as exc:
        return [result(name, False, detail=f"{type(exc).__name__}: {exc}")]


def stereo_boundary_checks(ctx, episode, left_camera, right_camera, max_frames=2.):
    """Compare acquisition timestamps at the actual first and last video frames."""
    left = ctx.times(episode, left_camera)
    right = ctx.times(episode, right_camera)
    if not left or not right or not finite(left) or not finite(right):
        return [result('timestamps_readable', False,
                       detail='Stereo boundaries require nonempty finite timestamps')]
    fps = ctx.fps(left_camera)
    checks = []
    for name, index in (('start', 0), ('end', -1)):
        delta = abs(float(left[index]) - float(right[index]))
        frames = delta * fps
        checks.append(result(name, frames <= max_frames or math.isclose(frames, max_frames, rel_tol=0., abs_tol=1e-6),
                             delta_s=delta, delta_frames=frames, limit_frames=max_frames,
                             reference_fps=fps, left_timestamp=left[index], right_timestamp=right[index]))
    return checks
