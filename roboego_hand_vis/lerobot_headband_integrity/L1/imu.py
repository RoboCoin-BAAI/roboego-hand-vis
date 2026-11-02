"""Native-rate IMU validity, signal anomalies and camera time coverage."""
import statistics
from ..common import CAMERAS, IMU_COLUMNS, finite, valid_times, guarded, result


def signal_quality(columns, *, jump_absolute=20., jump_ratio=100.):
    frozen, jumps, constant = [], [], []
    for name, values in columns.items():
        if len(values) >= 2 and all(v == values[0] for v in values[1:]):
            constant.append(name)
        deltas = [abs(b - a) for a, b in zip(values, values[1:])]
        if deltas and sum(d > 1e-12 for d in deltas) / len(deltas) <= 0.001:
            frozen.append(name)
        positive = [d for d in deltas if d > 1e-12]
        typical = statistics.median(positive) if positive else 0
        maximum = max(deltas, default=0)
        if maximum >= jump_absolute and (typical == 0 or maximum / typical >= jump_ratio):
            jumps.append(name)
    return dict(frozen_columns=frozen, jump_columns=jumps,
                constant_columns=constant,
                all_columns_frozen=bool(columns) and len(frozen) == len(columns),
                jump_absolute_threshold=jump_absolute, jump_ratio_threshold=jump_ratio)


def check_episode(ctx, episode):
    table = ctx.table(episode, "observation.imu")
    ts = table["timestamp"]
    finite_ts = bool(ts) and finite(ts)
    checks = [result("timestamps", valid_times(ts), samples=len(ts),
                     level="warning" if finite_ts else "fail",
                     detail="Non-increasing finite IMU timestamps are accepted by QC" if finite_ts and not valid_times(ts) else None)]
    columns = {k: table[k] for k in IMU_COLUMNS}
    valid = bool(ts) and all(len(v) == len(ts) and finite(v) for v in columns.values())
    checks.append(result("finite_values", valid))
    if not valid:
        return checks
    checks.append(result("all_zero", not all(abs(x) <= 1e-12 for v in columns.values() for x in v)))
    stats = signal_quality(columns)
    checks.append(result("frozen", not stats["frozen_columns"],
                         level="fail", **stats))
    checks.append(result("jump", not stats["jump_columns"], **stats))
    if finite_ts:
        # Coverage uses ordered unique instants without modifying stored samples.
        coverage_ts = sorted(set(ts))
        for camera in CAMERAS:
            cam = ctx.times(episode, camera)
            if not valid_times(cam):
                continue
            # Include internal gaps, not just the first and last timestamps.
            gaps = [max(0, coverage_ts[0] - cam[0]), max(0, cam[-1] - coverage_ts[-1])]
            gaps.extend(b - a for a, b in zip(coverage_ts, coverage_ts[1:]) if b > cam[0] and a < cam[-1])
            missing_frames = max(gaps) * ctx.fps(camera)
            checks.append(result(f"coverage/{camera}", missing_frames <= ctx.imu_gap_frames,
                                 max_gap_frames=missing_frames, limit_frames=ctx.imu_gap_frames))
    return checks


def check(ctx):
    checks = []
    for episode in ctx.episodes:
        items = guarded("readable", lambda: check_episode(ctx, episode))
        for item in items:
            item["name"] = f"{episode.parent.name}/{episode.name}/{item['name']}"
        checks.extend(items)
    return checks
