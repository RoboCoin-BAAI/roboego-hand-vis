"""Camera timestamp validity and alignment by nearest acquisition time."""
from bisect import bisect_left
from ..common import CAMERAS, result, valid_times, guarded, finite, stereo_boundary_checks


def alignment(reference, target, tolerance):
    errors = []
    outside = 0
    for t in reference:
        if t < target[0] or t > target[-1]:
            outside += 1
            continue
        i = bisect_left(target, t)
        errors.append(min(abs(target[j] - t) for j in (max(0, i - 1), min(i, len(target) - 1))))
    return dict(over_tolerance_rate=sum(e > tolerance for e in errors) / len(errors) if errors else 1.0,
                max_nearest_error_s=max(errors, default=None), interior_frames=len(errors),
                outside_range_frames=outside, start_offset_s=target[0] - reference[0],
                end_offset_s=target[-1] - reference[-1])


def check_camera(ctx, episode, camera):
    ts = ctx.times(episode, camera)
    checks = [result("monotonic", valid_times(ts), frames=len(ts),
                     level="warning" if ts and finite(ts) else "fail")]
    if not ts or not finite(ts):
        return checks
    fps = ctx.fps(camera)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    checks.append(result("gaps", not any(g > 2.5 / fps for g in gaps), level="warning",
                         max_gap_s=max(gaps, default=0)))
    master = ctx.table(episode, "master_index")["timestamp"]
    checks.append(result("master_index_valid", valid_times(master),
                         level="warning" if master and finite(master) else "fail"))
    checks.append(result("master_count", len(ts) == len(master), frames=len(ts), master_frames=len(master)))
    if camera == "head_left":
        checks.append(result("master_timestamps", bool(master) and finite(master) and len(ts) == len(master)
            and all(abs(a - b) <= 1e-4 for a, b in zip(ts, master))))
    else:
        ref = ctx.times(episode, "head_left")
        if valid_times(ref) and valid_times(ts):
            stats = alignment(ref, ts, ctx.tolerance_s)
            checks.append(result("nearest_alignment", stats["over_tolerance_rate"] <= 0.001,
                                 level="warning", **stats))
            checks.append(result("boundary_coverage", stats["outside_range_frames"] == 0,
                                 level="warning", outside_range_frames=stats["outside_range_frames"]))
    return checks


def check(ctx):
    checks = []
    for episode in ctx.episodes:
        boundaries = guarded('readable', lambda: stereo_boundary_checks(ctx, episode, 'head_left', 'head_right'))
        for item in boundaries:
            item['name'] = f"{episode.parent.name}/{episode.name}/stereo_boundary/{item['name']}"
        checks.extend(boundaries)
        for camera in CAMERAS:
            prefix = f"{episode.parent.name}/{episode.name}/{camera}"
            items = guarded("readable", lambda: check_camera(ctx, episode, camera))
            for item in items:
                item["name"] = f"{prefix}/{item['name']}"
            checks.extend(items)
    return checks
