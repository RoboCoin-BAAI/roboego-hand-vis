"""Time-owned batches with visual context, validated timestamps and bounded requests."""

from bisect import bisect_left
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Batch:
    index: int
    targets: tuple[int, ...]
    context: tuple[int, ...]


def plan_batches(times, batch_size, overlap_frames, fps, *, windows=None):
    if batch_size < 1 or overlap_frames < 0 or fps <= 0:
        raise ValueError('batch_size/fps must be positive; overlap_frames must be nonnegative')
    if not times or any(not math.isfinite(float(t)) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Sample times must be finite and strictly increasing')
    # Discontinuous valid windows must not be presented as continuous motion.
    segments = None
    if windows is not None:
        merged = []
        for window in sorted(windows, key=lambda window: float(window['start'])):
            start, end = float(window['start']), float(window['end'])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError('Valid windows must have finite increasing bounds')
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        segments = []
        for time in times:
            segment = next((i for i, (start, end) in enumerate(merged) if start-1e-6 <= time < end), None)
            if segment is None:
                raise ValueError('Sample timestamp is outside the supplied valid windows')
            segments.append(segment)
    cuts = [0] + [i for i in range(1, len(times))
                  if times[i] - times[i-1] > 1.5 / fps + 1e-6
                  or (segments is not None and segments[i] != segments[i-1])] + [len(times)]
    plans = []
    for low, high in zip(cuts, cuts[1:]):
        for start in range(low, high, batch_size):
            end = min(high, start + batch_size)
            plans.append(Batch(len(plans), tuple(range(start, end)),
                               tuple(range(max(low, start-overlap_frames), min(high, end+overlap_frames)))))
    return plans


def select_target_frames(frames, targets, allowed):
    """Context labels may be returned, but only owned targets are saved."""
    allowed = sorted(float(t) for t in allowed)
    tolerance = min(0.000501, min((b-a for a, b in zip(allowed, allowed[1:])), default=1) / 4)
    by_time = {}
    for frame in frames:
        value = frame.get('t')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid frame timestamp: {value!r}')
        pos = bisect_left(allowed, value)
        candidates = allowed[max(0, pos-1):pos+1]
        if not candidates or min(abs(t-value) for t in candidates) > tolerance:
            raise ValueError(f'Unrequested frame timestamp: {value}')
        exact = min(candidates, key=lambda t: abs(t-value))
        if exact in by_time:
            raise ValueError(f'Duplicate frame timestamp: {value}')
        by_time[exact] = dict(frame, t=exact)
    missing = sorted(set(targets) - by_time.keys())
    if missing:
        raise ValueError(f'Missing target frame timestamps: {missing}')
    return [by_time[t] for t in targets]


def batch_prompt(user_prompt, targets, context):
    return user_prompt + '\n\n' + (
        'This request is a window of one continuous recording, not a separate task. '
        'Images are ordered by their original video timestamps. Use preceding and following '
        'context images to distinguish action onset, continuation and completion. '
        'Do not infer a new action or a completed task merely from the first or last image. '
        'Describe only visible evidence; a gap or an occlusion is not evidence of completion. '
        'Keep object names consistent with the task vocabulary across the sequence. '
        f'Target timestamps to annotate exactly once: {list(targets)}. '
        f'Context-only timestamps: {list(context)}. '
        'Return labels for target timestamps only; keep the existing JSON frame schema.'
    )


def iter_results(plans, workers, request):
    if workers < 1:
        raise ValueError('max_concurrency must be positive')
    if workers == 1:
        for plan in plans:
            yield plan, request(plan)
        return
    pool = ThreadPoolExecutor(max_workers=workers)
    pending, remaining = {}, iter(plans)
    try:
        def fill():
            while len(pending) < workers:
                plan = next(remaining, None)
                if plan is None:
                    return
                pending[pool.submit(request, plan)] = plan
        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield pending.pop(future), future.result()
            fill()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
