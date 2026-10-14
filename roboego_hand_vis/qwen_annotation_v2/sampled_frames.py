"""Stable sampled-frame names shared by annotation and its downstream readers."""

import math


def sampled_frame_name(timestamp):
    timestamp = float(timestamp)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Sample time must be finite and nonnegative')
    seconds, fraction = f'{timestamp:.6f}'.split('.')
    suffix = '' if fraction == '000000' else '_' + fraction
    return f'frame_t{int(seconds):06d}{suffix}.jpg'
