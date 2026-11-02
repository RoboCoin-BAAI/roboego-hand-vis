"""Repair overstated PCM WAV lengths in place, then validate the audio."""
import os
import struct
import wave
from ..common import result, valid_times


def repair_wav_header_sizes(path):
    """Patch RIFF/data lengths only; never rewrite the payload or create a backup."""
    with open(path, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        header = f.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            return dict(applied=False, reason='not_riff_wave')
        riff = struct.unpack_from('<I', header, 4)[0]
        fmt = None
        while f.tell() + 8 <= size:
            offset = f.tell()
            chunk, length = struct.unpack('<4sI', f.read(8))
            if chunk == b'data':
                if fmt is None:
                    return dict(applied=False, reason='missing_pcm_fmt')
                encoding, channels, rate, byte_rate, align, bits = fmt
                if (encoding != 1 or channels <= 0 or rate <= 0 or bits <= 0
                        or bits % 8 or align != channels * (bits // 8) or byte_rate != rate * align):
                    return dict(applied=False, reason='unsupported_or_invalid_pcm_fmt')
                available = size - offset - 8
                if length <= available and riff <= size - 8:
                    return dict(applied=False, reason='no_overstated_lengths')
                if length > available and (available <= 0 or available % align):
                    return dict(applied=False, reason='payload_not_block_aligned')
                actual = available if length > available else length
                if size - 8 > 0xffffffff or actual > 0xffffffff:
                    return dict(applied=False, reason='riff_size_overflow')
                report = dict(applied=True, reason='overstated_header_lengths',
                              original_riff_size=riff, actual_riff_size=size - 8,
                              original_data_size=length, actual_data_size=actual)
                break
            if length > size - f.tell():
                return dict(applied=False, reason='invalid_chunk_length')
            if chunk == b'fmt ' and length >= 16:
                fmt = struct.unpack('<HHIIHH', f.read(16))
            f.seek(offset + 8 + length + length % 2)
        else:
            return dict(applied=False, reason='missing_data_chunk')
    with open(path, 'r+b') as f:
        f.seek(4)
        f.write(struct.pack('<I', size - 8))
        f.seek(offset + 4)
        f.write(struct.pack('<I', actual))
    return report


def inspect_wav(path):
    repair = repair_wav_header_sizes(path)
    with wave.open(str(path), "rb") as wav:
        frames, rate, channels, width = wav.getnframes(), wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        if min(frames, rate, channels, width) <= 0:
            raise ValueError("Empty WAV or invalid audio parameters")
        actual = 0
        while True:
            block = wav.readframes(65536)
            if not block:
                break
            actual += len(block)
    return dict(duration_s=frames / rate, sample_rate=rate, channels=channels,
                expected_bytes=frames * channels * width, readable_bytes=actual,
                wav_header_repair=repair)


def check(ctx):
    info = inspect_wav(ctx.root / "audio/audio.wav")
    checks = [result("data_length", info["expected_bytes"] == info["readable_bytes"], **info)]
    if info['wav_header_repair']['applied']:
        checks.append(result('header_repaired', False, level='warning',
                             detail='WAV length fields repaired in place without backup',
                             **info['wav_header_repair']))
    # Current L1 converter writes one session/episode per dataset. A session WAV
    # has no episode mapping for general multi-episode datasets.
    if len(ctx.episodes) != 1:
        checks.append(result("episode_mapping", False, level="warning",
                             detail="Session WAV cannot be assigned to multiple episodes"))
        return checks
    ts = ctx.times(ctx.episodes[0], "head_left")
    if valid_times(ts):
        duration = ts[-1] - ts[0] + 1 / ctx.fps()
        delta = abs(info["duration_s"] - duration)
        checks.append(result("video_duration", delta <= ctx.duration_tolerance_s,
                             delta_s=delta, limit_s=ctx.duration_tolerance_s))
    return checks
