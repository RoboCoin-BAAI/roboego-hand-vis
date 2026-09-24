from __future__ import annotations

import struct
import wave
from pathlib import Path

from roboego_hand_vis.raw_headband_integrity.logic import inspect_audio


def _write_wav(path: Path, *, frames: int = 48_000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\0" * frames * 2 * 2)


def test_inspect_audio_repairs_oversized_wav_data_header(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"
    _write_wav(path)
    raw = bytearray(path.read_bytes())
    struct.pack_into("<I", raw, 4, 0x80000024)
    struct.pack_into("<I", raw, 40, 0x80000000)
    path.write_bytes(raw)

    summary = inspect_audio(path)

    assert summary is not None
    assert summary["duration_s"] == 1.0
    assert summary["wav_header_repair"]["applied"] is True
    assert summary["wav_header_repair"]["original_data_size"] == 0x80000000

    repaired = path.read_bytes()
    assert struct.unpack_from("<I", repaired, 4)[0] == len(repaired) - 8
    assert struct.unpack_from("<I", repaired, 40)[0] == len(repaired) - 44
