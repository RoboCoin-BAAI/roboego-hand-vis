"""Stream millimeter depth directly into monochrome 12-bit HEVC."""
from pathlib import Path
import subprocess
from threading import Event, Lock

_processes = set()
_process_lock = Lock()
_cancelled = Event()


def _start_process(*args, **kwargs):
    with _process_lock:
        if _cancelled.is_set():
            raise RuntimeError('Depth processing cancelled')
        process = subprocess.Popen(*args, **kwargs)
        _processes.add(process)
        return process


def cancel_active_processes():
    """Stop only FFmpeg children started by this module in the current process."""
    with _process_lock:
        _cancelled.set()
        for process in list(_processes):
            if process.poll() is None:
                process.kill()


def _forget_process(process):
    with _process_lock:
        _processes.discard(process)


class HEVC12Writer:
    def __init__(self, path, *, size, fps, max_depth_m=4., ffmpeg="ffmpeg", preset="slow", threads=2, lossless=False):
        from roboego_hand_vis.archive_depth_cache.compute import build_hevc12_command

        if not 0 < max_depth_m <= 4.095:
            raise ValueError("HEVC12 millimeter depth requires 0 < max_depth_m <= 4.095")
        self.size, self.count, self.max_depth_m = size, 0, max_depth_m
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = build_hevc12_command(width=size[0], height=size[1], output_video=self.path,
            fps=fps, lossless=lossless, preset=preset, ffmpeg_bin=ffmpeg)
        # Keep the legacy archive module unchanged; FFmpeg 9 removed -vsync.
        command[command.index("-vsync")] = "-fps_mode"
        index = command.index("-x265-params") + 1
        command[index] += f":pools={threads}:frame-threads=1:log-level=error"
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("wb")
        self.process = None
        self.mask = None
        try:
            self.process = _start_process(command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self.log)
            if not lossless:
                from roboego_hand_vis.archive_depth_cache.compute import build_valid_mask_command
                self.mask_path = self.path.parent / "valid_mask_ffv1.mkv"
                mask_command = build_valid_mask_command(width=size[0], height=size[1],
                    output_video=self.mask_path, fps=fps, ffmpeg_bin=ffmpeg)
                mask_command[mask_command.index("-vsync")] = "-fps_mode"
                self.mask = _start_process(mask_command, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
            if self.process is not None:
                self.process.stdin.close()
                self.process.wait()
                _forget_process(self.process)
            self.log.close()
            raise

    def write(self, depth_m):
        from roboego_hand_vis.common.depth_io import depth_m_to_uint16_mm

        if depth_m.shape != (self.size[1], self.size[0]):
            raise ValueError("Depth video frame dimensions changed")
        # gray12le uses one little-endian 16-bit word per 12-bit sample.
        # Declare gray12le on input to avoid gray16-to-gray12 intensity scaling.
        encoded = depth_m_to_uint16_mm(depth_m, max_depth_m=self.max_depth_m).astype("<u2", copy=False)
        try:
            self.process.stdin.write(encoded.tobytes())
            if self.mask is not None:
                import numpy as np
                self.mask.stdin.write(np.where(encoded != 0, 255, 0).astype(np.uint8).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"HEVC12 encoding failed; see {self.log_path}") from exc
        self.count += 1

    def close(self):
        if self.log.closed:
            return
        errors = []
        try:
            for process in (self.process, self.mask):
                if process is None:
                    continue
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                if process.wait():
                    errors.append(f"Depth encoder failed; see {self.log_path}")
                _forget_process(process)
            if errors:
                raise RuntimeError("; ".join(errors))
        finally:
            self.log.close()
        self.log_path.unlink(missing_ok=True)


def _raw_frames(path, *, size, ffmpeg, pixel_format, dtype):
    import numpy as np
    process = _start_process([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", pixel_format, "pipe:1"], stdout=subprocess.PIPE)
    nbytes = size[0] * size[1] * np.dtype(dtype).itemsize
    try:
        while True:
            block = bytearray()
            while len(block) < nbytes:
                piece = process.stdout.read(nbytes - len(block))
                if not piece:
                    break
                block.extend(piece)
            if not block:
                break
            if len(block) != nbytes:
                raise ValueError("Incomplete depth/mask frame or incorrect dimensions")
            yield np.frombuffer(block, dtype=dtype).reshape(size[1], size[0])
        if process.wait():
            raise RuntimeError(f"Cannot decode {path}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
        _forget_process(process)


def read_depth_video(path, *, size, ffmpeg="ffmpeg", mask_path=None):
    """Decode millimeters; lossy MP4 requires its exact validity mask."""
    import numpy as np
    from itertools import zip_longest
    path = Path(path)
    if mask_path is None and path.suffix.lower() == '.mp4':
        mask_path = path.parent / 'valid_mask_ffv1.mkv'
    if mask_path is not None and not Path(mask_path).is_file():
        raise FileNotFoundError(f"Depth validity mask missing: {mask_path}")
    depths = _raw_frames(path, size=size, ffmpeg=ffmpeg, pixel_format='gray12le', dtype='<u2')
    masks = (_raw_frames(mask_path, size=size, ffmpeg=ffmpeg, pixel_format='gray', dtype='u1')
             if mask_path is not None else None)
    try:
        if masks is None:
            from roboego_hand_vis.common.depth_io import uint16_mm_to_depth_m
            for frame in depths:
                yield uint16_mm_to_depth_m(frame)
        else:
            for frame, mask in zip_longest(depths, masks):
                if frame is None or mask is None:
                    raise ValueError('Depth and validity mask frame counts differ')
                depth = frame.astype(np.float32) / 1000.
                depth[mask == 0] = np.nan
                yield depth
    finally:
        depths.close()
        if masks is not None:
            masks.close()


def read_episode_depth(output, *, ffmpeg="ffmpeg"):
    """Read published metadata, resolving video and mask relative to the session."""
    import json
    output = Path(output)
    report = json.loads((output / 'lerobot_depth.json').read_text(encoding='utf-8'))
    storage = report['depth_storage']
    root = (output / report['session_root_relative']).resolve() if 'session_root_relative' in report else (output.parents[2] if storage.get('path_base') == 'session' else output)
    mask = root / storage['mask_file'] if storage.get('requires_valid_mask') else None
    yield from read_depth_video(root / storage['file'], size=(storage['width'], storage['height']),
                               ffmpeg=ffmpeg, mask_path=mask)
