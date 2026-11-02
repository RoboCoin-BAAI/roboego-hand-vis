"""Stream millimeter depth directly into lossless monochrome 12-bit HEVC."""
from pathlib import Path
import subprocess


class HEVC12Writer:
    def __init__(self, path, *, size, fps, max_depth_m=4., ffmpeg="ffmpeg", preset="ultrafast", threads=2):
        from roboego_hand_vis.archive_depth_cache.compute import build_hevc12_command

        if not 0 < max_depth_m <= 4.095:
            raise ValueError("HEVC12 millimeter depth requires 0 < max_depth_m <= 4.095")
        self.size, self.count, self.max_depth_m = size, 0, max_depth_m
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = build_hevc12_command(width=size[0], height=size[1], output_video=self.path,
            fps=fps, lossless=True, preset=preset, ffmpeg_bin=ffmpeg)
        # Keep the legacy archive module unchanged; FFmpeg 9 removed -vsync.
        command[command.index("-vsync")] = "-fps_mode"
        index = command.index("-x265-params") + 1
        command[index] += f":pools={threads}:frame-threads=1:log-level=error"
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self.log)
        except BaseException:
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
        except BrokenPipeError as exc:
            raise RuntimeError(f"HEVC12 encoding failed; see {self.log_path}") from exc
        self.count += 1

    def close(self):
        if self.log.closed:
            return
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            if code:
                raise RuntimeError(f"HEVC12 encoder exited with {code}; see {self.log_path}")
        finally:
            self.log.close()


def read_depth_video(path, *, size, ffmpeg="ffmpeg"):
    """Yield float32 meter depth; use explicit gray12 decoding, never RGB."""
    import numpy as np
    from roboego_hand_vis.common.depth_io import uint16_mm_to_depth_m

    process = subprocess.Popen([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "gray12le", "pipe:1"], stdout=subprocess.PIPE)
    nbytes = size[0] * size[1] * 2
    exhausted = False
    try:
        while True:
            block = bytearray()
            while len(block) < nbytes:
                piece = process.stdout.read(nbytes - len(block))
                if not piece:
                    break
                block.extend(piece)
            if not block:
                exhausted = True
                break
            if len(block) != nbytes:
                raise ValueError("Incomplete depth video frame or incorrect dimensions")
            frame = np.frombuffer(block, dtype="<u2").reshape(size[1], size[0])
            yield uint16_mm_to_depth_m(frame)
        if process.wait():
            raise RuntimeError("Cannot decode HEVC12 depth video")
    finally:
        process.stdout.close()
        if not exhausted and process.poll() is None:
            process.terminate()
        process.wait()
