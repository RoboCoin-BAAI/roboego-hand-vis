"""Bounded CPU prefetch and ordered background output for GPU inference."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import queue
import threading


def stereo_frame_range(left_count, right_count, start, num_frames, right_offset=0):
    """Right index = left index + offset; trim unavailable boundaries only."""
    end = left_count if num_frames is None else start + num_frames
    if start < 0 or end > left_count or end <= start:
        raise ValueError("Requested frame range is outside left video")
    begin = max(start, -right_offset)
    end = min(end, right_count - right_offset)
    if end <= begin:
        raise ValueError("No valid stereo frames for this offset")
    return begin, end


def prefetch(source, capacity):
    """Propagate producer errors and cancel a full producer on consumer failure."""
    if capacity < 1:
        raise ValueError("prefetch capacity must be positive")
    buffer = queue.Queue(capacity)
    stop = threading.Event()

    def send(value):
        while not stop.is_set():
            try:
                buffer.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def produce():
        try:
            for item in source:
                if stop.is_set():
                    break
                send(("item", item))
        except BaseException as exc:
            send(("error", exc))
        finally:
            close = getattr(source, "close", None)
            if close:
                close()
            send(("end", None))

    thread = threading.Thread(target=produce, name="stereo-decode", daemon=True)
    thread.start()
    try:
        while True:
            kind, item = buffer.get()
            if kind == "error":
                raise item
            if kind == "end":
                break
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


class OrderedWriter:
    """Single writer preserves preview order, with bounded outstanding frames."""
    def __init__(self, function, capacity):
        if capacity < 1:
            raise ValueError("writer capacity must be positive")
        self.function, self.capacity = function, capacity
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth-save")
        self.pending = deque()

    def submit(self, *args):
        if len(self.pending) >= self.capacity:
            self.pending.popleft().result()
        self.pending.append(self.pool.submit(self.function, *args))

    def close(self):
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)
