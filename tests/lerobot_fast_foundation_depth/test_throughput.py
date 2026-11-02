import os
from pathlib import Path
import tempfile
import unittest

from roboego_hand_vis.lerobot_fast_foundation_depth.L1.pipeline import OrderedWriter, prefetch, stereo_frame_range
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.throughput import run_jobs


class FakeEngine:
    def __init__(self, options):
        self.options = options
        (Path(options["init_dir"]) / str(os.getpid())).write_text("loaded")

    def process(self, task):
        if task.get("fail"):
            raise ValueError("injected task failure")
        return dict(episode=task["episode"], pid=os.getpid())


class BrokenEngine:
    def __init__(self, options):
        raise RuntimeError("model load failed")


class ThroughputTests(unittest.TestCase):
    def test_stereo_offset_range(self):
        self.assertEqual(stereo_frame_range(505, 505, 0, None, -1), (1, 505))
        self.assertEqual(stereo_frame_range(505, 505, 0, None, 1), (0, 504))
        self.assertEqual(stereo_frame_range(505, 505, 10, 100, -1), (10, 110))
        self.assertEqual(stereo_frame_range(505, 505, 0, None), (0, 505))
        with self.assertRaises(ValueError):
            stereo_frame_range(5, 5, 0, None, -10)

    def test_prefetch_order_and_failure(self):
        self.assertEqual(list(prefetch(iter(range(20)), 2)), list(range(20)))
        def broken():
            yield 1
            raise ValueError("decoder failure")
        with self.assertRaisesRegex(ValueError, "decoder failure"):
            list(prefetch(broken(), 1))

    def test_cancel_full_producer(self):
        stream = prefetch(iter(range(1000000)), 1)
        self.assertEqual(next(stream), 0)
        stream.close()

    def test_ordered_writer_and_failure(self):
        values = []
        writer = OrderedWriter(values.append, 2)
        for i in range(10):
            writer.submit(i)
        writer.close()
        self.assertEqual(values, list(range(10)))
        def broken(value):
            raise ValueError("writer failure")
        writer = OrderedWriter(broken, 2)
        writer.submit(1)
        with self.assertRaisesRegex(ValueError, "writer failure"):
            writer.close()

    def test_persistent_workers_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = [dict(task_id=i, session=str(root), episode=str(root / f"data/chunk-000/episode_{i:06d}"), fail=i == 2)
                    for i in range(8)]
            results = run_jobs(jobs, ["0", "1"], dict(init_dir=directory), FakeEngine)
            self.assertEqual(len(results), 8)
            self.assertEqual(sum(not r["passed"] for r in results), 1)
            self.assertEqual(sum(p.is_file() for p in root.iterdir()), 2)
            self.assertLessEqual(len({r["pid"] for r in results if r["passed"]}), 2)

    def test_model_startup_failure_does_not_hang(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = [dict(task_id=0, session=directory, episode=directory)]
            results = run_jobs(jobs, ["0"], {}, BrokenEngine)
            self.assertFalse(results[0]["passed"])


if __name__ == "__main__":
    unittest.main()
