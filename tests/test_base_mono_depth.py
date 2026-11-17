"""CPU tests for orchestration safety; these do not substitute for a DA3 GPU trial."""
import importlib.util
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_base_mono_depth.py'
spec = importlib.util.spec_from_file_location('base_mono', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BaseMonoDepthTests(unittest.TestCase):
    def test_prefetch_runs_ahead_and_cancels_full_queue(self):
        advanced = threading.Event()
        closed = threading.Event()

        def source():
            try:
                for index in range(100):
                    if index == 2:
                        advanced.set()
                    yield index
            finally:
                closed.set()

        with module.prefetched(source(), 1, 'test-prefetch') as values:
            self.assertEqual(next(values), 0)
            self.assertTrue(advanced.wait(2), 'Producer did not run ahead of consumer')
        self.assertTrue(closed.is_set())
        self.assertFalse(any(t.name == 'test-prefetch' for t in threading.enumerate()))

    def test_prefetch_propagates_failure_in_order(self):
        def source():
            yield 'first'
            raise ValueError('decoder failed')

        with module.prefetched(source(), 1, 'test-failure') as values:
            self.assertEqual(next(values), 'first')
            with self.assertRaisesRegex(ValueError, 'decoder failed'):
                next(values)

    def test_single_decode_validation_checks_count_and_return_code(self):
        stream = dict(width=64, height=48, pix_fmt='gray12le', codec_name='hevc')
        for frames, code, expected_ok in ((17, 0, True), (16, 0, False), (17, 1, False)):
            proc = SimpleNamespace(stdout=io.StringIO(f'frame={frames}\nprogress=end\n'),
                                   wait=lambda: code, poll=lambda: code)
            with patch.object(module, 'probe', return_value=stream) as probe:
                with patch.object(module.subprocess, 'Popen', return_value=proc) as popen:
                    if expected_ok:
                        module.validate_video('depth.mp4', 17, 64, 48)
                    else:
                        with self.assertRaises(ValueError):
                            module.validate_video('depth.mp4', 17, 64, 48)
                    probe.assert_called_once_with('depth.mp4')
                    self.assertEqual(popen.call_count, 1)

    def test_streaming_matches_original_alignment_and_resize(self):
        import numpy as np
        upstream = SCRIPT.parents[2] / 'RoboCOIN-with-Depth/src/depth_repair/da3.py'
        tree = ast.parse(upstream.read_text(encoding='utf-8'))
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_make_chunks_streaming')
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'DA3DepthRepair')
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_infer_streaming')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[helper, original], type_ignores=[]),
                     str(upstream), 'exec'), namespace)

        class Network:
            def _preprocess_inputs(self, rgb, *args):
                return np.stack(rgb), None, None

            def _prepare_model_inputs(self, images, *args):
                return images, None, None

            def forward(self, images, *args, **kwargs):
                # Different chunk scales and offsets exercise cumulative alignment.
                return images * (1 + images[0, 0, 0] * 0.07) + images[0, 0, 0] * 0.2

            def _convert_to_prediction(self, raw):
                return SimpleNamespace(depth=raw)

        for h, w in ((10, 12), (1, 1)):
            rgb = [np.arange(h*w, dtype=np.float32).reshape(h, w)*0.01 + i + 1
                   for i in range(17)]
            model = SimpleNamespace(_model=Network(), _process_res=504)
            ranges = namespace['_make_chunks_streaming'](17, 8, 3)
            expected = namespace['_infer_streaming'](model, rgb, 17, 8, 3, 20, 24)
            chunks = ((start, end, rgb[start:end]) for start, end in ranges)
            with patch.dict(sys.modules, {'torch': SimpleNamespace(
                    inference_mode=contextlib.nullcontext)}):
                actual = list(module.depth_frames(model, chunks, 17, 24, 20, 8, 3))
                chunks = ((start, end, rgb[start:end]) for start, end in ranges)
                with module.prefetched(chunks, 1, 'test-rgb') as queued:
                    blocks = module.depth_frames(model, queued, 17, 24, 20, 8, 3, blocks=True)
                    with module.prefetched(blocks, 2, 'test-depth') as output:
                        parallel = list(module.flatten_blocks(output))
            self.assertEqual(len(actual), 17)
            np.testing.assert_array_equal(np.stack(actual), np.stack(expected))
            np.testing.assert_array_equal(np.stack(parallel), np.stack(expected))

    def test_decoder_keeps_only_chunk_and_overlap_and_checks_count(self):
        import numpy as np
        helper = lambda n, cs, ol: [(i, min(i+cs, n))
            for i in range(0, max(1, n-ol), cs-ol)]
        upstream = SimpleNamespace(_make_chunks_streaming=helper)
        payload = bytes(i for i in range(17) for _ in range(3))
        proc = SimpleNamespace(stdout=io.BytesIO(payload), wait=lambda: 0, poll=lambda: 0)
        with patch.dict(sys.modules, {'src.depth_repair.da3': upstream}):
            with patch.object(module.subprocess, 'Popen', return_value=proc):
                seen = []
                for start, end, frames in module.decoded_chunks('fake', 1, 1, 17, 8, 3):
                    self.assertLessEqual(len(frames), 8)
                    seen.append([int(f[0, 0, 0]) for f in frames])
        self.assertEqual(seen, [list(range(8)), list(range(5, 13)), list(range(10, 17))])

    def test_stream_encoding_preserves_original_clip_and_truncate(self):
        import numpy as np
        captured = []

        class Pipe(io.BytesIO):
            def close(self):
                if not self.closed:
                    captured.append(self.getvalue())
                super().close()

        proc = SimpleNamespace(stdin=Pipe(), wait=lambda: 0, poll=lambda: 0)
        frames = (np.array([[-2.0, 12.9, 5000]], dtype=np.float32) for _ in range(2))
        encoder = SimpleNamespace(_build_cmd=lambda *args: ['fake-ffmpeg'])
        with patch.object(module.subprocess, 'Popen', return_value=proc):
            module.encode_stream(encoder, frames, 3, 1, 'output.mp4', 2)
        np.testing.assert_array_equal(np.frombuffer(captured[0], dtype=np.uint16),
                                      [0, 12, 4095, 0, 12, 4095])

    def test_defaults_match_requested_overlap_and_preset(self):
        args = module.parser().parse_args(['--root', 'data', '--work-dir', 'work'])
        self.assertEqual((args.process_res, args.chunk_size, args.overlap,
                          args.crf, args.preset), (504, 80, 20, 0, 'fast'))
        self.assertFalse(hasattr(args, 'depth_unit_mm'))

    def test_scan_only_selected_rgb_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for batch in (1, 2):
                for camera in ('head_left_rectified', 'head_right_rectified', 'depth_head_left_rectified'):
                    video = root / f'batch_{batch:07d}/session/videos/chunk-000' / (
                        'observation.images.' + camera) / 'episode_000000.mp4'
                    video.parent.mkdir(parents=True)
                    video.touch()
            found = list(module.sessions(root, 'batch_*/*',
                'observation.images.head_left_rectified', first=2, last=2))
            self.assertEqual(len(found), 1)
            self.assertEqual(len(found[0][1]), 1)
            self.assertEqual(found[0][0].parent.name, 'batch_0000002')

    def test_failed_publication_restores_all_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / 'work'
            work.mkdir()
            pairs = []
            for i in range(3):
                source, target = work / f'new{i}', root / f'target{i}'
                source.write_text('new')
                target.write_text('old')
                pairs.append((source, target))
            replace = module.os.replace

            def fail_second(source, target):
                if Path(source).name == 'new1':
                    raise OSError('simulated publish failure')
                return replace(source, target)

            with patch.object(module.os, 'replace', side_effect=fail_second):
                with self.assertRaises(OSError):
                    module.publish(work, pairs)
            self.assertEqual([target.read_text() for _, target in pairs], ['old'] * 3)
            self.assertFalse((work / 'transaction.json').exists())

    def test_recover_interrupted_publication_removes_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'new-output'
            target.write_text('partial publication')
            module.write_json(root / 'transaction.json', [dict(
                target=str(target), backup=str(root / 'unused'), existed=False)])
            module.rollback(root)
            self.assertFalse(target.exists())

    def test_metadata_replaces_stereo_references_and_preserves_other_features(self):
        args = module.parser().parse_args(['--root', 'data', '--work-dir', 'work'])
        args.model = 'local-model'
        info = {'features': {args.camera: {'dtype': 'video'},
                 args.depth_feature: {'info': {'mask_path': 'stale'}},
                 'other': {'dtype': 'float32'}}}
        result = module.update_info(info, args, dict(height=8, width=10,
                                                    avg_frame_rate='30000/1001'))
        feature = result['features'][args.depth_feature]
        self.assertEqual(feature['shape'], [8, 10, 1])
        self.assertEqual(feature['info']['depth.scale'], 1)
        self.assertNotIn('mask_path', feature['info'])
        self.assertEqual(result['features']['other'], info['features']['other'])
        self.assertIn('mask_path', info['features'][args.depth_feature]['info'])

    def test_dry_run_does_not_create_work_or_import_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            root.mkdir()
            work = Path(tmp) / 'work'
            result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(root),
                '--work-dir', str(work), '--dry-run'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(work.exists())


if __name__ == '__main__':
    unittest.main()
