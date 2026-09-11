import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roboego_hand_vis.video_privacy_blur.batch import main, run_batch


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.outputs = self.root / 'outputs'
        self.calls = []
        for name in ['left.mp4', 'right.mp4', 'left_3d.mp4', 'overlay.mp4', 'left_blurred.mp4']:
            (self.inputs / name).write_bytes(name.encode())

    def process(self, source, output, config, log):
        self.calls.append(source.name)
        output.write_bytes(b'processed:' + source.read_bytes())
        return {'frames': 1}

    def run_selected(self, selection=None, config=None, processor=None):
        return run_batch(self.inputs, self.outputs, config or {}, retries=0,
                         processor=processor or self.process,
                         selected_videos=['left.mp4', 'right.mp4'] if selection is None else selection)

    def test_selection_excludes_derived_videos_and_resumes(self):
        self.assertEqual(self.run_selected(), {'completed': 2, 'failed': 0, 'skipped': 0})
        self.assertCountEqual(self.calls, ['left.mp4', 'right.mp4'])
        self.calls.clear()
        self.assertEqual(self.run_selected(), {'completed': 0, 'failed': 0, 'skipped': 2})
        self.assertEqual(self.calls, [])
        self.assertEqual(len(list(self.outputs.glob('*.mp4'))), 2)

    def test_selection_partial_failure_and_changed_config(self):
        def fail_right(source, output, config, log):
            if source.name == 'right.mp4':
                raise RuntimeError('decode failed')
            return self.process(source, output, config, log)
        self.assertEqual(self.run_selected(processor=fail_right), {'completed': 1, 'failed': 1, 'skipped': 0})
        self.calls.clear()
        self.assertEqual(self.run_selected(), {'completed': 1, 'failed': 0, 'skipped': 1})
        self.assertEqual(self.calls, ['right.mp4'])
        self.calls.clear()
        self.assertEqual(self.run_selected(config={'scale': 2}), {'completed': 2, 'failed': 0, 'skipped': 0})
        self.assertCountEqual(self.calls, ['left.mp4', 'right.mp4'])

    def test_invalid_selection_fails_before_mutations(self):
        (self.inputs / 'bad.txt').write_text('bad')
        (self.inputs / 'directory.mp4').mkdir()
        os.mkfifo(self.inputs / 'pipe.mp4')
        (self.inputs / 'link.mp4').symlink_to(self.inputs / 'left.mp4')
        (self.inputs / 'linked').symlink_to(self.inputs, target_is_directory=True)
        for selection in [[], ['left.mp4', 'missing.mp4'], ['left.mp4', 'left.mp4'],
                          ['left.mp4', './left.mp4'], ['../inputs/left.mp4'],
                          [str(self.inputs / 'left.mp4')], ['bad.txt'], ['directory.mp4'],
                          ['pipe.mp4'], ['link.mp4'], ['linked/left.mp4']]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.run_selected(selection)
            self.assertFalse(self.outputs.exists())
            self.assertEqual(self.calls, [])

    def test_nested_regular_files_are_supported(self):
        (self.inputs / 'nested').mkdir()
        (self.inputs / 'nested/video.MP4').write_bytes(b'video')
        self.assertEqual(self.run_selected(['nested/video.MP4'])['completed'], 1)
        self.assertTrue((self.outputs / 'nested/video_blurred.mp4').is_file())

    def test_selection_requires_directory(self):
        with self.assertRaises(ValueError):
            run_batch(self.inputs / 'left.mp4', self.outputs, {}, selected_videos=['left.mp4'])
        self.assertFalse(self.outputs.exists())

    def test_cli_plan_counts_selected_videos(self):
        weights = self.root / 'weights.pt'
        weights.write_bytes(b'fake weights')
        argv = ['privacy-blur-batch', str(self.inputs), '--video', 'left.mp4', '--video', 'right.mp4',
                '--face-weights', str(weights), '--output-dir', str(self.outputs), '--dry-run']
        stdout = io.StringIO()
        with patch('sys.argv', argv), patch('shutil.which', return_value='/mock/bin'), \
             patch('roboego_hand_vis.video_privacy_blur.resources.plan_workers', return_value={'workers': 8, 'device': 'cpu'}), \
             contextlib.redirect_stdout(stdout):
            self.assertEqual(main(), 0)
        self.assertIn('[plan] 2 video(s)', stdout.getvalue())
        report = json.loads(stdout.getvalue().splitlines()[-1])
        self.assertEqual(report['plan']['workers'], 2)
        self.assertFalse(self.outputs.exists())


if __name__ == '__main__':
    unittest.main()
