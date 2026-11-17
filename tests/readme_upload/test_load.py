from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from roboego_hand_vis.upload_data.readme_upload import check_load


class LoadCheckTests(unittest.TestCase):
    def test_reads_first_middle_last_and_all_declared_features(self):
        samples = []
        class Dataset:
            def __len__(self): return 5
            def __getitem__(self, index):
                samples.append(index)
                return {'timestamp': 0, 'observation.rgb': 'decoded'}
        loader = Mock(return_value=Dataset())
        check_load.validate_dataset_load(Path('/tmp/example'), 'HumanCOIN/demo',
            {'total_frames':5,'features':{'observation.rgb':{}}},loader)
        self.assertEqual(samples, [0,2,4])
        self.assertEqual(loader.call_args.kwargs['video_backend'], 'pyav')

    def test_decode_failure_is_not_suppressed(self):
        class Broken:
            def __len__(self): return 3
            def __getitem__(self, index): raise OSError('Missing video')
        with self.assertRaisesRegex(RuntimeError, 'Missing video'):
            check_load.validate_dataset_load(Path('/tmp/example'), 'HumanCOIN/demo',
                {'features':{}},Mock(return_value=Broken()))

    def test_empty_mismatched_and_missing_features_fail(self):
        for rows,info in [([],{'features':{}}),
                          ([{'timestamp':0}],{'total_frames':2,'features':{}}),
                          ([{'timestamp':0}],{'features':{'observation.eeg':{}}})]:
            with self.assertRaises(RuntimeError):
                check_load.validate_dataset_load(Path('/tmp/example'),'HumanCOIN/demo',info,Mock(return_value=rows))
