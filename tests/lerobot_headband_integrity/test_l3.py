import json
from pathlib import Path
import tempfile
import unittest
import wave

import pyarrow as pa
import pyarrow.parquet as pq

from roboego_hand_vis.lerobot_headband_integrity.common import Context
from roboego_hand_vis.lerobot_headband_integrity.L3.compute import (
    check_feature, export_report, build_report, HEADBAND_CAMERAS)
from roboego_hand_vis.lerobot_headband_integrity.L3.run import main
from roboego_hand_vis.lerobot_headband_integrity.L3.calibration_repair import repair_missing


class L3ChecksTest(unittest.TestCase):
    def test_missing_fourth_headband_camera_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'data/chunk-000/episode_000000'
            episode.mkdir(parents=True)
            (root / 'meta').mkdir()
            features = {}
            for camera in HEADBAND_CAMERAS:
                if camera == 'headband_bleft_rgb':
                    continue
                key = f'observation.images.{camera}'
                features[key] = dict(dtype='video')
                (episode / f'{key}.parquet').touch()
                video = root / 'videos/chunk-000' / key / 'episode_000000.mp4'
                video.parent.mkdir(parents=True)
                video.touch()
            (root / 'meta/info.json').write_text(json.dumps(dict(features=features, total_episodes=1,
                                                               total_frames=0, fps=30)))
            report = build_report(root)
            self.assertIn('headband/episode_000000/headband_bleft_rgb/present',
                          report['gate']['failed_checks'])
            self.assertIsNone(report['calibration_repair'])
            self.assertIn('dataset/imu_feature', report['gate']['failed_checks'])
            self.assertIn('chunk-000/episode_000000/observation.headband_imu0/present', report['gate']['failed_checks'])

    def test_calibration_repair_identity_and_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def session(batch, name, model):
                path = root / batch / name
                (path / 'meta').mkdir(parents=True)
                (path / 'data/chunk-000/episode_000000').mkdir(parents=True)
                (path / 'meta/info.json').write_text('{}')
                (path / 'meta/collect_info.jsonl').write_text(json.dumps(dict(
                    collector_id=name[:6], hardware=dict(ego_headband=model))))
                return path
            target = session('batch_1', 'L3-001-2026-10-19-18-10-53', 'V1.5')
            donor = session('batch_2', 'L3-001-2026-10-13-16-35-08', 'V1.5')
            wrong = session('batch_1', 'L3-002-2026-10-19-18-10-53', 'V1.5')
            camera = dict(timeshift_cam_imu=0., T_cn_cnm1=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], intrinsics=[500, 500, 320, 240], resolution=[640, 480],
                          distortion_coeffs=[0., 0., 0., 0.],
                          T_cam_imu=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            payload = {'ego_headband': {'camera-camchain': {f'cam{i}': camera for i in range(4)},
                       'camera_imu-camchain-imucam': {f'cam{i}': camera for i in range(4)},
                       'camera_imu-imu': {'imu0': dict(update_rate=200, accelerometer_noise_density=.01, accelerometer_random_walk=.001, gyroscope_noise_density=.01, gyroscope_random_walk=.001)}}}
            for path in (donor, wrong):
                (path / 'meta/calibrations.jsonl').write_text(json.dumps(dict(
                    episode_index=0, session='donor', calibrations=payload)))
            ctx = Context(target)
            self.assertEqual(repair_missing(ctx, [wrong])['status'], 'no_matching_donor')
            info = target / 'meta/collect_info.jsonl'
            original = info.read_text()
            info.write_text(original.replace('V1.5', 'V2'))
            self.assertEqual(repair_missing(ctx, [donor])['status'], 'no_matching_donor')
            info.write_text(original)
            receipt = repair_missing(ctx, [wrong, donor])
            self.assertEqual(receipt['status'], 'copied')
            repaired = json.loads((target / 'meta/calibrations.jsonl').read_text())
            self.assertEqual(repaired['session'], '2026-10-19-18-10-53')
            self.assertEqual(repaired['calibrations'], payload)
            self.assertEqual(repair_missing(ctx, [])['status'], 'existing')

    def test_audio_duration_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'data/chunk-000/episode_000000'
            episode.mkdir(parents=True)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps(dict(fps=10)))
            feature = 'observation.headband_microphone'
            pq.write_table(pa.table({'timestamp': [0., 0.5], 'num_samples': [50, 50]}),
                           episode / f'{feature}.parquet')
            path = root / 'audios/chunk-000' / feature / 'episode_000000.wav'
            path.parent.mkdir(parents=True)
            with wave.open(str(path), 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(100)
                audio.writeframes(b'\x01\x00' * 100)
            declaration = dict(dtype='audio', sample_rate=100)
            def checks(master):
                return {c['name']: c for c in check_feature(
                    Context(root, duration_tolerance_s=0.1), episode, feature, declaration, master)}
            self.assertTrue(checks([0., 0.9])['audio/duration']['passed'])
            self.assertFalse(checks([0., 1.9])['audio/duration']['passed'])
            path.write_bytes(path.read_bytes()[:-20])
            self.assertFalse(checks([0., 0.9])['audio/readable']['passed'])

    def test_non_imu_sensor_is_not_read(self):
        from unittest.mock import Mock
        ctx = Mock()
        for feature in ('observation.eeg', 'observation.left_wrist_emg',
                        'observation.left_wrist_imu', 'observation.right_wrist_imu',
                        'observation.eye_imu',
                        'observation.eye_gaze', 'observation.hand_pose', 'action'):
            self.assertEqual(check_feature(ctx, None, feature, {'dtype': 'float32'}, []), [])
        ctx.table.assert_not_called()

    def test_imu_jump(self):
        from unittest.mock import Mock
        ctx = Mock()
        declaration = dict(dtype='float32', shape=[1], names=['ax'])
        for values, expected in (([i * 0.01 for i in range(100)], True),
                                 ([i * 0.01 for i in range(99)] + [100.], False)):
            ctx.table.return_value = {'timestamp': list(range(100)), 'ax': values}
            checks = check_feature(ctx, None, 'observation.headband_imu0', declaration, [])
            jump = next(c for c in checks if c['name'] == 'jump')
            self.assertEqual(jump['passed'], expected)
            self.assertEqual(jump['level'], 'fail')
            self.assertEqual(jump['statistics']['jump_columns'], [] if expected else ['ax'])

    def test_native_rate_and_corrupt_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'data/chunk-000/episode_000000'
            episode.mkdir(parents=True)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps({'features': {}, 'fps': 30}))
            declaration = dict(dtype='float32', shape=[1], names=['x'], tolerance_s=0.02)
            path = episode / 'observation.headband_imu0.parquet'
            pq.write_table(pa.table({'timestamp': [0., 0.01, 0.02, 0.03], 'x': [1., 2., 3., 4.]}), path)
            checks = check_feature(Context(root), episode, 'observation.headband_imu0', declaration, [0., 0.03])
            self.assertTrue(all(c['passed'] for c in checks))
            pq.write_table(pa.table({'timestamp': [0., 0., 0.02], 'x': [1., 2., 3.]}), path)
            checks = check_feature(Context(root), episode, 'observation.headband_imu0', declaration, [0., 0.03])
            timestamp = next(c for c in checks if c['name'] == 'timestamps')
            self.assertFalse(timestamp['passed'])
            self.assertEqual(timestamp['level'], 'warning')
            self.assertTrue(any(c['name'] == 'finite_values' for c in checks))
            for times in ([1., 0.], [float('nan'), float('inf')], []):
                pq.write_table(pa.table({'timestamp': times, 'x': [1.] * len(times)}), path)
                checks = check_feature(Context(root), episode, 'observation.headband_imu0', declaration, [0., 0.03])
                timestamp = next(c for c in checks if c['name'] == 'timestamps')
                self.assertFalse(timestamp['passed'])
                self.assertEqual(timestamp['level'], 'warning' if times == [1., 0.] else 'fail')
                self.assertTrue(next(c for c in checks if c['name'] == 'finite_values')['passed'])
            pq.write_table(pa.table({'timestamp': [0., 0.01], 'x': [float('nan'), 2.]}), path)
            checks = check_feature(Context(root), episode, 'observation.headband_imu0', declaration, [0., 0.01])
            self.assertFalse(next(c for c in checks if c['name'] == 'finite_values')['passed'])

    def test_failure_report_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text('{}')
            self.assertEqual(main(['--source', str(root), '--dry-run']), 0)
            self.assertFalse((root / 'annotation').exists())
            status = export_report(root)
            self.assertFalse(status['passed'])
            report = json.loads((root / 'annotation/quality_report.json').read_text())
            self.assertIn('dataset/readable', report['gate']['failed_checks'])
            self.assertEqual(list((root / 'annotation').glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
