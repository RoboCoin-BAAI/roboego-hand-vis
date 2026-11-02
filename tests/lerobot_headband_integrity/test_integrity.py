"""New-format regression tests, independent of the original raw checker."""
import hashlib
import json
import math
import struct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from roboego_hand_vis.lerobot_headband_integrity.L1 import compute, run
from roboego_hand_vis.lerobot_headband_integrity.L1.audio import inspect_wav
from roboego_hand_vis.lerobot_headband_integrity.common import CAMERAS, IMU_COLUMNS
from roboego_hand_vis.lerobot_headband_integrity.L1.timestamps import alignment


class IntegrityTests(unittest.TestCase):
    def test_camera_nearest_alignment_over_tolerance_is_warning(self):
        self.write_table('observation.images.head_right', {'timestamp': [0., .06, .12]})
        report = self.report()
        self.assertTrue(report['passed'], report['gate'])
        self.assertTrue(any(k.endswith('/nearest_alignment') for k in report['gate']['warnings']))
        self.assertFalse(any(k.endswith('/nearest_alignment') for k in report['gate']['failed_checks']))

    def test_audio_duration_default_accepts_five_seconds_not_more(self):
        path = self.root / 'audio/audio.wav'
        for duration, expected in ((4.1, True), (5.1, True), (5.2, False)):
            with self.subTest(duration=duration):
                with wave.open(str(path), 'wb') as wav:
                    wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                    wav.writeframes(b'\0\0' * round(duration * 8000))
                report = self.report()
                check = next(c for c in report['checks'] if c['name'] == 'audio/video_duration')
                self.assertEqual(check['passed'], expected)
                self.assertEqual(check['statistics']['limit_s'], 5.)

    def test_video_validation_cache_reused_and_invalidated(self):
        probe_path = 'roboego_hand_vis.lerobot_headband_integrity.L1.video.subprocess.run'
        with patch(probe_path, side_effect=self.probe) as probe:
            self.assertTrue(compute.build_report(self.root)['passed'])
            self.assertEqual(probe.call_count, 4)
        with patch(probe_path, side_effect=AssertionError('must not decode again')):
            report = compute.build_report(self.root)
        self.assertTrue(report['passed'])
        self.assertTrue(all(c['statistics']['reused_validation'] for c in report['checks'] if c['name'].startswith('video/') and c['name'].endswith('/decodable')))
        video = self.root / 'videos/chunk-000/observation.images.head_left/episode_000000.mp4'
        video.write_bytes(video.read_bytes() + b'changed')
        with patch(probe_path, side_effect=self.probe) as probe:
            compute.build_report(self.root)
            self.assertEqual(probe.call_count, 1)
        self.write_table('observation.images.head_left', {'timestamp': [0., .03, .07]})
        with patch(probe_path, side_effect=self.probe) as probe:
            compute.build_report(self.root)
            self.assertEqual(probe.call_count, 1)
        with patch(probe_path, side_effect=self.probe) as probe:
            compute.build_report(self.root, full_video_check=True)
            self.assertEqual(probe.call_count, 4)

    def test_converter_receipt_avoids_first_qc_decode(self):
        from roboego_hand_vis.lerobot_headband_integrity.common import Context
        from roboego_hand_vis.lerobot_headband_integrity.L1.video_cache import identity
        ctx = Context(self.root)
        receipt = dict(version=1, producer='l1_converter', videos={})
        for camera in CAMERAS:
            video = ctx.video(self.ep, camera)
            receipt['videos'][video.relative_to(self.root).as_posix()] = dict(
                identity=identity(video, self.ep / f'observation.images.{camera}.parquet'),
                stream=dict(width=12, height=8, nb_read_frames='3', avg_frame_rate='30/1'), decoder_errors='')
        (self.root / 'meta/l1_video_validation.json').write_text(json.dumps(receipt))
        with patch('roboego_hand_vis.lerobot_headband_integrity.L1.video.subprocess.run', side_effect=AssertionError('converter already validated')):
            self.assertTrue(compute.build_report(self.root)['passed'])

    def test_invalid_vio_calibration_blocks(self):
        path = self.root / 'calibrations/camera-camchain.yaml'
        original = yaml.safe_load(path.read_text())
        raw = {}
        # Deliberately swap outer camera order: use topics, not assumed indices.
        for i, (topic, name) in enumerate((('left', 'head_left'), ('right', 'head_right'),
                                          ('bright', 'head_right_outer0'), ('bleft', 'head_left_outer0'))):
            raw[f'cam{i}'] = dict(original[name], rostopic=f'/fisheye/{topic}/image_raw/compressed')
            if i:
                raw[f'cam{i}']['T_cn_cnm1'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        path.write_text(yaml.safe_dump(raw))
        (path.parent / 'camera_imu-camchain-imucam.yaml').write_text('not calibrated')
        (path.parent / 'camera_imu-imu.yaml').write_text('')
        report = self.report()
        self.assertFalse(report['passed'])
        self.assertTrue(any('repair_structure' in k for k in report['gate']['failed_checks']))
        self.assertTrue(any('content_present' in k for k in report['gate']['warnings']))
        raw['cam1']['intrinsics'][0] = -1
        path.write_text(yaml.safe_dump(raw))
        self.assertFalse(self.report()['passed'])

    def test_empty_primary_calibration_still_blocks(self):
        (self.root / 'calibrations/camera-camchain.yaml').write_text('')
        self.assertFalse(self.report()['passed'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "batch/session"
        self.ep = self.root / "data/chunk-000/episode_000000"
        self.ep.mkdir(parents=True)
        (self.root / "meta").mkdir()
        features = {f"observation.images.{c}": {"shape": [8, 12, 3], "fps": 30} for c in CAMERAS}
        features["observation.imu"] = {"shape": [6]}
        (self.root / "meta/info.json").write_text(json.dumps({"features": features, "fps": 30, "total_episodes": 1}))
        self.write_table("master_index", {"timestamp": [0., 1/30, 2/30]})
        for camera in CAMERAS:
            offset = 1/30 if camera in ("head_right", "head_right_outer0") else 0
            self.write_table(f"observation.images.{camera}", {"timestamp": [offset + i/30 for i in range(3)]})
            path = self.root / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"mocked-video")
        self.write_table("observation.imu", {"timestamp": [-0.01 + i * .005 for i in range(26)],
                         **{c: [math.sin(i / 10 + j) for i in range(26)] for j, c in enumerate(IMU_COLUMNS)}})
        (self.root / "audio").mkdir()
        with wave.open(str(self.root / "audio/audio.wav"), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 800)
        (self.root / "calibrations").mkdir()
        vio_camera = dict(camera_model='pinhole', distortion_model='equidistant',
                          intrinsics=[10,10,6,4], distortion_coeffs=[0]*4, resolution=[12,8],
                          timeshift_cam_imu=0., T_cam_imu=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        (self.root / 'calibrations/camera_imu-camchain-imucam.yaml').write_text(
            yaml.safe_dump(dict(cam0=vio_camera, cam1=vio_camera)))
        (self.root / 'calibrations/camera_imu-imu.yaml').write_text(yaml.safe_dump(dict(imu0=dict(
            accelerometer_noise_density=.01, accelerometer_random_walk=.001,
            gyroscope_noise_density=.001, gyroscope_random_walk=.0001, update_rate=200))))
        calibration = {c: dict(intrinsics=[10, 10, 6, 4], resolution=[12, 8], distortion_coeffs=[0, 0, 0, 0]) for c in CAMERAS}
        (self.root / "calibrations/camera-camchain.yaml").write_text(yaml.safe_dump(calibration))

    def write_table(self, name, data):
        pq.write_table(pa.table(data), self.ep / f"{name}.parquet")

    def probe(self, *args, **kwargs):
        return SimpleNamespace(stdout=json.dumps({"streams": [{"width": 12, "height": 8,
            "nb_read_frames": "3", "avg_frame_rate": "30/1"}]}), stderr="")

    def report(self):
        with patch("roboego_hand_vis.lerobot_headband_integrity.L1.video.subprocess.run", side_effect=self.probe):
            return compute.build_report(self.root)

    def test_good_dataset_passes_with_boundary_warning(self):
        report = self.report()
        self.assertTrue(report["passed"], report["gate"])
        self.assertTrue(any("boundary_coverage" in k for k in report["gate"]["warnings"]))

    def test_frame_count_mismatch_fails(self):
        self.write_table("observation.images.head_left", {"timestamp": [0., 1/30]})
        self.assertTrue(any("timestamp_count" in k for k in self.report()["gate"]["failed_checks"]))

    def test_duplicate_and_nonfinite_times_fail_without_crashing(self):
        for values in ([0., 0., .1], [0., float("nan"), .1]):
            self.write_table("observation.images.head_left", {"timestamp": values})
            report = self.report()
            self.assertFalse(report["passed"])
            json.dumps(report, allow_nan=False)

    def test_missing_audio_is_modality_failure(self):
        (self.root / "audio/audio.wav").unlink()
        report = self.report()
        self.assertIn("audio/readable", report["gate"]["failed_checks"])
        self.assertTrue(any(k["name"].startswith("calibration/") for k in report["checks"]))

    def test_aligned_wav_length_repaired_without_payload_change(self):
        path = self.root / "audio/audio.wav"
        path.write_bytes(path.read_bytes()[:-100])
        before = path.read_bytes()
        info = inspect_wav(path)
        self.assertEqual(info["readable_bytes"], info["expected_bytes"])
        self.assertTrue(info['wav_header_repair']['applied'])
        self.assertEqual(path.read_bytes()[44:], before[44:])
        self.assertFalse(inspect_wav(path)['wav_header_repair']['applied'])
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_placeholder_wav_header_is_fixed_before_duration_check(self):
        path = self.root / 'audio/audio.wav'
        original = path.read_bytes()
        raw = bytearray(original)
        struct.pack_into('<I', raw, 4, 0xffffffff)
        struct.pack_into('<I', raw, 40, 0x80000000)
        path.write_bytes(raw)
        report = self.report()
        self.assertTrue(report['passed'], report['gate'])
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(any('header_repaired' in k for k in report['gate']['warnings']))

    def test_unaligned_payload_is_not_repaired(self):
        path = self.root / 'audio/audio.wav'
        path.write_bytes(path.read_bytes()[:-1])
        before = path.read_bytes()
        info = inspect_wav(path)
        self.assertFalse(info['wav_header_repair']['applied'])
        self.assertNotEqual(info['readable_bytes'], info['expected_bytes'])
        self.assertEqual(path.read_bytes(), before)

    def test_all_zero_imu_fails(self):
        self.write_table("observation.imu", {"timestamp": [0., .005], **{c: [0., 0.] for c in IMU_COLUMNS}})
        self.assertTrue(any("all_zero" in k for k in self.report()["gate"]["failed_checks"]))

    def test_internal_imu_gap_fails(self):
        self.write_table("observation.imu", {"timestamp": [-1., 1.], **{c: [1., 2.] for c in IMU_COLUMNS}})
        self.assertTrue(any("coverage/" in k for k in self.report()["gate"]["failed_checks"]))

    def test_missing_camera_entry_does_not_block_presence_check(self):
        path = self.root / "calibrations/camera-camchain.yaml"
        data = yaml.safe_load(path.read_text())
        del data["head_right"]
        path.write_text(yaml.safe_dump(data))
        self.assertTrue(self.report()["passed"])

    def test_frozen_imu_blocks(self):
        self.write_table("observation.imu", {"timestamp": [-.01 + i * .005 for i in range(26)],
                         **{c: [1.] * 26 for c in IMU_COLUMNS}})
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertTrue(any("frozen" in k for k in report["gate"]["failed_checks"]))

    def test_nonincreasing_imu_times_warn_and_coverage_still_checked(self):
        values = [-.01 + i * .005 for i in range(26)]
        values[1], values[2] = values[2], values[1]
        values[4] = values[3]
        self.write_table('observation.imu', {'timestamp': values,
                         **{c: [math.sin(i / 10 + j) for i in range(26)] for j,c in enumerate(IMU_COLUMNS)}})
        report = self.report()
        self.assertTrue(report['passed'], report['gate'])
        self.assertTrue(any(k.startswith('imu/') and k.endswith('/timestamps') for k in report['gate']['warnings']))
        self.assertTrue(any('/coverage/' in c['name'] for c in report['checks']))
        values[0] = float('nan')
        self.write_table('observation.imu', {'timestamp': values,
                         **{c: [float(i) for i in range(26)] for c in IMU_COLUMNS}})
        report = self.report()
        self.assertTrue(any(k.startswith('imu/') and k.endswith('/timestamps') for k in report['gate']['failed_checks']))

    def test_disordered_imu_cannot_hide_large_coverage_gap(self):
        self.write_table('observation.imu', {'timestamp': [1., -1., -1.],
                         **{c: [1., 2., 3.] for c in IMU_COLUMNS}})
        report = self.report()
        self.assertTrue(any('/coverage/' in k for k in report['gate']['failed_checks']))

    def test_severe_imu_jump_blocks_and_small_spike_passes(self):
        def check_spike(amplitude):
            values = [i * .001 for i in range(26)]
            values[12] += amplitude
            self.write_table('observation.imu', {
                'timestamp': [-.01 + i * .005 for i in range(26)],
                **{c: values for c in IMU_COLUMNS}})
            return self.report()
        report = check_spike(30.)
        self.assertFalse(report['passed'])
        self.assertTrue(any(k.endswith('/jump') for k in report['gate']['failed_checks']))
        report = check_spike(10.)
        self.assertTrue(report['passed'], report['gate'])

    def test_report_is_only_written_file(self):
        def snapshot():
            return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.root.rglob("*") if p.is_file() and "annotation" not in p.parts}
        before = snapshot()
        with patch("roboego_hand_vis.lerobot_headband_integrity.L1.video.subprocess.run", side_effect=self.probe):
            item = compute.export_report(self.root)
        self.assertTrue(item["passed"])
        self.assertEqual(snapshot(), before)
        self.assertTrue((self.root / "annotation/quality_report.json").is_file())

    def test_discovery_ignores_staging_and_dry_run_writes_nothing(self):
        hidden = self.root.parent / ".session.partial-123/meta"
        hidden.mkdir(parents=True)
        (hidden / "info.json").write_text("{}")
        self.assertEqual(run.discover(self.root.parent), [self.root])
        self.assertEqual(run.main(["--source", str(self.root.parent), "--dry-run"]), 0)
        self.assertFalse((self.root / "annotation").exists())

    def test_nonoverlap_fails_alignment(self):
        stats = alignment([0., .03], [2., 2.03], .02)
        self.assertEqual(stats["over_tolerance_rate"], 1.)

    def test_real_process_worker_reports_missing_probe(self):
        code = run.main(["--source", str(self.root.parent), "--workers", "1", "--ffprobe", "not-an-installed-probe-123"])
        self.assertEqual(code, 1)
        report = json.loads((self.root / "annotation/quality_report.json").read_text(encoding="utf-8"))
        self.assertTrue(any(k.startswith("video/") for k in report["gate"]["failed_checks"]))


if __name__ == "__main__":
    unittest.main()
