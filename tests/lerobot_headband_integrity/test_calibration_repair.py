import json
from pathlib import Path
import tempfile
import unittest

import yaml

from roboego_hand_vis.lerobot_headband_integrity.L1.calibration_repair import repair_sessions, complete_vio_file


class RepairTests(unittest.TestCase):
    def test_same_device_passed_donor_copied_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            donor = root / 'batch_1/cp_0002_session_20261009_120000'
            target = root / 'batch_2/cp_0002_session_20261010_120000'
            other = root / 'batch_2/cp_0003_session_20261010_120000'
            for s in (donor, target, other):
                (s / 'calibrations').mkdir(parents=True)
                (s / 'calibrations/camera-camchain.yaml').write_text('placeholder')
            imu = dict(imu0=dict(accelerometer_noise_density=.01, accelerometer_random_walk=.001,
                                gyroscope_noise_density=.001, gyroscope_random_walk=.0001, update_rate=200))
            cameras = {f'cam{i}': dict(camera_model='pinhole', distortion_model='equidistant',
                                      intrinsics=[500,500,300,300], distortion_coeffs=[0]*4, resolution=[1920,1088],
                                      timeshift_cam_imu=0., T_cam_imu=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) for i in range(2)}
            for name, data in (('camera_imu-imu.yaml', imu), ('camera_imu-camchain-imucam.yaml', cameras)):
                path = donor / 'calibrations' / name
                path.write_text(yaml.safe_dump(data))
                self.assertTrue(complete_vio_file(path)[0])
                for recipient in (target, other):
                    (recipient / 'calibrations' / name).write_text('placeholder')
            camera = donor / 'calibrations/camera_imu-camchain-imucam.yaml'
            # Complete files without QC success are ineligible.
            first = repair_sessions([target], root)
            self.assertFalse(first[str(target.resolve())]['applied'])
            (donor / 'annotation').mkdir()
            report = donor / 'annotation/quality_report.json'
            report.write_text(json.dumps(dict(passed=True, source_session=str(donor.resolve()))))
            result = repair_sessions([target, other], root)
            fixed = result[str(target.resolve())]
            self.assertTrue(fixed['applied'])
            for copied in fixed['files'].values():
                self.assertEqual(Path(copied['backup']).read_text(), 'placeholder')
            self.assertEqual((target / 'calibrations/camera_imu-camchain-imucam.yaml').read_bytes(), camera.read_bytes())
            self.assertFalse(result[str(other.resolve())]['applied'])
            self.assertEqual((target / 'calibrations/camera-camchain.yaml').read_text(), 'placeholder')
            self.assertTrue(complete_vio_file(target / 'calibrations/camera_imu-imu.yaml')[0])
            self.assertEqual(repair_sessions([target], root), {})
            self.assertFalse(complete_vio_file(other / 'calibrations/camera-camchain.yaml')[0])


if __name__ == '__main__':
    unittest.main()
