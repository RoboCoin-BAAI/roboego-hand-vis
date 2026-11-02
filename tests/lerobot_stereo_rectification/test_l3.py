import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import Encoder
from roboego_hand_vis.lerobot_stereo_rectification.L3.compute import load_geometry, process_session


def calibration(index=0, baseline=-0.1):
    transform = np.eye(4)
    transform[0, 3] = baseline
    cam = dict(camera_model='pinhole', distortion_model='equidistant',
               intrinsics=[32., 32., 32., 24.], distortion_coeffs=[0., 0., 0., 0.], resolution=[64, 48])
    return dict(episode_index=index, calibrations={'ego_headband': {'camera-camchain': {
        'cam0': dict(cam, rostopic='/fisheye/left/image_raw/compressed'),
        'cam1': dict(cam, rostopic='/fisheye/right/image_raw/compressed', T_cn_cnm1=transform.tolist()),
        'cam2': dict(cam, rostopic='/fisheye/bright/image_raw/compressed'),
    }}})


class L3Tests(unittest.TestCase):
    def test_per_episode_jsonl_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'meta').mkdir()
            path = root / 'meta/calibrations.jsonl'
            rows = [calibration(), calibration(1, -0.2)]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8-sig')
            geometry, source = load_geometry(root, 'episode_000001')
            self.assertAlmostEqual(geometry[2][0, 3], -0.2)
            self.assertEqual(source['episode_index'], 1)
            with self.assertRaises(ValueError):
                load_geometry(root, 'episode_000002')
            path.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                load_geometry(root, 'episode_000000')
            bad = calibration()
            bad['calibrations']['ego_headband']['camera-camchain']['cam1']['distortion_model'] = 'radtan'
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'equidistant'):
                load_geometry(root, 'episode_000000')

    def test_l3_video_outputs_and_reuse(self):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / 'data/chunk-000/episode_000000'
            episode.mkdir(parents=True)
            (root / 'meta').mkdir()
            features = {}
            originals = {}
            for side, count in (('left', 4), ('right', 3)):
                key = f'observation.images.headband_{side}_rgb'
                features[key] = dict(dtype='video', shape=[48, 64, 3], fps=30)
                timestamps = episode / f'{key}.parquet'
                pq.write_table(pa.table({'timestamp': [0.] * count}), timestamps)
                video = root / 'videos/chunk-000' / key / 'episode_000000.mp4'
                encoder = Encoder(video, (64, 48), 30, ffmpeg)
                try:
                    for i in range(count):
                        encoder.write(np.full((48,64,3), 40+i*30, dtype=np.uint8))
                finally:
                    encoder.close()
                originals[timestamps] = timestamps.read_bytes()
                originals[video] = video.read_bytes()
            (root / 'meta/info.json').write_text(json.dumps(dict(features=features, fps=30)))
            cal = root / 'meta/calibrations.jsonl'
            cal.write_text(json.dumps(calibration()))
            originals[cal] = cal.read_bytes()
            process_session(root, dict(dry_run=True))
            self.assertFalse((root / 'calibrations').exists())
            result, = process_session(root, dict(ffmpeg=ffmpeg))
            self.assertEqual(result['frame_counts'], {'left': 4, 'right': 3})
            manifest = json.loads((root / 'calibrations/dataset_manifest.json').read_text())
            self.assertEqual(manifest['stereo_pair'], ['headband_left_rgb', 'headband_right_rgb'])
            self.assertFalse(manifest['synchronization_checked'])
            info = json.loads((root / 'meta/info.json').read_text())
            self.assertEqual(info['total_videos'], 4)
            for side in ('left', 'right'):
                key = f'observation.images.headband_{side}_rgb'
                self.assertIn(key+'_rectified', info['features'])
                self.assertEqual((episode / f'{key}.parquet').read_bytes(),
                                 (episode / f'{key}_rectified.parquet').read_bytes())
            self.assertTrue(all(path.read_bytes() == value for path, value in originals.items()))
            with patch.object(Encoder, 'write', side_effect=AssertionError('must reuse')):
                reused, = process_session(root, dict(ffmpeg=ffmpeg))
                self.assertTrue(reused['skipped_existing'])


if __name__ == '__main__':
    unittest.main()
