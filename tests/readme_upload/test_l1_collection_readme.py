import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import yaml

from roboego_hand_vis.upload_data.readme_upload import gen_readme
from roboego_hand_vis.upload_data.readme_upload._common import repo_id_for


class L1CollectionReadmeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'l1_kitchen_clean_task_1920x1456_30fps_2views'
        self.root.mkdir()
        self.project = {'license': 'cc-by-nc-4.0', 'authors': ['Example Author'], 'citation': {'year': 2026}}
        self.add_task('zzz', 60, 2.0, 'Second description', ['wash', 'clean'], [{'name': 'cup', 'attributes': {'color': 'red'}}, {'name': 'bowl'}])
        self.add_task('aaa', 30, 1.0, 'First description', ['clean'], [{'name': 'cup'}])

    def add_task(self, name, frames, duration, description, actions, objects):
        root = self.root/name
        (root/'meta').mkdir(parents=True)
        (root/'annotation').mkdir()
        (root/'calibrations').mkdir()
        (root/'annotation/episode_000000_structured.json').write_text('{}')
        (root/'calibrations/rectify_params.yaml').write_text('example: true')
        info = {'fps': 30, 'total_frames': frames, 'total_episodes': 1,
                'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/{feature_key}.parquet',
                'index_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/master_index.parquet',
                'video_path': 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                'features': {
                    'observation.images.head_left_rectified': {'dtype': 'video', 'shape': [1456, 1920, 3], 'fps': 30},
                    'observation.images.depth_head_left_rectified': {'dtype': 'video', 'shape': [1456, 1920, 1], 'fps': 30, 'info': {'video.is_depth_map': True}},
                    'eye_gaze_predict': {'dtype': 'float32', 'shape': [2], 'fps': 30},
                    'observation.imu': {'dtype': 'float32', 'shape': [6], 'fps': 200},
                    'observation.eeg': {'dtype': 'float32', 'shape': [8], 'fps': 200},
                    'observation.bad_frame_pose_hand': {'dtype': 'bool', 'shape': [1], 'fps': 30}}}
        (root/'meta/info.json').write_text(json.dumps(info))
        files = [p for p in root.rglob('*') if p.is_file()]
        card = {'dataset_name': self.root.name, 'unique_task_name': name, 'task_name': 'clean cup',
                'common_task_name': 'clean_task', 'task_description': description,
                'scene': {'type': 'indoor', 'category': 'household', 'name': 'kitchen'},
                'frame_count': frames, 'duration_s': duration, 'file_count': len(files),
                'size_bytes': sum(p.stat().st_size for p in files), 'actions': actions, 'objects': objects,
                'cameras': {'primary_camera': 'observation.images.head_left_rectified'}}
        (root/'meta/readme.json').write_text(json.dumps(card))

    def test_aggregation_counts_recordings_including_repeated_task_names(self):
        (self.root/'README.md').write_text('old card excluded from statistics')
        (self.root/'portal.yaml').write_text('old portal index excluded from statistics')
        with patch.dict('sys.modules', {'humancoin': None}):
            text = gen_readme.render_readme(self.root, self.project)
        self.assertIn('| Tasks | 2 |', text)
        self.assertIn('| Master frames | 90 |', text)
        self.assertIn('| Duration | 3.00 s', text)
        self.assertNotIn('| Episodes |', text)
        files = [p for p in self.root.rglob('*') if p.is_file() and p.name not in ('README.md', 'portal.yaml')]
        self.assertIn(f'| Files (excluding README.md and portal.yaml) | {len(files)} |', text)
        self.assertIn(f'| Size (excluding README.md and portal.yaml) | {sum(p.stat().st_size for p in files)/1024**2:.2f} MiB |', text)

    def test_cli_writes_collection_portal_yaml(self):
        video = self.root/'aaa/videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
        video.parent.mkdir(parents=True)
        video.touch()
        def fake_ffmpeg(command, **kwargs):
            self.assertEqual(command[command.index('-frames:v') + 1], '1')
            Path(command[-1]).write_bytes(b'preview JPEG')

        with patch.object(gen_readme.shutil, 'which', return_value='ffmpeg'), \
                patch.object(gen_readme.subprocess, 'run', side_effect=fake_ffmpeg):
            self.assertEqual(gen_readme.main(['--dataset', str(self.root)]), 0)
        self.assertEqual((self.root/'preview.jpg').read_bytes(), b'preview JPEG')
        metadata = yaml.safe_load((self.root/'portal.yaml').read_text(encoding='utf-8'))
        self.assertEqual(metadata['num_recordings'], 2)
        self.assertEqual(metadata['duration_s'], 3.0)
        self.assertEqual(metadata['task_name'], 'clean_task')
        self.assertEqual(metadata['example_task_description'], 'First description')
        self.assertEqual(metadata['preview_video'], video.relative_to(self.root).as_posix())
        self.assertEqual(metadata['preview_image'], 'preview.jpg')
        self.assertEqual(metadata['resolution'], '1920x1456')
        self.assertIn({'feature_key': 'eye_gaze_predict', 'dtype': 'float32',
                       'shape': [2], 'sample_rate_hz': 30}, metadata['sensor_streams'])

    def test_first_task_description_and_deduplicated_vocabulary(self):
        text = gen_readme.render_readme(self.root, self.project)
        self.assertIn('First description', text)
        self.assertNotIn('Second description', text)
        self.assertIn('**Atomic actions (deduplicated union):** clean, wash', text)
        self.assertEqual(text.count('| cup |'), 1)
        self.assertIn('| bowl |', text)
        self.assertNotIn('attributes', text)
        self.assertNotIn('| red |', text)

    def test_repo_and_task_loading_paths_and_l1_stream_policy(self):
        text = gen_readme.render_readme(self.root, self.project)
        self.assertIn('HumanCOIN/'+self.root.name, text)
        self.assertEqual(repo_id_for(self.root, 'HumanCOIN'), 'HumanCOIN/'+self.root.name)
        self.assertIn('viewer: false', text)
        self.assertIn('root=task_dir', text)
        self.assertIn('<unique_task_name>/data/', text)
        self.assertIn('<unique_task_name>/calibrations/rectify_params.yaml', text)
        self.assertIn('observation.imu', text)
        self.assertIn('eye_gaze_predict', text)
        self.assertNotIn('observation.eeg', text)
        self.assertNotIn('observation.bad_frame_pose_hand', text)
        self.assertNotIn('Samples (metadata)', text)
        self.assertNotIn('Supplementary depth metadata and valid-pixel masks are stored', text)

    def test_rejects_mixed_dataset_names_before_output(self):
        path = self.root/'zzz/meta/readme.json'
        data = json.loads(path.read_text());data['dataset_name'] = 'another_dataset'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'dataset_name differs'):
            gen_readme.render_readme(self.root, self.project)


if __name__ == '__main__':
    unittest.main()
