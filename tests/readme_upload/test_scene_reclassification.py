import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[2] / 'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location('scene_reclassify_test_module', DIRECTORY / 'reclassify_scenes_qwen.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TAXONOMY = json.loads((DIRECTORY / 'scene_taxonomy_v2.json').read_text(encoding='utf-8'))


class SceneReclassificationTests(unittest.TestCase):
    def test_only_commercial_can_add_area_and_known_owners_are_enforced(self):
        module.validate_taxonomy(TAXONOMY)
        module.validate_decision(dict(category='commercial_service', area='shoe_repair_shop', reason='Visible shop counter'), TAXONOMY)
        for category, area in [('household', 'home_gym'), ('professional_work', 'meeting_room'),
                               ('industrial_production', 'new_factory'), ('public_space', 'sports_equipment_area'),
                               ('commercial_service', 'kitchen'), ('commercial_service', 'food_service_area'),
                               ('unknown', 'unknown'), ('other', 'unidentified')]:
            with self.subTest(category=category, area=area), self.assertRaises(ValueError):
                module.validate_decision(dict(category=category, area=area, reason='Evidence'), TAXONOMY)

    def test_targets_use_frame_metadata_or_duration_without_frame_scan(self):
        targets = module.sample_targets(dict(frames=100, fps=10, duration=10))
        self.assertEqual([row['frame_index'] for row in targets], [10, 50, 90])
        self.assertEqual([row['seek_seconds'] for row in targets], [1, 5, 9])
        fallback = module.sample_targets(dict(frames=None, fps=None, duration=10))
        self.assertEqual([row['seek_seconds'] for row in fallback], [1, 5, 9])
        with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(
            dict(streams=[dict(avg_frame_rate='0/0', r_frame_rate='30/1', nb_frames='N/A')],
                 format=dict(duration='10'))))) as run:
            info = module.probe_video(Path('example.mp4'), 'ffprobe', 10)
        self.assertEqual(info['fps'], 30)
        self.assertIsNone(info['frames'])
        self.assertNotIn('-count_frames', run.call_args.args[0])
        self.assertNotIn('-show_frames', run.call_args.args[0])

    def test_input_seek_extracts_one_frame_per_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            def fake(command, **kwargs):
                Path(command[-1]).write_bytes(b'jpeg-test')
                self.assertLess(command.index('-ss'), command.index('-i'))
                self.assertEqual(command[command.index('-frames:v') + 1], '1')
            with patch.object(module.subprocess, 'run', side_effect=fake) as run:
                frames = module.extract_frames(Path('video.mp4'), module.sample_targets(
                    dict(frames=100, fps=10, duration=10)), folder, 'ffmpeg', 960, 20)
            self.assertEqual(len(frames), 3)
            self.assertEqual(run.call_count, 3)

    def fixture(self, temporary):
        root = Path(temporary) / 'data'
        session = root / 'batch_0000040' / 'any-session-name'
        annotation = session / 'annotation/episode_000000_structured.json'
        annotation.parent.mkdir(parents=True)
        raw = (b'{\r\n "task_description": "Cook a meal", "keep": 1.00, '
               b'"scene_classification": {"category": "other", "area": "other", '
               b'"original_description": "A kitchen"}, "nested":{"area":"keep"}\r\n}\r\n')
        annotation.write_bytes(raw)
        video = session / 'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fixture-video')
        args = SimpleNamespace(root=root, report_dir=Path(temporary) / 'reports', video_key=None,
            context='', model='qwen3.7-plus', width=960, ffprobe='ffprobe', ffmpeg='ffmpeg',
            media_timeout=20, timeout=30, max_tokens=4096, dry_run=False, apply=False, retry_failed=False,
            batches=None, start=40, end=40, only_other=False, limit=0)
        return annotation, raw, args

    @staticmethod
    def frames(video, targets, folder, *args):
        files = [folder / f'{i}.jpg' for i in range(3)]
        for file in files:
            file.write_bytes(b'jpeg-test')
        return files

    @staticmethod
    def response():
        return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
            dict(category='household', area='kitchen', reason='Visible cooking counter'))))], usage=dict(total_tokens=123))

    def test_cached_model_result_can_be_applied_without_rebilling_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation, raw, args = self.fixture(temporary)
            endpoint = SimpleNamespace(settings=lambda model: ('secret', 'https://example.invalid', 'mock'))
            with patch.object(module, 'probe_video', return_value=dict(frames=100, fps=10, duration=10)), \
                 patch.object(module, 'extract_frames', side_effect=self.frames), \
                 patch.object(module, 'request_model', return_value=(self.response(), 'mock')) as request:
                result = module.process(annotation, args, TAXONOMY, endpoint)
                self.assertFalse(result['applied'])
                self.assertEqual(annotation.read_bytes(), raw)
                args.apply = True
                applied = module.process(annotation, args, TAXONOMY, endpoint)
                self.assertTrue(applied['reused'])
                self.assertTrue(applied['applied'])
                self.assertEqual(request.call_count, 1)
                again = module.process(annotation, args, TAXONOMY, endpoint)
                self.assertFalse(again['changed'])
            expected = raw.replace(b'"category": "other"', b'"category": "household"').replace(
                b'"area": "other"', b'"area": "kitchen"')
            self.assertEqual(annotation.read_bytes(), expected)
            self.assertEqual((Path(applied['state']) / 'annotation_before.json').read_bytes(), raw)
            self.assertEqual(module.discover(args), [annotation])

    def test_unknown_network_outcome_requires_explicit_retry_and_never_edits_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation, raw, args = self.fixture(temporary)
            args.apply = True
            endpoint = SimpleNamespace(settings=lambda model: ('secret', 'https://example.invalid', 'mock'))
            with patch.object(module, 'probe_video', return_value=dict(frames=100, fps=10, duration=10)), \
                 patch.object(module, 'extract_frames', side_effect=self.frames), \
                 patch.object(module, 'request_model', side_effect=TimeoutError('unknown outcome')) as request:
                with self.assertRaises(TimeoutError):
                    module.process(annotation, args, TAXONOMY, endpoint)
                with self.assertRaisesRegex(RuntimeError, 'may have been billed'):
                    module.process(annotation, args, TAXONOMY, endpoint)
                self.assertEqual(request.call_count, 1)
                self.assertEqual(annotation.read_bytes(), raw)
            args.retry_failed = True
            with patch.object(module, 'probe_video', return_value=dict(frames=100, fps=10, duration=10)), \
                 patch.object(module, 'extract_frames', side_effect=self.frames), \
                 patch.object(module, 'request_model', return_value=(self.response(), 'mock')):
                self.assertTrue(module.process(annotation, args, TAXONOMY, endpoint)['applied'])

    def test_truncated_and_invalid_model_outputs_fail_closed(self):
        with self.assertRaises(ValueError):
            module.parse_response(dict(choices=[]), TAXONOMY)
        with self.assertRaises(ValueError):
            module.parse_response(dict(choices=[dict(finish_reason='stop', message=dict(content=None))]), TAXONOMY)
        with self.assertRaises(ValueError):
            module.parse_response(dict(choices=[dict(finish_reason='length')]), TAXONOMY)
        with self.assertRaises(ValueError):
            module.parse_response(dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(
                dict(category='household', area='invented_room', reason='Guess'))))]), TAXONOMY)

    def test_rgb_fallback_and_duplicate_episode_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation, _, _ = self.fixture(temporary)
            video = module.choose_video(annotation, None)
            band = video.parent.parent / 'observation.images.headband_left_rectified' / video.name
            band.parent.mkdir()
            video.rename(band)
            self.assertEqual(module.choose_video(annotation, None), band)
            duplicate = band.parents[2] / 'chunk-001' / band.parent.name / band.name
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(b'other chunk')
            with self.assertRaisesRegex(ValueError, 'multiple chunks'):
                module.choose_video(annotation, None)

    def test_request_contains_three_images_and_omits_gateway_specific_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, args = self.fixture(temporary)
            files = []
            for index in range(3):
                file = Path(temporary) / f'{index}.jpg'
                file.write_bytes(b'jpeg')
                files.append(file)
            endpoint = SimpleNamespace(settings=lambda model: ('secret', 'https://example.invalid/v1/chat/completions', 'mock'),
                                       gateway=lambda model: True)
            import io
            with patch.object(module.urllib.request, 'urlopen') as urlopen:
                urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(self.response()).encode())
                module.request_model(dict(task_description='task', original_description='scene'), files, args, endpoint, 'rules')
                request = urlopen.call_args.args[0]
                body = json.loads(request.data)
                self.assertEqual(body['model'], 'qwen3.7-plus')
                self.assertNotIn('chat_template_kwargs', body)
                content = body['messages'][1]['content']
                self.assertEqual(sum(row['type'] == 'image_url' for row in content), 3)
                self.assertNotIn('old_category', content[0]['text'])


if __name__ == '__main__':
    unittest.main()
