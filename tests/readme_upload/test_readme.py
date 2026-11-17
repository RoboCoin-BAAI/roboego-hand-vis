import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

from roboego_hand_vis.upload_data.readme_upload import gen_readme
from roboego_hand_vis.upload_data.readme_upload.upload2hub import repo_id_for


class ReadmeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'local-session'
        (self.root/'meta').mkdir(parents=True)
        self.description = {
            'dataset_name': 'named_dataset', 'task_name': 'assemble_sandwich',
            'task_description': 'Assemble a sandwich.', 'duration_s': 1,
            'size_bytes': 99999999, 'file_count': 99999,
            'cameras': {'primary_camera':'observation.rgb', 'items':[]},
        }
        self.info = {
            'total_frames':30, 'total_episodes':1,'fps':30,
            'codebase_version':'v3.1','splits':{'train':'0:1'},
            'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
            'features': {
                'observation.rgb': {'dtype':'video','shape':[2,3,3],'fps':30},
                'observation.depth': {'dtype':'video','shape':[2,3,1],'fps':30,
                    'info':{'video.is_depth_map':True},
                    'depth_storage':{'unit':'millimeter','lossless':False}},
                'observation.mic': {'dtype':'audio','shape':[320],'sample_rate':16000},
                'observation.eeg': {'dtype':'float32','shape':[32],'fps':250,
                    'window':[-0.033,0]},
            },
        }
        self.project = {'license':'cc-by-nc-4.0','authors':['Author One','Author Two'],
                        'citation':{'title':'HumanCOIN','year':2026}}
        self.save()
        (self.root/'meta/tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'做三明治'},ensure_ascii=False)+'\n')

    def save(self):
        for name,data in [('readme',self.description),('info',self.info)]:
            (self.root/f'meta/{name}.json').write_text(json.dumps(data))

    def test_access_request_form(self):
        import yaml
        for tier in ('l1', 'l2', 'l3'):
            self.description['dataset_name'] = tier + '_example'
            self.save()
            text = gen_readme.render_readme(self.root, self.project)
            frontmatter = yaml.safe_load(text.split('---', 2)[1])
            self.assertIs(frontmatter['gated'], True)
            self.assertEqual(frontmatter['extra_gated_fields'], {
                'Name': 'text', 'Email': 'text', 'Intended use': 'text'})
            self.assertIn('name, email address and intended use', frontmatter['extra_gated_prompt'])
            self.assertEqual(frontmatter['license'], 'cc-by-nc-4.0')
            self.assertIs(frontmatter['viewer'], tier != 'l1')

    def test_generation_does_not_require_loader(self):
        with patch.dict('sys.modules', {'humancoin':None}):
            text = gen_readme.render_readme(self.root, self.project)
        self.assertIn('# named_dataset', text)

    def test_current_sdk_instructions(self):
        text = gen_readme.render_readme(self.root, self.project)
        self.assertIn('https://github.com/FlagOpen/HumanCOIN', text)
        self.assertIn('pip install -U "humancoin>=0.1.1"', text)
        self.assertIn('from humancoin import HumanCOINDataset', text)
        self.assertNotIn('humancoin_lerobot', text)
        self.assertNotIn('after PyPI release', text)

    def test_name_and_merged_modalities(self):
        text = gen_readme.render_readme(self.root,self.project)
        self.assertIn('# named_dataset',text)
        self.assertIn('HumanCOIN/named_dataset',text)
        self.assertIn('做三明治',text)
        self.assertIn('observation.depth | depth video',text)
        self.assertIn('observation.mic | audio',text)
        self.assertIn('16000',text)
        self.assertNotIn('## Units, coordinates and limitations',text)
        self.assertIn('license: cc-by-nc-4.0',text)
        self.assertIn('https://creativecommons.org/licenses/by-nc/4.0/',text)
        self.assertIn('Author One and Author Two',text)
        self.assertNotIn('Apache License',text)
        self.assertEqual(repo_id_for(self.root,'HumanCOIN'),'HumanCOIN/named_dataset')

    def test_recalculates_file_stats_excludes_root_readme(self):
        (self.root/'README.md').write_text('old content')
        (self.root/'backup').mkdir()
        (self.root/'backup/large').write_text('excluded')
        text = gen_readme.render_readme(self.root,self.project)
        size = sum(p.stat().st_size for p in (self.root/'meta').iterdir())
        self.assertIn(f'| Size (excluding README.md) | {size / 1024**2:.2f} MiB |',text)
        self.assertIn('| Files (excluding README.md) | 3 |',text)
        self.assertNotIn('99999999',text)

    def test_viewer_follows_tier_without_readme_video(self):
        video=self.root/'videos/chunk-000/observation.rgb/episode_000000.mp4'
        video.parent.mkdir(parents=True);video.write_bytes(b'video')
        for tier, enabled in [('l1', False), ('l2', True), ('l3', True)]:
            with self.subTest(tier=tier):
                self.description['dataset_name'] = tier + '_named_dataset'
                self.save()
                text = gen_readme.render_readme(self.root, self.project)
                self.assertIn('viewer: ' + str(enabled).lower(), text)
                self.assertNotIn('## Preview', text)
                self.assertNotIn('<video ', text)
                self.assertNotIn('[Open primary-view video]', text)

    def test_refuses_missing_or_invalid_dataset_name(self):
        for name in (None,'../bad','a/b',''):
            self.description['dataset_name']=name;self.save()
            with self.assertRaises(ValueError):
                gen_readme.render_readme(self.root,self.project)

    def test_existing_output_not_overwritten(self):
        cfg=Path(self.temp.name)/'project.json';cfg.write_text(json.dumps(self.project))
        out=self.root/'README.md';out.write_text('keep')
        args=['--dataset',str(self.root),'--project-config',str(cfg)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gen_readme.main(args),1)
        self.assertEqual(out.read_text(),'keep')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_readme.main(args+['--force']),0)
        self.assertIn('# named_dataset',out.read_text())

    def test_readable_metadata(self):
        self.description.update(scene={'type':'indoor','category':'household','name':'dining_area'},
                                actions=['grasp','retract_hand'], duration_s=37.533333,
                                objects=[{'name':'plate','attributes':{'color':None,'material':'wood'}}])
        self.save()
        text = gen_readme.render_readme(self.root,self.project)
        self.assertIn('Indoor / Household / Dining area',text)
        self.assertIn('**Atomic actions:** grasp, retract hand',text)
        self.assertIn('| plate | material: wood |',text)
        self.assertIn('| Duration | 37.53 s |',text)
        self.assertIn('| Data format | HumanCOIN-LeRobot |',text)
        self.assertNotIn('null',text)

    def test_metadata_table_escaping(self):
        self.assertEqual(gen_readme.cell('a|b\nc'),'a&#124;b<br>c')


if __name__ == '__main__':
    unittest.main()
