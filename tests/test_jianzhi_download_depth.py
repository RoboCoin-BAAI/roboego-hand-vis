import pytest
from scripts.jianzhi.download_depth import select_objects


def test_client_options_are_specific_to_copy(tmp_path, monkeypatch):
    from scripts.jianzhi import download_depth as module
    from types import SimpleNamespace
    commands = []
    def run(command, **options):
        commands.append(command)
        assert options['cwd'] == tmp_path
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    client = module.Client('tool', 'config', tmp_path)
    client._run(['ls', '-d', 'ks3://bucket/root/'])
    client._run(['cp', 'ks3://bucket/root/file', '/tmp/file'])
    assert not any(arg.startswith(('--output-dir=', '--checkpoint-dir=')) for arg in commands[0])
    assert any(arg.startswith('--output-dir=') for arg in commands[1])
    assert any(arg.startswith('--checkpoint-dir=') for arg in commands[1])


def test_directory_copy_keeps_paths_and_uses_tool_concurrency(tmp_path):
    from scripts.jianzhi.download_depth import copy_selected
    selected = [dict(relative=p, size=1) for p in [
        'meta/info.json', 'data/chunk-000/episode_000000/a.parquet',
        'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4',
        'videos/chunk-000/observation.images.head_right_rectified/episode_000000.mp4']]
    commands = []
    class Client:
        def _run(self, cmd):
            commands.append(cmd)
            dest = __import__('pathlib').Path(cmd[2])
            for item in selected:
                path = tmp_path/item['relative']
                if path.is_relative_to(dest):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'x')
    copy_selected(Client(), 'ks3://bucket/session/', selected, tmp_path, 64)
    assert len(commands) == 4
    assert all('-r' in c and '-u' in c and '--jobs=64' in c for c in commands)
    assert all(c[1].endswith('/') for c in commands)


def test_download_keeps_only_rectified_video_and_all_other_folders():
    source = 'ks3://bucket/session/'
    paths = ['meta/info.json', 'calibrations/dataset_manifest.json',
             'annotation/result.json', 'data/chunk-000/episode_000000/images.parquet',
             'videos/chunk-000/observation.images.head_left/episode_000000.mp4',
             'videos/chunk-000/observation.images.head_right_outer0/episode_000000.mp4']
    wanted = [f'videos/chunk-000/observation.images.head_{side}_rectified/episode_000000.mp4'
              for side in ('left', 'right')]
    objects = [dict(uri=source+p, size=1, etag='') for p in paths+wanted]
    result = select_objects(objects, source)
    assert {i['relative'] for i in result} == set(paths[:4]+wanted)
    with pytest.raises(ValueError, match='矫正左右目'):
        select_objects(objects[:-1], source)
    with pytest.raises(ValueError, match='不安全'):
        select_objects(objects+[dict(uri=source+'../escape', size=1)], source)
