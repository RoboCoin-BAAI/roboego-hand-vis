import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.guanglun.reencode_depth import FEATURE, convert, main
from roboego_hand_vis.lerobot_fast_foundation_depth.L1.depth_video import HEVC12Writer, read_episode_depth


def legacy(root, ffmpeg):
    source = root / 'batch_0000242/session/depth/chunk-000' / FEATURE
    source.mkdir(parents=True)
    video = source / 'episode_000000.mkv'
    depth = np.linspace(.2, 3.9, 4096, dtype=np.float32).reshape(64, 64)
    depth[::3, ::2] = np.nan
    writer = HEVC12Writer(video, size=(64, 64), fps=30, ffmpeg=ffmpeg, lossless=True, threads=1)
    try:
        writer.write(depth)
        writer.write(depth[::-1].copy())
    finally:
        writer.close()
    report = {'processed_frames': [4, 7], 'timestamp_file': FEATURE + '.parquet', 'depth_storage': {
        'file': video.name, 'frames': 2, 'width': 64, 'height': 64, 'fps': 30,
        'max_depth_m': 4., 'lossless': True}}
    (source / 'lerobot_depth.json').write_text(json.dumps(report))
    times = pa.table({'timestamp': [1.25, 1.35], 'episode_index': [0, 0]})
    pq.write_table(times, source / (FEATURE + '.parquet'))
    return source, times, depth


@pytest.mark.parametrize('verify_video', [True, False])
def test_encode_migrate_resume_and_cleanup(tmp_path, monkeypatch, verify_video):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source, times, original = legacy(tmp_path, ffmpeg)
    session = source.parents[2]
    assert convert(source) == 'planned'
    assert not (session / 'annotation').exists()
    if not verify_video:
        from scripts.guanglun import reencode_depth
        monkeypatch.setattr(reencode_depth, 'read_episode_depth', lambda *a, **kw: pytest.fail('Unexpected verification decode'))
    args = ['--root', str(tmp_path), '--execute', '--ffmpeg', ffmpeg]
    if not verify_video:
        args.append('--skip-video-verification')
    assert main(args) == 0
    output = session / 'annotation/chunk-000' / FEATURE / 'episode_000000'
    report = json.loads((output / 'lerobot_depth.json').read_text())
    assert report['processed_frames'] == [4, 7]
    assert report['timestamp_index'] == 'video_frame'
    assert 'depth' not in (session / report['timestamp_file']).relative_to(session).parts
    assert pq.read_table(session / report['timestamp_file']).equals(times)
    frames = list(read_episode_depth(output, ffmpeg=ffmpeg))
    np.testing.assert_array_equal(np.isnan(frames[0]), np.isnan(original))
    assert source.is_dir()
    assert convert(source, execute=True, delete_source=True, ffmpeg=ffmpeg, verify_video=verify_video) == 'reused'
    assert not (session / 'depth').exists()
    assert len(list(read_episode_depth(output, ffmpeg=ffmpeg))) == 2


def test_bad_source_preserved(tmp_path):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source, _, _ = legacy(tmp_path, ffmpeg)
    path = source / 'lerobot_depth.json'
    report = json.loads(path.read_text())
    report['depth_storage']['frames'] = 3
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='must match'):
        convert(source, execute=True, delete_source=True, ffmpeg=ffmpeg)
    assert (source / 'episode_000000.mkv').is_file()
    assert not (source.parents[2] / 'videos').exists()


def test_l3_selects_source_rows_and_preserves_sidecar(tmp_path):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source, _, _ = legacy(tmp_path, ffmpeg)
    feature = 'observation.images.depth_headband_left_rectified'
    target = source.with_name(feature)
    source.rename(target)
    source = target
    (source / (FEATURE + '.parquet')).unlink()
    table = pa.table({'timestamp': np.arange(10) / 30., 'episode_index': [0] * 10})
    pq.write_table(table, source / (feature + '.parquet'))
    report_path = source / 'lerobot_depth.json'
    report = json.loads(report_path.read_text())
    report['timestamp_file'] = feature + '.parquet'
    report_path.write_text(json.dumps(report))
    (source / 'episode_000000_meta.json').write_text('{"depth_encoding":"hevc12_lossless"}')
    session = source.parents[2]
    assert convert(source, execute=True, delete_source=True, ffmpeg=ffmpeg, verify_video=False) == 'converted'
    output = session / 'annotation/chunk-000' / feature / 'episode_000000'
    saved = json.loads((output / 'lerobot_depth.json').read_text())
    assert saved['depth_feature'] == feature
    assert pq.read_table(session / saved['timestamp_file']).equals(table.take(pa.array([4, 7])))
    assert json.loads((output / 'episode_000000_meta.json').read_text())['depth_encoding'] == 'hevc12_crf0'
    assert not (session / 'depth').exists()


def test_cancel_stops_active_children_and_rejects_new_ones(monkeypatch):
    import sys
    from threading import Event
    from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import depth_video
    monkeypatch.setattr(depth_video, '_cancelled', Event())
    process = depth_video._start_process([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        depth_video.cancel_active_processes()
        assert process.wait(timeout=5) != 0
        with pytest.raises(RuntimeError, match='cancelled'):
            depth_video._start_process([sys.executable, '-c', 'pass'])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        depth_video._forget_process(process)


def test_discovery_uses_fixed_levels_and_batch_filter(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts.guanglun.reencode_depth import discover_sources
    monkeypatch.setattr(Path, 'rglob', lambda *a, **k: pytest.fail('Recursive scan forbidden'))
    for batch in ('batch_0000001', 'batch_0000002'):
        source = tmp_path / batch / 'session/depth/chunk-000' / FEATURE
        source.mkdir(parents=True)
        (source / 'lerobot_depth.json').write_text('{}')
    expected = tmp_path / 'batch_0000002/session/depth/chunk-000' / FEATURE
    assert list(discover_sources(tmp_path, 2, 2)) == [expected]
    assert list(discover_sources(expected.parents[1])) == [expected]
    assert list(discover_sources(expected.parents[2])) == [expected]


def test_first_job_starts_before_all_sessions_are_discovered(tmp_path, monkeypatch):
    from scripts.guanglun import reencode_depth
    from threading import Event
    started = Event()
    def discover(*args):
        yield tmp_path / 'first'
        assert started.wait(3), 'Work did not start before discovery resumed'
        yield tmp_path / 'second'
    def convert(*args, **kwargs):
        started.set()
        return 'converted'
    monkeypatch.setattr(reencode_depth, 'discover_sources', discover)
    monkeypatch.setattr(reencode_depth, 'convert', convert)
    assert main(['--root', str(tmp_path), '--workers', '2']) == 0


def test_batches_are_discovered_in_numeric_order(tmp_path):
    from scripts.guanglun.reencode_depth import discover_sources
    for number in (241, 10, 2, 1):
        source = tmp_path / f'batch_{number:07d}' / 'session/depth/chunk-000' / FEATURE
        source.mkdir(parents=True)
        (source / 'lerobot_depth.json').write_text('{}')
    assert [int(p.parents[3].name[6:]) for p in discover_sources(tmp_path)] == [1, 2, 10, 241]
