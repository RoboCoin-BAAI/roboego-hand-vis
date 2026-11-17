from scripts.guanglun.cleanup_release import main


def test_preview_then_delete_only_requested_sidecars(tmp_path):
    session = tmp_path / 'batch_0000001/session'
    remove = ['annotation/episode_000000_structured.json.before_enrichment.bak',
              'annotation/episode_000012_structured.json.before_enrichment.bak',
              'annotation/semantic_segments.json', 'annotation/stereo_quality/check.json',
              'meta/guanglun_source.json', 'meta/guanglun_video_validation.json']
    keep = ['annotation/episode_000000_structured.json',
            'annotation/episode_000000/valid_mask_ffv1.mkv', 'meta/info.json',
            'videos/chunk-000/depth/episode_000000.mp4',
            'data/chunk-000/episode_000000/depth.parquet']
    for name in remove + keep:
        path = session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('retained content')
    assert main(['--root', str(tmp_path)]) == 0
    assert all((session / name).exists() for name in remove + keep)
    assert main(['--root', str(tmp_path), '--execute']) == 0
    assert all(not (session / name).exists() for name in remove)
    assert all((session / name).read_text() == 'retained content' for name in keep)
    assert main(['--root', str(tmp_path), '--execute']) == 0


def test_flat_session_and_single_session_roots(tmp_path):
    session = tmp_path / 'hash'
    path = session / 'annotation/semantic_segments.json'
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    assert main(['--root', str(tmp_path)]) == 0
    assert path.is_file()
    assert main(['--root', str(session), '--execute']) == 0
    assert not path.exists()
