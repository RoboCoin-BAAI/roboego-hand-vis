import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from roboego_hand_vis.archive_depth_cache.database_run import run_pending
from roboego_hand_vis.archive_depth_cache.compute import DepthArchiveConfig


class Service:
    def __init__(self, paths):
        self.rows = [dict(id=i, local_path=p) for i, p in enumerate(paths)]
        self.results = []

    def claim_archive_depth_cache(self):
        return self.rows.pop(0) if self.rows else None

    def finish_archive_depth_cache(self, data_id, *, succeeded):
        self.results.append((data_id, succeeded))
        return True


def session(root):
    png = root/'session/depth/depth_uint16_png'
    png.mkdir(parents=True)
    for i in range(3):
        assert cv2.imwrite(str(png/f'frame_{i:06d}_depth_mm.png'), np.full((64,96), 1000+i, np.uint16))
    return png


def config():
    return DepthArchiveConfig(write_avi=False, write_preview=False, write_hevc12=True,
                              write_valid_mask=True, delete_png=True, hevc12_preset='ultrafast')


def publish(task, cfg):
    assert not cfg.delete_png and not cfg.delete_npy
    depth = task['depth_dir']
    for name in ('depth_hevc12.mkv', 'valid_mask_ffv1.mkv'):
        (depth/name).write_bytes(b'video')
    report = dict(source_depth_dir=str(depth), output_dir=str(depth), frame_count=3,
                  deleted_source_npy=False, deleted_uint16_png=False,
                  frames=[{'frame_idx': i} for i in range(3)],
                  outputs={'hevc12_video': str(depth/'depth_hevc12.mkv'),
                           'valid_mask_video': str(depth/'valid_mask_ffv1.mkv'),
                           'png_dir': str(depth/'depth_uint16_png')})
    (depth/'depth_archive_meta.json').write_text(json.dumps(report))


def test_success_checks_then_cleans_png(tmp_path):
    png=session(tmp_path)
    service=Service(['session'])
    def probe(path, _task):
        assert png.is_dir()  # cleanup must wait until all archives are checked
        return 3
    result=run_pending(service, tmp_path, config(), runner=publish, probe=probe)
    assert result[0]['status']=='completed'
    assert service.results==[(0,True)]
    assert not png.exists()
    meta=json.loads((tmp_path/'session/depth/depth_archive_meta.json').read_text())
    assert meta['deleted_uint16_png'] is True


def test_incomplete_video_preserves_sources_and_fails(tmp_path):
    png=session(tmp_path)
    service=Service(['session'])
    result=run_pending(service, tmp_path, config(), runner=publish, probe=lambda *args: 2)
    assert result[0]['status']=='failed'
    assert service.results==[(0,False)]
    assert len(list(png.glob('*.png')))==3


def test_bad_paths_and_interrupt_preserve_sources(tmp_path):
    png=session(tmp_path)
    service=Service(['../outside', 'missing', 'session'])
    def interrupt(*args):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,config(),runner=interrupt)
    assert service.results==[(0,False),(1,False),(2,False)]
    assert png.exists()


def test_dry_run_cannot_claim(tmp_path):
    service=Service(['session'])
    with pytest.raises(ValueError,match='dry_run'):
        run_pending(service,tmp_path,replace(config(),dry_run_avi=True))
    assert len(service.rows)==1


@pytest.mark.parametrize("all_outputs", [False, True])
def test_actual_command_encodes_and_verifies_three_frames(tmp_path, all_outputs):
    png=session(tmp_path)
    service=Service(['session'])
    cfg=replace(config(),write_avi=all_outputs,write_preview=all_outputs)
    result=run_pending(service,tmp_path,cfg)
    assert result[0]['status']=='completed',result
    assert service.results==[(0,True)]
    assert not png.exists()


def test_mixed_cache_cannot_delete_png_frames_missing_from_npy(tmp_path):
    png=session(tmp_path)
    np.save(png.parent/'frame_000000_foundation_depth_m.npy',np.ones((64,96),np.float32))
    service=Service(['session'])
    result=run_pending(service,tmp_path,config(),runner=lambda *args: pytest.fail('must not encode inconsistent cache'))
    assert result[0]['status']=='failed'
    assert len(list(png.glob('*.png')))==3


def test_png_target_symlink_cannot_write_outside_session(tmp_path):
    png=session(tmp_path)
    outside=tmp_path/'outside.png'
    outside.write_bytes(b'unchanged')
    (png/'frame_000000_depth_mm.png').unlink()
    (png/'frame_000000_depth_mm.png').symlink_to(outside)
    np.save(png.parent/'frame_000000_foundation_depth_m.npy',np.ones((64,96),np.float32))
    result=run_pending(Service(['session']),tmp_path,config(),runner=lambda *args: pytest.fail('must not write outside'))
    assert result[0]['status']=='failed'
    assert outside.read_bytes()==b'unchanged'
