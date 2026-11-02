import json
from types import SimpleNamespace

import pytest

from scripts.L3 import rectify_vio as batch


@pytest.mark.parametrize('rectify_code,vio_code,expected', [
    (1, 0, 'rectify_failed'), (0, 1, 'vio_failed'), (0, 0, 'done'),
])
def test_stage_gating_and_preserved_failure(tmp_path, monkeypatch, rectify_code, vio_code, expected):
    session = tmp_path / 'session'
    (session / 'meta').mkdir(parents=True)
    (session / 'meta/info.json').write_text('{}')
    calls = []
    def run(command, path, label):
        calls.append(command)
        return dict(returncode=rectify_code if len(calls) == 1 else vio_code,
                    records=[dict(status='failed_quality' if vio_code else 'done')], log=str(path))
    monkeypatch.setattr(batch, 'run_stage', run)
    args = SimpleNamespace(python='python', ffmpeg='ffmpeg', balance=0., openvins_root='open_vins',
                           force_rectify=False, zupt=True, zupt_fallback=False,
                           time_offset_mode='estimate', timing_bootstrap='auto', init_mode='static')
    result = batch.process(session, 1, args, tmp_path)
    assert result['status'] == expected
    assert len(calls) == (1 if rectify_code else 2)
    if not rectify_code:
        assert '--zupt' in calls[-1]
        assert '--input-mode' in calls[-1]
        assert calls[-1][calls[-1].index('--init-mode') + 1] == 'static'
    saved = json.loads((tmp_path / '0001_session/result.json').read_text())
    assert saved == result
