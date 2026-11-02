import json
import pytest
from roboego_hand_vis.lerobot_vio.L1 import compute


@pytest.mark.parametrize('first,last,expected,calls', [
    ('done', 'done', 'normal', 1),
    ('failed_quality', 'done', 'zupt', 2),
    ('failed_quality', 'failed_quality', None, 2),
    ('backend_error', 'done', 'zupt', 2),
])
def test_normal_first_policy(tmp_path, monkeypatch, first, last, expected, calls):
    episode = tmp_path / 'data/chunk-000/episode_000000'
    plan = dict(session=tmp_path, episode=episode, input_mode='rectified', output=tmp_path / 'vio/chunk-000/episode_000000')
    recorded = []
    def run(plan, **options):
        recorded.append((plan['output'], options['zupt']))
        status = first if len(recorded) == 1 else last
        if status == 'backend_error':
            raise RuntimeError('OpenVINS exited 3; see console.log')
        return dict(status=status, output=str(plan['output']))
    monkeypatch.setattr(compute, 'run_episode', run)
    result = compute.run_with_policy(plan, dict(zupt=False, zupt_fallback=True))
    assert len(recorded) == calls
    assert recorded[0][1] is False
    if calls == 2:
        assert recorded[1] == (tmp_path / 'vio_zupt/chunk-000/episode_000000', True)
    assert result['selected_mode'] == expected
    assert json.loads((plan['output'] / 'fallback_selection.json').read_text())['selected_mode'] == expected


def test_input_failure_does_not_retry(tmp_path, monkeypatch):
    def run(*args, **kwargs):
        raise ValueError('Invalid sensor timestamps')
    monkeypatch.setattr(compute, 'run_episode', run)
    with pytest.raises(ValueError):
        compute.run_with_policy({}, dict(zupt_fallback=True))


@pytest.mark.parametrize('policy,modes', [
    ('static-zupt', ['static-zupt']), ('dynamic', ['dynamic']),
    ('static-then-dynamic', ['static-zupt', 'dynamic']),
    ('dynamic-then-static', ['dynamic', 'static-zupt']),
])
@pytest.mark.parametrize('failure', ['failed_quality', 'backend', 'uninitialized', 'no_poses'])
def test_initialization_policies(tmp_path, monkeypatch, policy, modes, failure):
    plan = dict(session=tmp_path, episode=tmp_path / 'data/chunk-000/episode_000000',
                output=tmp_path / 'vio/chunk-000/episode_000000')
    calls = []
    def run(attempt, **options):
        mode = 'static-zupt' if options['zupt'] else 'dynamic'
        calls.append(mode)
        assert options['init_mode'] == ('static' if options['zupt'] else 'dynamic')
        assert attempt['output'].parts[-3] == ('vio_zupt' if options['zupt'] else 'vio')
        if len(calls) == 1:
            if failure == 'backend':
                raise RuntimeError('OpenVINS exited 1; see console.log')
            if failure == 'uninitialized':
                raise ValueError('OpenVINS report does not indicate successful initialization')
            if failure == 'no_poses':
                raise ValueError('No valid poses; diagnostics: work')
            return dict(status='failed_quality')
        return dict(status='done')
    monkeypatch.setattr(compute, 'run_episode', run)
    result = compute.run_with_policy(plan, dict(init_mode=policy))
    assert calls == modes
    assert result['selected_mode'] == (modes[-1] if len(modes) == 2 else None)


@pytest.mark.parametrize('status', ['done', 'prepared'])
def test_policy_stops_on_success_or_prepare(tmp_path, monkeypatch, status):
    plan = dict(session=tmp_path, episode=tmp_path / 'data/chunk-000/episode_000000')
    calls = []
    def run(attempt, **options):
        calls.append(options)
        return dict(status=status)
    monkeypatch.setattr(compute, 'run_episode', run)
    result = compute.run_with_policy(plan, dict(init_mode='static-then-dynamic', prepare_only=status == 'prepared'))
    assert len(calls) == 1
    assert result['attempted_modes'] == ['static-zupt']


def test_new_policy_input_error_does_not_retry(tmp_path, monkeypatch):
    plan = dict(session=tmp_path, episode=tmp_path / 'data/chunk-000/episode_000000')
    def run(*args, **kwargs):
        raise ValueError('Invalid sensor timestamps')
    monkeypatch.setattr(compute, 'run_episode', run)
    with pytest.raises(ValueError, match='Invalid sensor timestamps'):
        compute.run_with_policy(plan, dict(init_mode='static-then-dynamic'))
