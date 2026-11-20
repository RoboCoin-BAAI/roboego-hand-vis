from pathlib import Path
import json
import pytest

from roboego_hand_vis.video_privacy_blur import jianzhi_inplace as module


class FakeWorker:
    def __init__(self):
        self.calls = 0

    def run(self, path):
        self.calls += 1
        request = json.loads(Path(path).read_text())
        output = Path(request['output'])
        output.write_bytes(b'blurred')
        output.with_suffix('.report.json').write_text(json.dumps({'status': 'complete', 'frames': 3}))


def setup_state(tmp_path):
    source = tmp_path / 'episode.mp4'
    source.write_bytes(b'original')
    state = tmp_path / 'state'
    for name in ('jobs', 'logs', 'work'):
        (state / name).mkdir(parents=True)
    return source, state


def test_inplace_resume_and_changed_input(tmp_path):
    source, state = setup_state(tmp_path)
    worker = FakeWorker()
    config = {'threads': 2}
    assert module.process_video(source, state, config, '2', worker) == 'completed'
    assert source.read_bytes() == b'blurred'
    assert module.process_video(source, state, config, '2', worker) == 'skipped'
    assert worker.calls == 1
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Source changed'):
        module.process_video(source, state, config, '2', worker)


def test_interrupted_after_replace_is_recovered(tmp_path, monkeypatch):
    source, state = setup_state(tmp_path)
    worker = FakeWorker()
    original_write = module.write_json
    def interrupted(path, record):
        if record.get('status') == 'complete':
            raise OSError('simulated crash')
        original_write(path, record)
    monkeypatch.setattr(module, 'write_json', interrupted)
    with pytest.raises(OSError):
        module.process_video(source, state, {'threads': 2}, '0', worker)
    monkeypatch.setattr(module, 'write_json', original_write)
    assert module.process_video(source, state, {'threads': 2}, '0', worker) == 'skipped'
    assert worker.calls == 1


def test_persistent_worker_reuses_model(tmp_path, monkeypatch):
    import io
    import sys
    from types import ModuleType
    from roboego_hand_vis.video_privacy_blur import worker
    loads, renders = [], []
    detectors = ModuleType('roboego_hand_vis.video_privacy_blur.detectors')
    def make_detector(*args):
        loads.append(args)
        return object()
    detectors.FaceDetectorYOLO = make_detector
    utils = ModuleType('roboego_hand_vis.video_privacy_blur.utils')
    utils.choose_device = lambda device: device
    monkeypatch.setitem(sys.modules, detectors.__name__, detectors)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    from roboego_hand_vis.video_privacy_blur import batch
    monkeypatch.setattr(batch, 'sha256', lambda _: 'hash')
    monkeypatch.setattr(worker, 'render', lambda source, output, config, detector: renders.append(detector) or {'status': 'complete', 'frames': 1})
    config = dict(face_weights='weights', weights_sha256='hash', device='cuda', conf=.35, imgsz=960)
    requests = []
    for index in range(2):
        path = tmp_path / f'{index}.json'
        path.write_text(json.dumps({'source': 'input', 'output': str(tmp_path / f'{index}.mp4'), 'config': config}))
        requests.append(json.dumps({'request': str(path)}))
    monkeypatch.setattr(sys, 'stdin', io.StringIO('\n'.join(requests) + '\n'))
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', output)
    worker.persistent_main()
    assert len(loads) == 1
    assert len(renders) == 2 and renders[0] is renders[1]
    assert all(json.loads(line)['ok'] for line in output.getvalue().splitlines())
