import json

import pytest

from roboego_hand_vis.video_privacy_blur.database_run import run_pending
from roboego_hand_vis.video_privacy_blur.batch import run_batch


class Service:
    def __init__(self, paths):
        self.rows=[dict(id=i,local_path=p) for i,p in enumerate(paths)]
        self.results=[]
    def claim_video_privacy_blur(self):
        return self.rows.pop(0) if self.rows else None
    def finish_video_privacy_blur(self, data_id, *, succeeded):
        self.results.append((data_id,succeeded))
        return True


def setup(root):
    weights=root/'face.pt'
    weights.write_bytes(b'model')
    videos=root/'session/videos'
    videos.mkdir(parents=True)
    for name in ('left.mkv','right.mkv'):
        (videos/name).write_bytes(b'input video')
    return {'paths':{'face_weights':str(weights)},'parameters':{'device':'cpu','jobs':1}}


def publish(config):
    def process(source, output, data, log):
        output.write_bytes(b'encoded video')
        return {'status':'complete','frames':3,'width':64,'height':64,'fps':30}
    return run_batch(config.input_dir,config.output_dir,{},processor=process,jobs=1,
                     selected_videos=config.selected_videos or None)


def test_success_and_cached_outputs_complete(tmp_path):
    data=setup(tmp_path)
    for _ in range(2):
        service=Service(['session'])
        result=run_pending(service,tmp_path,data,runner=publish)
        assert result[0]['status']=='completed',result
        assert result[0]['video_count']==2
        assert service.results==[(0,True)]


def test_failed_video_cannot_complete_whole_session(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    def process(config):
        publish(config)
        (config.output_dir/'right_blurred.mp4').unlink()
    assert run_pending(service,tmp_path,data,runner=process)[0]['status']=='failed'
    assert service.results==[(0,False)]


def test_selected_video_and_bad_paths(tmp_path):
    data=setup(tmp_path)
    data['parameters']['videos']=['left.mkv']
    service=Service(['../outside','missing','session'])
    results=run_pending(service,tmp_path,data,runner=publish)
    assert [r['status'] for r in results]==['failed','failed','completed']
    assert results[-1]['video_count']==1
    assert not (tmp_path/'session/privacy/video_privacy_blur/right_blurred.mp4').exists()


def test_preflight_dry_run_and_missing_weights_do_not_claim(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    data['parameters']['dry_run']=True
    with pytest.raises(ValueError,match='dry_run'):
        run_pending(service,tmp_path,data,runner=publish)
    data['parameters']['dry_run']=False
    (tmp_path/'face.pt').unlink()
    with pytest.raises(ValueError,match='weights'):
        run_pending(service,tmp_path,data,runner=publish)
    assert len(service.rows)==1


def test_interruption_marks_failed(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    def interrupted(config):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,data,runner=interrupted)
    assert service.results==[(0,False)]


def test_privacy_interrupt_stops_detached_worker_descendants(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from roboego_hand_vis.common import processes
    from roboego_hand_vis.video_privacy_blur import batch
    original = subprocess.Popen
    marker = tmp_path / "grandchild.pid"
    children = []
    grandchild_code = ("import os, signal, time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                      f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)")
    parent_code = f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]); time.sleep(60)"

    def child(command, **kwargs):
        process = original([sys.executable, "-c", parent_code], **kwargs)
        children.append(process)
        original_wait = process.wait
        first = True

        def interrupted_wait(*args, **kw):
            nonlocal first
            if first:
                first = False
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert marker.exists()
                raise KeyboardInterrupt
            return original_wait(*args, **kw)

        process.wait = interrupted_wait
        return process

    monkeypatch.setattr(batch.subprocess, "Popen", child)
    with pytest.raises(KeyboardInterrupt):
        batch.process_file(tmp_path / "source.mp4", tmp_path / "out.mp4", {}, tmp_path / "stage.log")
    assert children[0].poll() is not None
    try:
        assert not processes.group_has_live_processes(children[0].pid)
    finally:
        processes.stop_process_group(children[0])



def test_batch_interrupt_exit_stops_session_queue(tmp_path,monkeypatch):
    from roboego_hand_vis.video_privacy_blur.database_run import _run_session
    from roboego_hand_vis.video_privacy_blur import batch
    data=setup(tmp_path)
    service=Service(['session','session'])
    monkeypatch.setattr(batch,'main',lambda argv: 130)
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,data,runner=_run_session)
    assert service.results==[(0,False)]
    assert len(service.rows)==1
