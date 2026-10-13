import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.hand.database_run import run_pending


class Service:
    def __init__(self, paths):
        self.rows=[dict(id=i,local_path=p) for i,p in enumerate(paths)]
        self.results=[]
    def claim_hand(self):
        return self.rows.pop(0) if self.rows else None
    def finish_hand(self, data_id, *, succeeded):
        self.results.append((data_id,succeeded))
        return True


def video(path, count=3):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30,(32,32))
    assert writer.isOpened()
    for _ in range(count):
        writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()


def setup(root):
    mint,ego=root/'mint',root/'ego'
    for p in (mint/'mint',mint/'scripts',ego/'scripts',root/'session/depth'):
        p.mkdir(parents=True)
    (mint/'model.safetensors').write_bytes(b'weights')
    (ego/'run.py').touch()
    (ego/'scripts/run_without_front.sh').write_text('# EGOHANDKIT_PYTHON')
    (ego/'scripts/export_egohandkit_cache_mint.py').write_text('exporter')
    video(root/'session/videos/left_rectified.mp4')
    return {'paths':{'mint_root':str(mint),'mint_python':sys.executable,'mint_checkpoint':'model.safetensors',
                     'egohandkit_root':str(ego),'egohandkit_python':sys.executable}}


def publish(plan, *, resume=False):
    output=Path(plan['output_dir'])
    output.mkdir(exist_ok=True)
    pq.write_table(pa.table({'frame_idx':list(range(3))}),output/'hand_tracking.parquet')
    video(output/'render.mp4')
    (output/'run.log').write_text('completed test launcher\n')
    from roboego_hand_vis.hand import run as native
    from roboego_hand_vis.hand.state import atomic_json, SCHEMA_VERSION
    work=output/'.work'
    work.mkdir(exist_ok=True)
    atomic_json(work/'state.json',{'schema_version':SCHEMA_VERSION,'plan':plan,
                'fingerprint':native.fingerprint(plan),'stages':{},
                'delivery':native.artifact_hashes(output,output,'egohandkit')})
    if plan['output_mode']=='compact':
        from roboego_hand_vis.hand import outputs
        outputs.cleanup_work(output)
    return 0


def test_delivery_verified_and_receipt_avoids_duplicate_inference(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    result=run_pending(service,tmp_path,data,runner=publish)
    assert result[0]['status']=='completed',result
    assert result[0]['frame_count']==3
    assert service.results==[(0,True)]
    result=run_pending(Service(['session']),tmp_path,data,runner=lambda *a,**kw:pytest.fail('do not rerun completed inference'))
    assert result[0]['status']=='completed'


def test_wrong_parquet_frame_count_fails(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    def partial(plan,**kwargs):
        publish(plan,**kwargs)
        pq.write_table(pa.table({'frame_idx':[0]}),Path(plan['output_dir'])/'hand_tracking.parquet')
        return 0
    result=run_pending(service,tmp_path,data,runner=partial)
    assert result[0]['status']=='failed'
    assert service.results==[(0,False)]


def test_resume_existing_checkpoint_and_interruption(tmp_path):
    data=setup(tmp_path)
    work=tmp_path/'session/hand/.work'
    work.mkdir(parents=True)
    from roboego_hand_vis.hand import run as native
    from roboego_hand_vis.hand.state import atomic_json, SCHEMA_VERSION
    plan=native.build_plan(data,tmp_path/'session',tmp_path/'session/hand',full_output=True)
    atomic_json(work/'state.json',{'schema_version':SCHEMA_VERSION,'plan':plan,
                'fingerprint':native.fingerprint(plan),'stages':{},'delivery':None})
    service=Service(['session','session'])
    def interrupt(plan, *, resume):
        assert resume is True
        return 130
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,data,runner=interrupt)
    assert service.results==[(0,False)]
    assert len(service.rows)==1


def test_unmanaged_output_and_unsafe_paths_are_rejected(tmp_path):
    data=setup(tmp_path)
    (tmp_path/'session/hand').mkdir()
    service=Service(['../outside','missing','session'])
    result=run_pending(service,tmp_path,data,runner=lambda *a,**kw:pytest.fail('do not overwrite existing result'))
    assert all(r['status']=='failed' for r in result)


def test_global_missing_model_does_not_claim(tmp_path):
    data=setup(tmp_path)
    (tmp_path/'mint/model.safetensors').unlink()
    service=Service(['session'])
    with pytest.raises(ValueError,match='checkpoint'):
        run_pending(service,tmp_path,data,runner=publish)
    assert len(service.rows)==1


def test_changed_delivery_is_not_silently_accepted(tmp_path):
    data=setup(tmp_path)
    run_pending(Service(['session']),tmp_path,data,runner=publish)
    (tmp_path/'session/hand/render.mp4').write_bytes(b'changed')
    result=run_pending(Service(['session']),tmp_path,data,runner=lambda *a,**kw:pytest.fail('do not overwrite changed result'))
    assert result[0]['status']=='failed'


def test_hand_interrupt_stops_encoder_descendants(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from roboego_hand_vis.common import processes
    from roboego_hand_vis.hand import run as native
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

    monkeypatch.setattr(native.subprocess, "Popen", child)
    with pytest.raises(KeyboardInterrupt):
        native.run_stage({"command": [sys.executable, "-c", "pass"], "cwd": str(tmp_path)}, {}, tmp_path / "stage.log")
    assert children[0].poll() is not None
    try:
        assert not processes.group_has_live_processes(children[0].pid)
    finally:
        processes.stop_process_group(children[0])



def test_input_changed_during_inference_cannot_complete(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    def changed(plan,**kwargs):
        publish(plan,**kwargs)
        (Path(plan['depth'])/'changed-depth.npy').write_bytes(b'changed')
        return 0
    result=run_pending(service,tmp_path,data,runner=changed)
    assert result[0]['status']=='failed'
    assert not (tmp_path/'session/annotation/hand_database_receipt.json').exists()


def test_publication_checkpoint_with_changed_inputs_is_rejected(tmp_path):
    from roboego_hand_vis.hand import run as native
    data=setup(tmp_path)
    plan=native.build_plan(data,tmp_path/'session',tmp_path/'session/hand',full_output=True)
    publish(plan)
    video(tmp_path/'session/videos/left_rectified.mp4')  # same frame count, changed input identity
    result=run_pending(Service(['session']),tmp_path,data,runner=lambda *a,**kw:pytest.fail('must reject old delivery'))
    assert result[0]['status']=='failed'
    assert not (tmp_path/'session/annotation/hand_database_receipt.json').exists()


def test_receipt_write_failure_keeps_native_checkpoint_for_retry(tmp_path,monkeypatch):
    from roboego_hand_vis.hand import database_run
    data=setup(tmp_path)
    original=database_run.atomic_json
    def fail_receipt(*args):
        raise OSError('simulated receipt write failure')
    monkeypatch.setattr(database_run,'atomic_json',fail_receipt)
    result=run_pending(Service(['session']),tmp_path,data,runner=publish)
    assert result[0]['status']=='failed'
    assert (tmp_path/'session/hand/.work/state.json').exists()
    monkeypatch.setattr(database_run,'atomic_json',original)
    result=run_pending(Service(['session']),tmp_path,data)
    assert result[0]['status']=='completed',result
    assert not (tmp_path/'session/hand/.work').exists()


def test_cleanup_interruption_recovers_without_inference(tmp_path,monkeypatch):
    from roboego_hand_vis.hand import outputs
    data=setup(tmp_path)
    service=Service(['session'])
    original=outputs.cleanup_work
    def interrupt(output):
        raise KeyboardInterrupt
    monkeypatch.setattr(outputs,'cleanup_work',interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,data,runner=publish)
    assert service.results==[(0,False)]
    assert (tmp_path/'session/annotation/hand_database_receipt.json').exists()
    monkeypatch.setattr(outputs,'cleanup_work',original)
    result=run_pending(Service(['session']),tmp_path,data,runner=lambda *a,**kw:pytest.fail('do not repeat inference'))
    assert result[0]['status']=='completed'
    assert not (tmp_path/'session/hand/.work').exists()
