import json
from pathlib import Path
from scripts.L1 import gpu_supplement as gpu


def test_resident_spans_batches_and_failures_do_not_mark_done(tmp_path):
    tasks=[]
    for batch in ('batch_0000001','batch_0000002'):
        folder=tmp_path/batch
        folder.mkdir()
        done=folder/'depth_done.json'
        gpu.save(done,set())
        tasks.extend((str(folder/name),done) for name in ('good','bad'))
    created=[]
    class FakeWorker:
        def __init__(self,stage,device,logs):
            self.calls=[];self.closed=False;created.append(self)
        def execute(self,cmd,log):
            self.calls.append(cmd)
            return int(cmd.endswith('bad'))
        def close(self):self.closed=True
    failed=gpu.run_phase('depth',tasks,['0'],tmp_path,lambda stage,s,d:s,FakeWorker)
    assert failed==2
    assert len(created)==1 and created[0].closed
    assert len(created[0].calls)==4
    for batch in ('batch_0000001','batch_0000002'):
        assert gpu.load(tmp_path/batch/'depth_done.json')=={str(tmp_path/batch/'good')}


def test_preview_merges_batches_and_respects_stage_done_lists(tmp_path,capsys):
    for batch in ('batch_0000001','batch_0000002'):
        folder=tmp_path/batch;folder.mkdir()
        ready={str(folder/'one'),str(folder/'two')}
        gpu.save(folder/'rectification_done.json',ready)
        gpu.save(folder/'depth_done.json',{str(folder/'one')})
        gpu.save(folder/'hand_done.json',set())
    assert gpu.main(['--root',str(tmp_path)])==0
    output=capsys.readouterr().out
    assert 'depth: 2 sessions' in output
    assert 'hand: 4 sessions' in output
