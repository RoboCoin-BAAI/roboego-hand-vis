from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.pipeline import run as pipeline


@pytest.fixture
def portable_execution(monkeypatch):
    # Real Linux fcntl locks are exercised on deployment; use the same FD-shaped
    # interface to test scheduler semantics on Windows without pretending to lock NFS.
    @contextmanager
    def local_claim(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a+') as stream:
            yield stream
    popen, run = subprocess.Popen, subprocess.run
    def launch(*args, **kwargs):
        kwargs.pop('pass_fds', None)
        return popen(*args, **kwargs)
    def execute(*args, **kwargs):
        kwargs.pop('pass_fds', None)
        return run(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'claim', local_claim)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', launch)
    monkeypatch.setattr(pipeline.subprocess, 'run', execute)


def config(tmp_path):
    session = tmp_path / 'data/batch_1/session'
    session.mkdir(parents=True)
    stages = {}
    for name, role, deps in (('qc','cpu',[]), ('rectify','cpu',['qc']), ('depth','gpu',['qc','rectify'])):
        target = session / (name+'.txt')
        program = f'from pathlib import Path;Path({str(target)!r}).write_text("done")'
        stages[name] = dict(role=role, depends=deps, command=[sys.executable,'-c',program], outputs=[name+'.txt'])
    return dict(dataset='demo',level='L3',start_at='qc',paths=dict(input_root=str(tmp_path/'data'),
                output_root=str(tmp_path/'data'),state_root=str(tmp_path/'state')),
                run_stages=list(stages), force_stages=[], stages=stages)


def task():
    return dict(batch='batch_1',name='session',source='ks3://bucket/session')


def test_role_gates_force_and_artifact_changes(tmp_path, portable_execution):
    c=config(tmp_path)
    cpu=pipeline.Runner(c,'cpu');gpu=pipeline.Runner(c,'gpu')
    assert not gpu.step(task())
    assert cpu.step(task())  # QC
    assert not gpu.step(task())
    assert cpu.step(task())  # Rectification
    assert gpu.step(task())
    assert not gpu.step(task())
    c['force_stages']=['qc']
    again=pipeline.Runner(c,'cpu')
    assert again.step(task())
    states=pipeline.read(again.folder(task())/'state.json')
    assert states['rectify']['status']=='stale'
    assert states['depth']['status']=='stale'
    assert not gpu.step(task())
    assert again.step(task())  # Not QC again within this request.
    assert gpu.step(task())
    (tmp_path/'data/batch_1/session/qc.txt').unlink()
    states=pipeline.read(cpu.folder(task())/'state.json')
    assert not cpu.current('depth',task(),states)


def test_failed_force_restores_previous_and_does_not_loop(tmp_path, portable_execution):
    c=config(tmp_path);c['run_stages']=['qc']
    r=pipeline.Runner(c,'cpu');assert r.step(task())
    c['force_stages']=['qc']
    c['stages']['qc'].update(command=[sys.executable,'-c','raise SystemExit(1)'],backup_paths=['{session}/qc.txt'])
    retry=pipeline.Runner(c,'cpu',retry=True)
    assert retry.step(task())
    assert (tmp_path/'data/batch_1/session/qc.txt').read_text()=='done'
    assert not retry.step(task())
    states=pipeline.read(retry.folder(task())/'state.json')
    assert states['qc']['status']=='failed'
    assert not retry.current('qc',task(),states)


def test_config_cycles_and_start_at(tmp_path):
    c=config(tmp_path)
    pipeline.validate_config(c)
    c['stages']['qc']['depends']=['depth']
    c['start_at']='download'
    with pytest.raises(ValueError,match='Cyclic'):
        pipeline.validate_config(c)


def test_templates_for_future_dataset(tmp_path):
    c=config(tmp_path);c['dataset']='jianzhi'
    runner=pipeline.Runner(c,'gpu')
    assert '/jianzhi/' in runner.folder(task()).as_posix()
    assert runner.command(['tool','{session}','--gpu','{gpu}'],runner.values(task(),'3'))[-1]=='3'


def test_adoption_does_not_reapprove_stale_records(tmp_path, portable_execution):
    c=config(tmp_path);c['run_stages']=['qc']
    c['stages']['qc']['adopt_command']=[sys.executable,'-c','pass']
    first=pipeline.Runner(c,'cpu')
    assert first.step(task())
    # A changed implementation makes the record stale. --adopt-existing must
    # run the new command, not reapprove its old output with the importer.
    c['stages']['qc']['command']=[sys.executable,'-c','raise SystemExit(1)']
    again=pipeline.Runner(c,'cpu',adopt=True)
    assert again.step(task())
    state=pipeline.read(again.folder(task())/'state.json')
    assert state['qc']['status']=='failed'
    assert not state['qc']['adopted']



def test_rectification_index_and_gpu_discovery(tmp_path, portable_execution):
    c = config(tmp_path)
    cpu = pipeline.Runner(c, 'cpu')
    assert cpu.step(task())  # QC alone must not publish readiness.
    assert pipeline.gpu_tasks(c, []) == []
    assert cpu.step(task())
    session = str((tmp_path / 'data/batch_1/session').resolve())
    index = pipeline.rectified_index(c, 'batch_1')
    assert json.loads(index.read_text()) == [session]
    assert pipeline.gpu_tasks(c, []) == [task()]
    assert pipeline.gpu_tasks(c, ['batch_2']) == []
    pipeline.update_rectified_index(c, session, True)
    assert json.loads(index.read_text()) == [session]
    pipeline.update_rectified_index(c, session, False)
    assert pipeline.gpu_tasks(c, []) == []


def test_failed_rectification_not_listed(tmp_path, portable_execution):
    c = config(tmp_path)
    c['stages']['rectify']['command'] = [sys.executable, '-c', 'raise SystemExit(1)']
    cpu = pipeline.Runner(c, 'cpu')
    assert cpu.step(task())
    assert cpu.step(task())
    assert pipeline.gpu_tasks(c, []) == []
    assert json.loads(pipeline.rectified_index(c, 'batch_1').read_text()) == []



def test_gpu_depth_phase_precedes_hand_and_has_double_slots(tmp_path):
    from types import SimpleNamespace
    from threading import Lock
    import time
    c = config(tmp_path)
    c['stages']['hand'] = dict(role='gpu', enabled=True)
    c['run_stages'] = ['hand', 'depth']
    selected = [dict(name=str(i)) for i in range(8)]
    lock = Lock()
    finished, busy = set(), set()
    active = {'depth': 0, 'hand': 0}
    peak = {'depth': 0, 'hand': 0}
    class FakeRunner:
        config = c
        def step(self, task, device, stages=None):
            stage = stages[0]
            key = (task['name'], stage)
            with lock:
                if key in finished or key in busy:
                    return False
                if stage == 'hand':
                    assert len([k for k in finished if k[1] == 'depth']) == 8
                    assert active['depth'] == 0
                busy.add(key)
                active[stage] += 1
                peak[stage] = max(peak[stage], active[stage])
            time.sleep(.02)
            with lock:
                busy.remove(key)
                finished.add(key)
                active[stage] -= 1
            return True
        def folder(self, task):
            return tmp_path / task['name']
        def current(self, stage, task, states):
            return (task['name'], stage) in finished
    args = SimpleNamespace(depth_workers_per_gpu=2,watch=False)
    assert pipeline.run_gpu_phases(FakeRunner(), selected, ['0','1'], args) == 0
    assert peak == {'depth': 4, 'hand': 2}



def test_qc_start_discovers_converted_sessions_without_csv(tmp_path):
    c = config(tmp_path)
    session = tmp_path / 'data/batch_1/session'
    (session/'meta').mkdir()
    (session/'meta/info.json').write_text('{}')
    assert pipeline.tasks(c, ['batch_1']) == [dict(batch='batch_1',name='session',source=str(session.resolve()))]
    assert pipeline.tasks(c, []) == pipeline.tasks(c, ['batch_1'])
    with pytest.raises(ValueError,match='No converted sessions'):
        pipeline.tasks(c, ['batch_2'])


def test_download_start_still_requires_csv(tmp_path):
    c = config(tmp_path)
    c['start_at'] = 'download'
    with pytest.raises(ValueError, match='No batch CSVs'):
        pipeline.tasks(c, [])


@pytest.mark.parametrize('level', ['L1', 'L3'])
def test_cpu_full_range_barrier_and_resume(tmp_path, portable_execution, level):
    from types import SimpleNamespace
    c = config(tmp_path)
    c['level'] = level
    names = ['download'] + (['convert'] if level == 'L1' else []) + ['qc', 'rectify', 'review', 'vio']
    events = tmp_path / 'events.txt'
    c['stages'] = {}
    for i, name in enumerate(names):
        program = ('from pathlib import Path;import sys;'
                   's=Path(sys.argv[1]);s.mkdir(parents=True,exist_ok=True);'
                   ' (s/sys.argv[2]).write_text("done");'
                   'f=open(sys.argv[3],"a");f.write(sys.argv[2]+":"+s.name+"\\n");f.close()')
        c['stages'][name] = dict(role='cpu', depends=names[i-1:i],
            command=[sys.executable, '-c', program, '{session}', name, str(events)], outputs=[name])
    c['run_stages'] = names
    selected = [dict(batch='batch_1',name=n,source=n) for n in ['a','b','c']]
    runner = pipeline.Runner(c, 'cpu')
    args = SimpleNamespace(watch=False, poll_s=1)
    assert pipeline.run_cpu_phases(runner, selected, ['0'], args) == 0
    assert events.read_text().splitlines() == [name+':'+t['name'] for name in names for t in selected]
    before = events.read_text()
    assert pipeline.run_cpu_phases(pipeline.Runner(c, 'cpu'), selected, ['0'], args) == 0
    assert events.read_text() == before


def test_cpu_failed_phase_finishes_peers_and_blocks_next(tmp_path, portable_execution):
    from types import SimpleNamespace
    c = config(tmp_path)
    c['run_stages'] = ['qc','rectify']
    c['stages']['qc'].update(command=[sys.executable,'-c',
        'from pathlib import Path;import sys;s=Path(sys.argv[1]);s.mkdir(parents=True,exist_ok=True);'
        'sys.exit(1) if s.name=="bad" else (s/"qc.txt").write_text("done")', '{session}'])
    selected = [dict(batch='batch_1',name=n,source=n) for n in ['bad','good']]
    runner = pipeline.Runner(c,'cpu')
    assert pipeline.run_cpu_phases(runner, selected, ['0'], SimpleNamespace(watch=True,poll_s=1)) == 1
    bad, good = [pipeline.read(runner.folder(t)/'state.json') for t in selected]
    assert bad['qc']['status'] == 'failed'
    assert good['qc']['status'] == 'done'
    assert 'rectify' not in bad and 'rectify' not in good


def test_l3_stage_barriers_filter_failures_without_running_their_downstream(tmp_path,portable_execution):
    from types import SimpleNamespace
    c=config(tmp_path)
    names=['download','qc','points','rectify','review']
    c.update(start_at='download',run_stages=names,cpu_failure_policy='continue_passed',stages={})
    events=tmp_path/'events.txt'
    for i,name in enumerate(names):
        program=('from pathlib import Path;import sys;'
                 's=Path(sys.argv[1]);s.mkdir(parents=True,exist_ok=True);'
                 'name=sys.argv[2];f=open(sys.argv[3],"a");f.write(name+":"+s.name+"\\n");f.close();'
                 'sys.exit(1) if s.name=="bad_"+name else (s/(name+".ok")).write_text("done")')
        c['stages'][name]=dict(role='cpu',depends=names[i-1:i],
            command=[sys.executable,'-c',program,'{session}',name,str(events)],outputs=[name+'.ok'])
    selected=[dict(batch='batch_1',name='bad_'+name,source=name) for name in names]
    selected.append(dict(batch='batch_1',name='good',source='good'))
    runner=pipeline.Runner(c,'cpu')
    assert pipeline.run_cpu_phases(runner,selected,['0'],SimpleNamespace(watch=False,poll_s=1))==1
    expected=[]
    active=list(selected)
    for name in names:
        expected.extend(name+':'+t['name'] for t in active)
        active=[t for t in active if t['name']!='bad_'+name]
    assert events.read_text().splitlines()==expected
    for index,task in enumerate(selected[:-1]):
        states=pipeline.read(runner.folder(task)/'state.json')
        assert states[names[index]]['status']=='failed'
        assert all(n not in states for n in names[index+1:])
    good=pipeline.read(runner.folder(selected[-1])/'state.json')
    assert all(good[n]['status']=='done' for n in names)


def test_actual_l3_cpu_config_has_points_last_and_current_review_settings():
    import yaml
    c=yaml.safe_load((Path(__file__).parents[1]/'scripts/L3/pipeline.yaml').read_text(encoding='utf-8'))
    pipeline.validate_config(c)
    cpu=[n for n in c['run_stages'] if c['stages'][n]['role']=='cpu']
    assert cpu==['download','qc','rectify','review','points']
    assert c['cpu_failure_policy']=='continue_passed'
    assert c['stages']['points']['depends']==['qc']
    assert c['stages']['rectify']['depends']==['qc']
    assert 'rectify' in c['stages']['review']['depends']
    cmd=c['stages']['review']['command']
    for key,value in [('--model','qwen3.8-27b'),('--structured-model','qwen3.8-27b'),('--sample-fps','1')]:
        assert cmd[cmd.index(key)+1]==value


@pytest.mark.parametrize('level', ['L1', 'L3'])
def test_cpu_batch_range_dry_run(tmp_path, capsys, level):
    import yaml
    c = config(tmp_path)
    c.update(level=level, start_at='download')
    root = Path(c['paths']['input_root'])
    for i in range(1, 4):
        (root / f'batch_{i:07d}.csv').write_text('path\nks3://bucket/session'+str(i)+'\n')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(c))
    assert pipeline.main(['--config', str(path), '--role', 'cpu', '--batch-range', '1:2', '--dry-run']) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row['task']['batch'] for row in rows] == ['batch_0000001','batch_0000002']
    assert not Path(c['paths']['state_root']).exists()


@pytest.mark.parametrize('extra', [
    ['--role','cpu','--batch-range','3:1'],
    ['--role','cpu','--batch-range','0:1'],
    ['--role','cpu','--batch-range','1:2','--batches','batch_0000001'],
    ['--role','gpu','--batch-range','1:2'],
])
def test_invalid_batch_range(extra):
    with pytest.raises(SystemExit) as error:
        pipeline.main(['--config','unused.yaml'] + extra)
    assert error.value.code == 2


def test_batch_indices_isolate_updates_and_gpu_selection(tmp_path, portable_execution):
    c = config(tmp_path)
    tasks = [dict(batch=b, name='session', source=b) for b in ('batch_1','batch_2')]
    runner = pipeline.Runner(c,'cpu')
    for t in tasks:
        session = Path(c['paths']['output_root']) / t['batch'] / t['name']
        session.mkdir(parents=True,exist_ok=True)
        pipeline.write(runner.folder(t)/'state.json', {'rectify':dict(status='done',task=t)})
        pipeline.update_rectified_index(c,session,True)
        pipeline.update_rectified_index(c,session,True)
    assert pipeline.gpu_tasks(c,[]) == tasks
    assert pipeline.gpu_tasks(c,['batch_2']) == tasks[1:]
    assert len(pipeline.read(pipeline.rectified_index(c,'batch_1'))) == 1
    before = pipeline.rectified_index(c,'batch_2').read_bytes()
    pipeline.update_rectified_index(c,Path(c['paths']['output_root'])/'batch_1/session',False)
    assert pipeline.gpu_tasks(c,[]) == tasks[1:]
    assert pipeline.rectified_index(c,'batch_2').read_bytes() == before
    assert not (Path(c['paths']['output_root'])/'rectification_done.json').exists()
    pipeline.write(pipeline.rectified_index(c,'batch_1'),
                   [str((Path(c['paths']['output_root'])/'batch_2/session').resolve())])
    with pytest.raises(ValueError,match='indexed batch'):
        pipeline.gpu_tasks(c,[])




def test_temporary_l1_recovery_phase_order_and_skips(tmp_path, portable_execution, monkeypatch):
    from scripts.L1 import temporary_rerun as recovery
    from scripts.pipeline import check
    from types import SimpleNamespace
    root = tmp_path/'L1_lerobot'
    sessions = [root/'batch_0000001'/('cp_'+name) for name in ('existing','new','bad')]
    for s in sessions:
        video=s/'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
        video.parent.mkdir(parents=True);video.touch()
        (s/'meta').mkdir();(s/'meta/info.json').write_text('{}')
    for session in sessions:
        (session/'data/chunk-000/episode_000000').mkdir(parents=True)
    right=sessions[0]/'videos/chunk-000/observation.images.head_right_rectified/episode_000000.mp4'
    right.parent.mkdir(parents=True);right.touch()
    (sessions[0]/'rectify.ok').touch()
    job=sessions[0]/'annotation/argus/run_final'
    (job/'out').mkdir(parents=True)
    (job/'run.json').write_text('{"status":"done"}')
    (job/'out/episode_000000.json').write_text('{"labels":[1]}')
    manifest=tmp_path/'inventory.json'
    recovery.save(manifest,[dict(session=str(s)) for s in sessions])
    def validate(stage, session, level):
        if not (session/(stage+'.ok')).exists(): raise ValueError('incomplete')
    monkeypatch.setattr(check,'validate',validate)
    calls=[]
    def execute(cmd, **kwargs):
        if '--data' in cmd:
            stage='review';session=Path(cmd[cmd.index('--data')+1]).parents[3]
        else:
            stage=next(n for n in ('qc','rectify','vio') if
                {'qc':'lerobot_headband_integrity','rectify':'lerobot_stereo_rectification','vio':'lerobot_vio'}[n] in cmd[1])
            session=Path(cmd[cmd.index('--source')+1])
        calls.append((stage,session.name))
        if stage=='rectify':
            assert '--no-skip-existing' in cmd
            right=session/'videos/chunk-000/observation.images.head_right_rectified/episode_000000.mp4'
            right.parent.mkdir(parents=True,exist_ok=True);right.touch()
        if stage=='qc' and session.name=='cp_bad': return SimpleNamespace(returncode=1)
        (session/(stage+'.ok')).touch()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(recovery.subprocess,'run',execute)
    output=tmp_path/'results'
    assert recovery.main(['--manifest',str(manifest),'--root',str(root),'--output',str(output),
                          '--workers','1','--review-workers','1','--execute']) == 1
    assert calls == [('qc','cp_existing'),('qc','cp_new'),('qc','cp_bad'),
                     ('rectify','cp_new'),('vio','cp_existing'),('vio','cp_new'),('review','cp_new')]
    assert recovery.semantic_complete(sessions[0])
    assert pipeline.read(output/'rectify.json')[0]['status']=='skipped_complete'
    assert len(pipeline.read(output/'failures.json')) == 1
    index=root/'batch_0000001/rectification_done.json'
    assert set(pipeline.read(index)) == set(map(str,sessions[:2]))


def test_temporary_rectify_presence_without_decoding(tmp_path):
    from scripts.L1.temporary_rerun import rectified_present
    assert not rectified_present(tmp_path)
    for i in range(2):
        episode=f'episode_{i:06d}'
        (tmp_path/'data/chunk-000'/episode).mkdir(parents=True)
        for side in ('left','right'):
            video=tmp_path/'videos/chunk-000'/f'observation.images.head_{side}_rectified'/f'{episode}.mp4'
            video.parent.mkdir(parents=True,exist_ok=True)
            video.touch()  # Empty files accepted: explicitly presence only, no decoder involved.
    assert rectified_present(tmp_path)
    video.unlink()
    assert not rectified_present(tmp_path)
