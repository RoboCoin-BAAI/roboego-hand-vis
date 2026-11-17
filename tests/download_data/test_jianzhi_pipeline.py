import json
from types import SimpleNamespace

import pytest

from scripts.jianzhi.pipeline import (
    DESTINATION, annotation_command, cleanup, process, verify_remote,
)
from roboego_hand_vis.upload_data.ks3 import atomic_json

NAME = 'a' * 32


@pytest.mark.parametrize('fail_readme', [False, True])
def test_annotation_then_readme_before_upload(tmp_path, monkeypatch, fail_readme):
    from scripts.jianzhi import pipeline
    args = SimpleNamespace(local_root=tmp_path/'local', state_root=tmp_path/'state',
                           destination=DESTINATION, python='python', model='qwen3.7-plus',
                           argus_repo=tmp_path/'argus')
    source = 'ks3://baai-eai-pwm/jianzhi/business/'+NAME+'/'
    work = args.local_root/'batch_0000001'/NAME
    work.mkdir(parents=True)
    atomic_json(args.state_root/'batch_0000001/records'/(NAME+'.json'), dict(
        source=source, destination=DESTINATION+'batch_0000001/'+NAME+'/',
        local_path=str(work), status='rectified'))
    events=[]
    def run(command, log, cwd=None):
        events.append(command[2])
        if command[2]=='roboego_hand_vis.build_json.base.build_readme':
            if fail_readme:
                raise RuntimeError('readme failed')
            atomic_json(work/'meta/readme.json',dict(dataset_name='test'))
    monkeypatch.setattr(pipeline,'run_stage',run)
    monkeypatch.setattr(pipeline,'validate_annotation',lambda _: None)
    result=process(source,'batch_0000001',args,None,stage='annotate')
    assert events==['review.jianzhi.structure_existing','roboego_hand_vis.build_json.base.build_readme']
    assert result['status']==('rectified' if fail_readme else 'annotated')
    assert bool(result.get('error'))==fail_readme


def test_runtime_artifacts_moved_outside_dataset(tmp_path):
    from scripts.jianzhi.pipeline import relocate_runtime_files
    work = tmp_path/'dataset'
    extra = [
        'annotation/.episode_000000_structured.json.execution.lock',
        'annotation/.episode_000000_structured_text.claim.json',
        'videos/chunk-000/observation.images.head_left_rectified/episode_000000.ffmpeg.log',
        '.rectification.partial-old/videos/partial.mp4',
    ]
    for name in extra + ['annotation/episode_000000_structured.json', 'meta/info.json']:
        path = work/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    relocate_runtime_files(work, tmp_path/'state')
    for name in extra:
        assert not (work/name).exists()
        assert (tmp_path/'state'/name).read_text() == 'original'
    assert (work/'meta/info.json').exists()
    assert (work/'annotation/episode_000000_structured.json').exists()


def test_requested_model_and_sampling_have_no_incompatible_budget():
    args = SimpleNamespace(python='python', model='qwen3.7-plus', sample_fps=1.,
                           annotation_budget_cny=None)
    command = annotation_command(args, '/home/jianzhi/batch_0000001/'+NAME)
    assert command[command.index('--model')+1] == 'qwen3.7-plus'
    assert command[2] == 'review.jianzhi.structure_existing'
    assert '--sample-fps' not in command
    assert '--estimated-cap-cny' not in command


def test_remote_mismatch_rejected():
    client = SimpleNamespace(objects=lambda uri: [dict(uri=uri+'meta/info.json', size=1)])
    with pytest.raises(ValueError):
        verify_remote(client, DESTINATION, {'meta/info.json': 2})


def test_cleanup_rejects_changed_files_and_root(tmp_path):
    work = tmp_path / 'batch_0000001' / NAME
    work.mkdir(parents=True)
    (work/'data.txt').write_text('changed')
    with pytest.raises(ValueError):
        cleanup(work, tmp_path, {'data.txt':1})
    assert work.exists()
    with pytest.raises(ValueError):
        cleanup(tmp_path, tmp_path, {})


@pytest.mark.parametrize('remote_valid', [False, True])
def test_resume_upload_verifies_before_cleanup(tmp_path, remote_valid):
    args = SimpleNamespace(local_root=tmp_path/'local', state_root=tmp_path/'state', destination=DESTINATION)
    work = args.local_root/'batch_0000001'/NAME
    work.mkdir(parents=True)
    (work/'data.txt').write_text('abc')
    destination = DESTINATION+'batch_0000001/'+NAME+'/'
    source = 'ks3://baai-eai-pwm/jianzhi/business/'+NAME+'/'
    state = args.state_root/'batch_0000001/records'/(NAME+'.json')
    atomic_json(state, dict(source=source, destination=destination, local_path=str(work),
        status='uploaded', upload={'file_sizes':{'data.txt':3}}))
    client = SimpleNamespace(objects=lambda uri: [dict(uri=uri+'data.txt',size=3 if remote_valid else 2)])
    result = process(source,'batch_0000001',args,client)
    assert work.exists() != remote_valid
    assert (result['status'] == 'complete') == remote_valid
    assert json.loads(state.read_text())['status'] == result['status']


def test_batch_stage_barriers_and_worker_counts(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from scripts.jianzhi import pipeline
    events = []
    class Pool:
        def __init__(self, max_workers):
            events.append(('pool', max_workers))
        def __enter__(self): return self
        def __exit__(self, *args): events.append(('barrier',))
        def submit(self, fn, source, batch, args, stage):
            events.append((stage, source))
            future = Future()
            future.set_result(dict(status={'download':'downloaded', 'rectify':'rectified', 'annotate':'annotated',
                                           'upload':'uploaded','cleanup':'complete'}[stage]))
            return future
    monkeypatch.setattr(pipeline, 'ProcessPoolExecutor', Pool)
    args = SimpleNamespace(rectify_workers=8,annotation_workers=8,upload_workers=4,
                           local_root=tmp_path/'local',state_root=tmp_path/'state')
    monkeypatch.setattr(pipeline, 'stage_worker', lambda source,batch,args,stage: (events.append((stage,source)) or dict(status={'download':'downloaded','upload':'uploaded','cleanup':'complete'}[stage])))
    assert pipeline.run_batch('batch_0000001',['a','b'],args)
    assert events == [('download','a'),('download','b')] + [event for stage,count in [('rectify',8),('annotate',8)]
                      for event in [('pool',count),(stage,'a'),(stage,'b'),('barrier',)]] + [('upload','a'),('upload','b'),('cleanup','a'),('cleanup','b')]


def test_download_stage_does_not_start_rectification(tmp_path, monkeypatch):
    from scripts.jianzhi import pipeline
    args = SimpleNamespace(local_root=tmp_path/'local',state_root=tmp_path/'state',destination=DESTINATION)
    source='ks3://baai-eai-pwm/jianzhi/business/'+NAME+'/'
    work=args.local_root/'batch_0000001'/NAME
    state=args.state_root/'batch_0000001/records'/(NAME+'.json')
    atomic_json(state,dict(source=source,destination=DESTINATION+'batch_0000001/'+NAME+'/',
                          local_path=str(work),status='downloaded'))
    monkeypatch.setattr(pipeline,'run_stage',lambda *a,**k:pytest.fail('Unexpected subprocess'))
    assert process(source,'batch_0000001',args,None,stage='download')['status']=='downloaded'


def test_download_calls_base_without_overriding_jobs(tmp_path, monkeypatch):
    from scripts.jianzhi import pipeline
    args = SimpleNamespace(local_root=tmp_path/'local', state_root=tmp_path/'state',
                           destination=DESTINATION, python='python', ks3_tool='ks3', ks3_config='config')
    source = 'ks3://baai-eai-pwm/jianzhi/business/'+NAME+'/'
    commands = []
    def run(command, log, cwd=None):
        commands.append(command)
        assert pipeline.read_batch(__import__('pathlib').Path(command[command.index('--batch-csv')+1])) == [source]
        work = args.local_root/'batch_0000001'/NAME/'meta'
        work.mkdir(parents=True, exist_ok=True)
        (work/'info.json').write_text('{}')
    monkeypatch.setattr(pipeline, 'run_stage', run)
    assert process(source,'batch_0000001',args,None,stage='download')['status']=='downloaded'
    assert len(commands) == 1
    assert commands[0][2] == 'roboego_hand_vis.download_data.base.run'
    assert not any(arg.startswith('--jobs') for arg in commands[0])


def test_rectify_stage_does_not_start_annotation(tmp_path, monkeypatch):
    from scripts.jianzhi import pipeline
    args = SimpleNamespace(local_root=tmp_path/'local', state_root=tmp_path/'state',
                           destination=DESTINATION, python='python', ffmpeg='ffmpeg')
    source = 'ks3://baai-eai-pwm/jianzhi/business/'+NAME+'/'
    work = args.local_root/'batch_0000001'/NAME
    atomic_json(args.state_root/'batch_0000001/records'/(NAME+'.json'),
                dict(source=source, destination=DESTINATION+'batch_0000001/'+NAME+'/',
                     local_path=str(work), status='downloaded'))
    calls = []
    monkeypatch.setattr(pipeline, 'run_stage', lambda cmd,*a: calls.append(cmd))
    assert process(source,'batch_0000001',args,None,stage='rectify')['status']=='rectified'
    assert len(calls) == 1
    assert 'roboego_hand_vis.lerobot_stereo_rectification.jianzhi.run' in calls[0]


def test_failed_item_excluded_from_later_stages_and_reruns(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from scripts.jianzhi import pipeline
    calls = []
    class Pool:
        def __init__(self, max_workers): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, source, batch, args, stage):
            calls.append((source,stage))
            future=Future()
            if source=='bad':
                future.set_result(dict(status='downloaded',error='rectification failed'))
            else:
                future.set_result(dict(status={'download':'downloaded','rectify':'rectified', 'annotate':'annotated',
                                               'upload':'uploaded','cleanup':'complete'}[stage]))
            return future
    monkeypatch.setattr(pipeline,'ProcessPoolExecutor',Pool)
    args=SimpleNamespace(rectify_workers=8,annotation_workers=8,upload_workers=4,
                         local_root=tmp_path/'local',state_root=tmp_path/'state')
    monkeypatch.setattr(pipeline, 'stage_worker', lambda source,batch,args,stage: (calls.append((source,stage)) or dict(status={'download':'downloaded','upload':'uploaded','cleanup':'complete'}[stage], **({'error':'download failed'} if source=='bad' else {}))))
    assert not pipeline.run_batch('batch_0000001',['bad','good'],args)
    assert ('good','upload') in calls and ('good','cleanup') in calls
    assert [stage for source,stage in calls if source=='bad']==['download']
    calls.clear()
    assert not pipeline.run_batch('batch_0000001',['bad','good'],args)
    assert all(source!='bad' for source,stage in calls)
