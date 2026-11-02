import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.lerobot_hand_pose.compute import process, plans, vio_succeeded


def test_current_vio_selection_including_zupt(tmp_path):
    root=tmp_path.resolve()
    episode=root/'data/chunk-000/episode_000000';episode.mkdir(parents=True)
    assert not vio_succeeded(root,episode)
    report=root/'vio_zupt/chunk-000/episode_000000/report.json';report.parent.mkdir(parents=True)
    good=dict(status='complete',initialized=True,quality_status='passed_basic_checks',
              quality=dict(status='passed_basic_checks'))
    report.write_text(json.dumps(good))
    provenance=episode/'relative_head_pose_provenance.json'
    provenance.write_text(json.dumps(dict(report=str(report.relative_to(root)),published_to_lerobot=True,
                                          quality=dict(status='passed_basic_checks'))))
    for name in ('observation.relative_head_pose.parquet','observation.relative_head_pose_status.parquet'):
        (episode/name).touch()
    assert vio_succeeded(root,episode)
    report.write_text(json.dumps(dict(good,quality_status='failed')))
    assert not vio_succeeded(root,episode)
    assert plans(root,'L1',vio_success_only=True)==[]


@pytest.mark.parametrize('level,camera',[('L1','head_left_rectified'),('L3','headband_left_rgb_rectified')])
def test_model_adapter_and_failure_preserves_result(tmp_path, monkeypatch, level, camera):
    episode=tmp_path/'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    video=tmp_path/'videos/chunk-000'/f'observation.images.{camera}'/'episode_000000.mp4'
    video.parent.mkdir(parents=True);video.write_bytes(b'video')
    timestamps=episode/f'observation.images.{camera}.parquet'
    pq.write_table(pa.table({'timestamp':[0.,.03], 'episode_index':[0,0]}),timestamps)
    model=tmp_path/'model';model.mkdir();(model/'infer.py').touch()
    def backend(command, **kwargs):
        kwargs['log_path'].write_text('model loading\n', encoding='utf-8')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']=='3'
        assert command[command.index('--device')+1]=='cuda:0'
        out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
        np.savez(out/'prediction.npz',frame_indices=np.arange(2),joints_3d=np.zeros((2,2,21,3)))
        (out/'metadata.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('roboego_hand_vis.lerobot_hand_pose.compute.run_logged',backend)
    options=dict(model_root=model,python='python',gpu='3')
    process(tmp_path,level,dry_run=True,**options)
    assert not (tmp_path/'hand').exists()
    process(tmp_path,level,**options)
    output=tmp_path/'hand/chunk-000/episode_000000'
    assert (output/'source_timestamps.parquet').read_bytes()==timestamps.read_bytes()
    receipt=json.loads((output/'lerobot_hand.json').read_text())
    assert receipt['camera_feature']==f'observation.images.{camera}'
    assert not receipt['calibrated_intrinsics_used_for_3d']
    assert process(tmp_path,level,**options)[0]['skipped_existing']
    previous=(output/'prediction.npz').read_bytes()
    monkeypatch.setattr('roboego_hand_vis.lerobot_hand_pose.compute.run_logged',lambda *a,**k:SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError):
        process(tmp_path,level,skip_existing=False,**options)
    assert (output/'prediction.npz').read_bytes()==previous


def test_model_output_is_streamed_and_saved(tmp_path, capsys):
    import os
    import sys
    from roboego_hand_vis.lerobot_hand_pose.compute import run_logged
    log=tmp_path/'model.log'
    result=run_logged([sys.executable,'-u','-c',
                       'import sys;print("loading model",flush=True);print("model error",file=sys.stderr);sys.exit(3)'],
                      cwd=tmp_path,env=os.environ,log_path=log,label='GPU=2 sample')
    assert result.returncode==3
    assert 'loading model' in log.read_text()
    assert 'model error' in log.read_text()
    console=capsys.readouterr().out
    assert '[GPU=2 sample] loading model' in console
    assert 'model PID=' in console


def test_standalone_unified_startup_log(tmp_path):
    from roboego_hand_vis.lerobot_hand_pose.run import main
    source=tmp_path/'L1_lerobot'
    (source/'session/meta').mkdir(parents=True)
    (source/'session/meta/info.json').write_text('{}')
    model=tmp_path/'model';model.mkdir();(model/'infer.py').touch()
    assert main('L1',['--source',str(source),'--model-root',str(model),'--dry-run'])==1
    text=(tmp_path/'pipeline_tasks/L1/hand_standalone/console.log').read_text(encoding='utf-8')
    assert '[hand-start]' in text
    assert '[hand-scan] found 1 sessions' in text
    assert 'No LeRobot episodes' in text


def test_vio_state_selection_without_dataset_recursion(tmp_path, monkeypatch):
    from roboego_hand_vis.lerobot_hand_pose.selection import from_pipeline_state
    source=tmp_path/'L1_lerobot';source.mkdir()
    state=tmp_path/'L1_pipeline_state/batch_0000042/hash'
    state.mkdir(parents=True)
    good='cp_0001_session_20261009_172508'
    bad='cp_0002_session_20261009_181412'
    def record(name,status):
        return json.dumps(dict(episode=f'/old/machine/data/L1_lerobot/batch_0000042/{name}/data/chunk-000/episode_000000',status=status))
    (state/'vio.100.log').write_text('\n'.join([record(good,'done'),record(bad,'done'),record(bad,'failed_quality')]))
    monkeypatch.setattr(Path,'rglob',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('No recursive scanning')))
    assert from_pipeline_state(source,'L1')==[source/'batch_0000042'/good]
    assert from_pipeline_state(source/'batch_0000042','L1')==[source/'batch_0000042'/good]


def test_resident_predictor_reused_for_multiple_sessions(tmp_path):
    import sys
    from roboego_hand_vis.lerobot_hand_pose.worker import run_jobs
    model=tmp_path/'model';model.mkdir();(model/'infer.py').touch()
    tasks=[]
    for name in ('session_a','session_b'):
        root=tmp_path/name
        episode=root/'data/chunk-000/episode_000000';episode.mkdir(parents=True)
        key='observation.images.head_left_rectified'
        video=root/'videos/chunk-000'/key/'episode_000000.mp4'
        video.parent.mkdir(parents=True);video.write_bytes(b'video')
        pq.write_table(pa.table({'timestamp':[0.,.03]}),episode/f'{key}.parquet')
        tasks.extend({k:str(v) for k,v in p.items()} for p in plans(root,'L1'))
    class Predictor:
        def __init__(self):
            self.calls=[]
        def run(self,video,output,**kwargs):
            self.calls.append(str(video))
            output.mkdir(parents=True)
            np.savez(output/'prediction.npz',frame_indices=np.arange(2),joints_3d=np.zeros((2,2,21,3)))
            (output/'metadata.json').write_text('{}')
            kwargs['progress'](.5,'inference')
    predictor=Predictor()
    request=dict(level='L1',tasks=tasks,results=str(tmp_path/'results.jsonl'),options=dict(
        model_root=str(model),python=sys.executable,gpu='0',precision='auto',window_batch_size=1,
        resize_mode='long_edge_pad',skip_existing=True,vio_success_only=False))
    assert run_jobs(request,predictor)==0
    assert len(predictor.calls)==2
    assert len((tmp_path/'results.jsonl').read_text().splitlines())==2
    for name in ('session_a','session_b'):
        output=tmp_path/name/'hand/chunk-000/episode_000000'
        assert (output/'prediction.npz').exists()
        assert 'inference' in (output/'model.log').read_text()
