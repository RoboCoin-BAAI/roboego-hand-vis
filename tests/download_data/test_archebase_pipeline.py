import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.archebase import pipeline as p
from roboego_hand_vis.upload_data.ks3 import atomic_json

SOURCE='ks3://baai-eai-pwm/archebase_1000/scene/task/collector/186460/'
BATCH='batch_0000001'


def arguments(tmp_path):
    return SimpleNamespace(local_root=tmp_path/'local',state_root=tmp_path/'state',destination=p.DESTINATION,
        python='python',converter_repo=tmp_path/'converter',argus_repo=tmp_path/'argus',
        model='qwen3.6-plus',balance=0.,ffmpeg='ffmpeg')


def mocked_pipeline(tmp_path,monkeypatch,*,remote_ok=True,model_ok=True):
    args=arguments(tmp_path)
    events=[]
    class Client:
        def objects(self,uri):
            if uri==SOURCE:
                return [dict(uri=uri+n,size=4,etag='opaque') for n in ('data.mcap','annotation.json')]
            state=json.loads((args.state_root/BATCH/'records/186460.json').read_text())
            assert state['status']=='uploaded'  # Durable receipt before cleanup.
            events.append('verify')
            return [dict(uri=uri+n,size=size if remote_ok else size+1) for n,size in self.files.items()]
        def download(self,item,target):
            events.append('download')
            target.write_bytes(b'mcap' if target.name=='data.mcap' else b'{}')
        def upload_directory(self,dataset,destination):
            assert dataset.name=='dataset'
            assert not (dataset/'data.mcap').exists()
            events.append('upload')
            self.files=p.file_sizes(dataset)
            return dict(file_sizes=self.files)
    def stage(command,log,cwd=None):
        if 'convert_archebase_mcap.py' in command[1]:
            events.append('convert')
            dataset=Path(command[command.index('--out')+1]);dataset.mkdir()
            atomic_json(dataset/'meta/archebase_source.json',dict(source_sha256=hashlib.sha256(b'mcap').hexdigest()))
            atomic_json(dataset/'annotation/episode_000000_source.json',dict(segments=[]))
            (dataset/'meta/tasks.jsonl').write_text('{"task_index":0,"task":"186460"}\n')
            (dataset/'meta/episodes.jsonl').write_text('{"episode_index":0,"tasks":["186460"]}\n')
        elif command[2].endswith('archebase.run') and 'lerobot_stereo_rectification' in command[2]:
            events.append('undistort')
            dataset=Path(command[command.index('--source')+1])
            atomic_json(dataset/'calibrations/archebase_undistortion.json',{})
        else:
            events.append('annotate')
            if not model_ok:
                raise RuntimeError('Model request failed')
            output=Path(command[command.index('--out')+1])
            atomic_json(output,dict(task_name='Organize books',task_description='Put books in a tray.',subtask=[]))
            atomic_json(output.with_name('.'+output.name+'.claim.json'),dict(status='complete',
                        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    monkeypatch.setattr(p,'run_stage',stage)
    monkeypatch.setattr(p,'validate_undistortion',lambda *a:None)
    monkeypatch.setattr(p,'validate_annotation',lambda *a:None)
    return args,events,Client()


def test_full_stage_order_and_cleanup_after_durable_upload(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch)
    record=p.process(SOURCE,BATCH,args,client)
    assert events==['download','download','convert','undistort','annotate','upload','verify']
    assert record['status']=='complete'
    assert not (args.local_root/BATCH/'186460').exists()
    assert (args.state_root/BATCH/'logs/186460/model/episode_000000_structured.json').is_file()


def test_failed_remote_check_retains_raw_and_dataset(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch,remote_ok=False)
    result=p.process(SOURCE,BATCH,args,client)
    assert result['status']=='uploaded' and result.get('error')
    work=args.local_root/BATCH/'186460'
    assert (work/'raw/data.mcap').exists() and (work/'dataset/meta/archebase_source.json').exists()


def test_failed_model_never_uploads_and_is_not_automatically_retried(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch,model_ok=False)
    result=p.process(SOURCE,BATCH,args,client)
    assert result['status']=='annotating' and 'error' in result
    assert 'upload' not in events
    previous=list(events)
    p.process(SOURCE,BATCH,args,client)
    assert events==previous


def test_download_barrier_stops_before_conversion(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch)
    result=p.process(SOURCE,BATCH,args,client,stage='download')
    assert result['status']=='downloaded'
    assert events==['download','download']


def test_upload_barrier_does_not_delete(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch)
    p.process(SOURCE,BATCH,args,client,stage='download')
    p.process(SOURCE,BATCH,args,client,stage='prepare')
    result=p.process(SOURCE,BATCH,args,client,stage='upload')
    assert result['status']=='uploaded'
    assert 'verify' not in events
    assert (args.local_root/BATCH/'186460/raw/data.mcap').exists()
    p.process(SOURCE,BATCH,args,client,stage='cleanup')
    assert events[-1]=='verify'


def test_changed_model_cannot_reuse_pipeline_state(tmp_path,monkeypatch):
    args,events,client=mocked_pipeline(tmp_path,monkeypatch)
    p.process(SOURCE,BATCH,args,client,stage='download')
    args.model='qwen-other'
    with pytest.raises(ValueError,match='changed'):
        p.process(SOURCE,BATCH,args,client,stage='prepare')


def test_cleanup_rejects_changed_local_files_and_unsafe_paths(tmp_path):
    work=tmp_path/BATCH/'186460'
    work.mkdir(parents=True)
    (work/'x').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        p.cleanup(work,tmp_path,{'x':1})
    with pytest.raises(ValueError,match='Unsafe'):
        p.cleanup(tmp_path,tmp_path,{})
    assert work.exists()


def test_annotation_command_is_text_cleaning_not_video_annotation(tmp_path):
    args=arguments(tmp_path)
    command=p.annotation_command(args,tmp_path/'dataset',tmp_path/'state/final.json')
    assert 'review.archebase.run' in command
    assert '--sample-fps' not in command
    assert command[command.index('--input')+1].endswith('episode_000000_source.json')
