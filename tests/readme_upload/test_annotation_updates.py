import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
import run_annotation_updates as m


def fixture(tmp_path):
    root=tmp_path/'data'
    path=root/'batch_0000036/sample/annotation/episode_000000_structured.json'
    path.parent.mkdir(parents=True)
    data=dict(objects=[dict(name='red cup',attributes={'color':'red'})],actions={'vocabulary':['wash']},
              task_description='Wash cup',task_name='old name',scene_classification={'area':'kitchen'},
              subtask=[{'action':'wash','object':'red cup'}])
    path.write_text(json.dumps(data))
    state=tmp_path/'state';state.mkdir()
    row=dict(path=path.relative_to(root).as_posix(),status='done',before=m.terms.source_fields(data),
             after=dict(object_names=['cup'],action_vocabulary=['wash']))
    (state/'prepare_progress.jsonl').write_text(json.dumps(row)+'\n')
    args=SimpleNamespace(dataset='jianzhi',root=root,terms_state_dir=state,argus_root=tmp_path/'argus',
                         report_dir=tmp_path/'report',start=36,end=36,model='qwen3.7-plus',workers=2,
                         task_max_tokens=4096,batch_size=32,apply=False,retry_failed=False)
    return args,path,data


def test_preview_and_incomplete_refuse_without_writing(tmp_path):
    args,path,_=fixture(tmp_path);raw=path.read_bytes()
    assert m.run(args)==0
    assert not args.report_dir.exists() and path.read_bytes()==raw
    (args.terms_state_dir/'prepare_progress.jsonl').write_text('')
    args.apply=True
    with pytest.raises(ValueError,match='no completed cleaning record'):m.run(args)
    assert not args.report_dir.exists() and path.read_bytes()==raw


def test_pipeline_resume_and_preserve_unrelated_fields(tmp_path,monkeypatch):
    args,path,data=fixture(tmp_path);args.apply=True
    calls=[]
    monkeypatch.setattr(m.tasks,'review_modules',lambda p:(None,SimpleNamespace(job_lock=lambda p:contextlib.nullcontext())))
    def prepare(path,cfg,endpoint,verbs):
        calls.append('task')
        current=json.loads(path.read_text())
        return dict(path=path.relative_to(cfg.root).as_posix(),status='done',source=m.tasks.evidence(current),
                    before=current['task_name'],before_present=True,after='wash cup',
                    decision=dict(verb='wash',object_name='cup',object_indices=[0],reason='Wash cup'))
    def classify(rows,cfg,endpoint,taxonomy):
        calls.append('common')
        return [dict(dataset=r['dataset'],path=r['path'],status='done',common_task_name='object_cleaning',source=r) for r in rows]
    monkeypatch.setattr(m.tasks,'prepare_one',prepare)
    monkeypatch.setattr(m.common,'classify_batch',classify)
    assert m.run(args)==0
    current=json.loads(path.read_text())
    expected=dict(data);expected['objects']=[dict(name='cup',attributes={'color':'red'})]
    expected.update(task_name='wash cup',common_task_name='object_cleaning')
    assert current==expected
    assert m.run(args)==0 and calls==['task','common']


def test_stage_repairs_torn_append_and_retries_only_errors(tmp_path):
    p=tmp_path/'work.jsonl'
    p.write_text('{"path":"a","status":"done"}\n{"path":')
    calls=[]
    def work(row):calls.append(row['path']);return dict(row,status='done')
    rows=[{'path':'a'},{'path':'b'}]
    assert len(m.stage(tmp_path,'work',rows,work,1))==2
    assert calls==['b']
    assert all(json.loads(line) for line in p.read_text().splitlines())


def test_select_ignores_pipeline_logs_but_checks_real_samples(tmp_path):
    args,path,_=fixture(tmp_path)
    batch=path.parents[2]
    for name in ('_batch_logs','logs','.cache'):
        (batch/name).mkdir()
    assert m.select(args)==[path.relative_to(args.root).as_posix()]
    (batch/'another_sample').mkdir()
    with pytest.raises(ValueError,match='Missing or unsafe annotation'):
        m.select(args)


def test_overwritten_progress_can_be_recovered_without_api(tmp_path,monkeypatch):
    args,path,data=fixture(tmp_path)
    original=json.loads((args.terms_state_dir/'prepare_progress.jsonl').read_text())
    (args.terms_state_dir/'prepare_progress.jsonl').write_text('')
    args.recover_cached_terms=True
    def cached(p,cfg,endpoint,objects,actions):
        assert endpoint is m.NoAPI and not cfg.apply and not cfg.retry_failed
        with pytest.raises(RuntimeError,match='forbidden'):endpoint.settings(cfg.model)
        return original
    monkeypatch.setattr(m.terms,'prepare_one',cached)
    assert m.run(args)==0
    assert json.loads(path.read_text())==data and not args.report_dir.exists()
