import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

BASE=Path(__file__).resolve().parents[2]/'roboego_hand_vis/upload_data/readme_upload/guanglun'
sys.path.insert(0,str(BASE))
spec=importlib.util.spec_from_file_location('common_task_test',BASE/'classify_common_tasks_qwen.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_packet_ids_and_other_reasons_are_strict():
    def payload(items):return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':items})}}]}
    classes={'object_cleaning':{},'other':{}}
    result=m.parse_response(payload([{'id':1,'common_task_name':'other','reason':'Unrelated purposes'},
                                    {'id':0,'common_task_name':'object_cleaning'}]),2,classes)
    assert result[0]['common_task_name']=='object_cleaning'
    for items in ([{'id':True,'common_task_name':'object_cleaning'}],
                  [{'id':0,'common_task_name':'invented'}],
                  [{'id':0,'common_task_name':'other'}]):
        with pytest.raises(ValueError):m.parse_response(payload(items),1,classes)


def test_add_common_field_preserves_every_other_byte(tmp_path):
    raw=b'\xef\xbb\xbf{\r\n"task_name":"wipe table", "keep":1.00, "objects":[{"name":"cloth"}]\r\n}\r\n'
    out=m.patch_common(raw,'object_cleaning')
    assert out.startswith(b'\xef\xbb\xbf') and b'"task_name":"wipe table"' in out and b'"keep":1.00' in out
    expected=json.loads(raw.decode('utf-8-sig'));expected['common_task_name']='object_cleaning'
    assert json.loads(out.decode('utf-8-sig'))==expected


def test_apply_backup_idempotence_and_stale_evidence(tmp_path):
    root=tmp_path/'data';p=root/'batch_0000001/sample/annotation/episode_000000_structured.json';p.parent.mkdir(parents=True)
    data=dict(task_name='wash cup',task_description='Wash cup',objects=[dict(name='cup')],actions={'vocabulary':['wash']})
    raw=json.dumps(data).encode();p.write_bytes(raw)
    source=dict(task_name='wash cup',task_description='Wash cup',object_names=['cup'])
    row=dict(dataset='jianzhi',path=p.relative_to(root).as_posix(),common_task_name='object_cleaning',source=source)
    args=SimpleNamespace(jianzhi_root=root.resolve(),guanglun_root=tmp_path/'unused',report_dir=tmp_path/'reports')
    result=m.apply_one(row,args,{'object_cleaning':{}});assert result['changed']
    assert (args.report_dir/'original_annotations/jianzhi'/row['path']).read_bytes()==raw
    assert not m.apply_one(row,args,{'object_cleaning':{}})['changed']
    latest=json.loads(p.read_bytes());latest['task_description']='Changed';p.write_text(json.dumps(latest))
    with pytest.raises(ValueError,match='evidence changed'):m.apply_one(row,args,{'object_cleaning':{}})


def test_existing_different_common_field_is_not_overwritten(tmp_path):
    root=tmp_path/'data';p=root/'batch_0000001/sample/annotation/episode_000000_structured.json';p.parent.mkdir(parents=True)
    data=dict(task_name='wash cup',common_task_name='other',objects=[]);p.write_text(json.dumps(data));raw=p.read_bytes()
    row=dict(dataset='jianzhi',path=p.relative_to(root).as_posix(),common_task_name='object_cleaning',source=dict(task_name='wash cup',object_names=[]))
    args=SimpleNamespace(jianzhi_root=root.resolve(),guanglun_root=tmp_path/'unused',report_dir=tmp_path/'reports')
    with pytest.raises(ValueError,match='refusing overwrite'):m.apply_one(row,args,{'object_cleaning':{}})
    assert p.read_bytes()==raw
