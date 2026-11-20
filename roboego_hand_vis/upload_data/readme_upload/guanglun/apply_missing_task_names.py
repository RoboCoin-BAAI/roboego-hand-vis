"""Apply a reviewed missing-task plan. Changes only task_name, with exact backups."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

from apply_scene_classification import atomic_write, object_value_spans
from reclassify_scenes_qwen import save_json


def patch(raw,name):
    text=raw.decode('utf-8-sig');data=json.loads(text);spans=object_value_spans(text)
    if data.get('task_name'):raise ValueError('Refusing to overwrite a nonempty task name')
    if 'task_name' in spans:
        first,last=spans['task_name'];updated=text[:first]+json.dumps(name)+text[last:]
    else:
        end=max((v[1] for v in spans.values()),default=text.index('{')+1)
        updated=text[:end]+(',' if spans else '')+'\n  "task_name": '+json.dumps(name)+text[end:]
    expected=dict(data,task_name=name)
    if json.loads(updated)!=expected:raise ValueError('Unexpected field modification')
    return (b'\xef\xbb\xbf' if raw.startswith(b'\xef\xbb\xbf') else b'')+updated.encode('utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args();plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    if plan.get('scope')!=['task_name']:raise ValueError('Unexpected scope')
    root=Path(plan['root']).resolve();report=a.report_dir.resolve();prepared=[];seen=set()
    if report==root or root in report.parents:raise ValueError('Report must be outside publication')
    for r in plan['records']:
        relative=Path(r['path'])
        if not re.fullmatch(r'batch_\d+/[a-f0-9]+/annotation/episode_000000_structured\.json',relative.as_posix()) or r['path'] in seen:raise ValueError('Unsafe/duplicate path')
        seen.add(r['path']);path=root/relative
        if path.is_symlink() or root not in path.resolve().parents:raise ValueError('Unsafe target')
        raw=path.read_bytes();d=json.loads(raw.decode('utf-8-sig'))
        if not re.fullmatch('[a-z]+ [a-z]+(?: [a-z]+)*',r['task_name']):raise ValueError('Invalid name')
        if d.get('task_name')==r['task_name']:
            prepared.append((r,path,raw,None));continue
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Source changed: '+r['path'])
        revised=patch(raw,r['task_name'])
        backup=report/'original_annotations'/relative
        if backup.exists() and backup.read_bytes()!=raw:raise ValueError('Backup mismatch')
        prepared.append((r,path,raw,revised))
    if not a.apply:print(json.dumps(dict(selected=len(prepared),apply=False)));return
    results=[]
    for r,path,raw,revised in prepared:
        if revised is not None:
            backup=report/'original_annotations'/r['path'];backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():
                with backup.open('xb') as stream:stream.write(raw)
            if path.read_bytes()!=raw:raise ValueError('Concurrent modification')
            atomic_write(path,revised,stat.S_IMODE(path.stat().st_mode))
            if path.read_bytes()!=revised:raise ValueError('Verification failed')
        results.append(dict(path=r['path'],task_name=r['task_name'],changed=revised is not None,status='done',reason=r['reason']))
        save_json(report/'apply_report.json',dict(scope=['task_name'],records=results))
    print(json.dumps(dict(selected=len(results),changed=sum(r['changed'] for r in results),verified=True)))


if __name__=='__main__':main()
