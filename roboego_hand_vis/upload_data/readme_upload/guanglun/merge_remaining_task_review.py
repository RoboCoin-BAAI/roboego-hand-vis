"""Assemble per-sample follow-up proposals, preserving prior and source snapshots."""
from collections import Counter,defaultdict
import json
from pathlib import Path

from build_task_name_mapping import write_csv

BASE=Path(__file__).resolve().parent
OUT=BASE/'remaining_task_review_20261119'


def load(p):return json.loads((BASE/p).read_text(encoding='utf-8'))


def main():
    source=load('remaining_task_evidence_1058_20261119.json')['records']
    timed=load('remaining_task_review_20261119/review.json')['records']
    manual=load('remaining_task_review_20261119/assistant_decisions.json')['records']
    by_key={(r['dataset'],r['path']):r for r in timed}
    assert len(by_key)==len(source)==1058
    for row in manual:
        key=row['dataset'],row['path']
        assert key in by_key and row['sha256']==by_key[key]['sha256'] and row['old_task_name']==by_key[key]['old_task_name']
        by_key[key]=row
    fallback_path=OUT/'user_fallback_578.json'
    fallback_count=0
    if fallback_path.exists():
        fallback=json.loads(fallback_path.read_text(encoding='utf-8'))
        assert fallback['task_name']=='origanize_some_object' and len(fallback['records'])==578
        assert len({(r['dataset'],r['path']) for r in fallback['records']})==578
        for entry in fallback['records']:
            key=entry['dataset'],entry['path'];row=by_key[key]
            assert row['sha256']==entry['sha256'] and row['status']=='review'
            row=dict(row,previous_review={k:row[k] for k in ('status','basis','reason')},
                     proposed=fallback['task_name'],status='candidate',basis='user_assigned_fallback',
                     reason='User assigned one shared label to all 578 remaining samples; not a semantic classification')
            by_key[key]=row;fallback_count+=1
    reviewed=[by_key[(r['dataset'],r['path'])] for r in source]
    flat=[dict(dataset=r['dataset'],path=r['path'],old_task_name=r['old_task_name'],proposed=r.get('proposed'),
               status=r['status'],basis=r['basis'],reason=r['reason']) for r in reviewed]
    write_csv(OUT/'reviewed_samples.csv',flat,list(flat[0]))
    write_csv(OUT/'still_unresolved.csv',[r for r in flat if r['status']=='review'],list(flat[0]))
    (OUT/'reviewed_results.json').write_text(json.dumps(dict(annotation_modified=False,model_called=False,records=reviewed),ensure_ascii=False,indent=2),encoding='utf-8')
    inventory=load('task_names_inventory_20261119.json')['records']
    baseline={r['old_task_name']:r for r in load('hand_task_consolidation_20261119/mapping.json')['records']}
    fills={r['path']:r for r in load('missing_task_names_14_apply_report.json')['records']}
    preview=[]
    for src in inventory:
        key=src['dataset'],src['path'];old=src.get('task_name');base=baseline.get(old)
        row=dict(dataset=src['dataset'],path=src['path'],snapshot_task_name=old,current_known_task_name=old,
                 proposed=None,status='review',basis='previous_review')
        if not old and src['path'] in fills and src['dataset']=='guanglun':
            name=fills[src['path']]['task_name']
            row.update(current_known_task_name=name,proposed=name,status='already_filled',basis='missing_14_writeback')
        elif base and base['status']=='mapped':
            row.update(proposed=base['proposed'],status='candidate',basis='previous_consolidation')
        elif key in by_key:
            r=by_key[key];row.update(proposed=r.get('proposed'),status=r['status'],basis=r['basis'])
        preview.append(row)
    assert len(preview)==16259 and len({(r['dataset'],r['path']) for r in preview})==len(preview)
    write_csv(OUT/'all_sample_preview.csv',preview,list(preview[0]))
    groups=defaultdict(list);name_targets=defaultdict(set)
    for r in preview:
        if r['proposed']:
            groups[r['proposed']].append(r)
            if r['snapshot_task_name']:name_targets[r['snapshot_task_name']].add(r['proposed'])
    write_csv(OUT/'all_candidate_groups.csv',[dict(task_name=name,samples=len(rows),
              jianzhi=sum(r['dataset']=='jianzhi' for r in rows),guanglun=sum(r['dataset']=='guanglun' for r in rows),
              old_names=' | '.join(sorted({r['snapshot_task_name'] for r in rows if r['snapshot_task_name']})))
              for name,rows in sorted(groups.items())],['task_name','samples','jianzhi','guanglun','old_names'])
    write_csv(OUT/'old_names_with_multiple_targets.csv',[dict(old_task_name=n,targets=' | '.join(sorted(ts)))
              for n,ts in sorted(name_targets.items()) if len(ts)>1],['old_task_name','targets'])
    summary=dict(selected=len(source),additional_candidates=sum(r['status']=='candidate' for r in reviewed),
                 user_assigned_fallback=fallback_count,
                 semantic_candidates=sum(r['status']=='candidate' and r['basis']!='user_assigned_fallback' for r in reviewed),
                 individually_reviewed=len(manual),remaining=sum(r['status']=='review' for r in reviewed),
                 totals={ds:dict(samples=len(rr),statuses=dict(Counter(r['status'] for r in rr)),
                                candidate_types=len({r['proposed'] for r in rr if r['proposed']}))
                         for ds in ('jianzhi','guanglun','combined')
                         for rr in [[r for r in preview if ds=='combined' or r['dataset']==ds]]},
                 old_names_with_multiple_targets=sum(len(ts)>1 for ts in name_targets.values()),
                 model_called=False,annotation_modified=False)
    (OUT/'combined_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
