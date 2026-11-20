"""Read-only structural inventory of Guanglun batches 1-49 after publication move."""
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from reclassify_scenes_qwen import save_json

BASE=Path('/share/project/liuyou/humancoin/data')
REPORT=BASE/'guanglun_publish_dataset_move_01_49_20261119'

def inspect(r):
    session=Path(r['target'])
    readme=json.loads((session/'meta/readme.json').read_text(encoding='utf-8-sig'))
    files=[]
    for directory,_,names in os.walk(session):
        for name in names:
            p=Path(directory)/name;relative=p.relative_to(session).as_posix()
            if relative=='meta/readme.json' or p.parent==session/'meta' and name.startswith('.readme.json.'):continue
            files.append(relative)
    return dict(path=str(session),readme_count=readme['file_count'],files=tuple(sorted(files)))

def main():
    move=json.loads((REPORT/'summary.json').read_text())
    if move['errors'] or move['moved']!=4461:raise ValueError('Move not complete')
    plan=json.loads((REPORT/'plan.json').read_text())['records'];results=[]
    with ThreadPoolExecutor(16) as pool:
        for future in as_completed([pool.submit(inspect,r) for r in plan]):
            results.append(future.result())
            if len(results)%1000==0:print(json.dumps(dict(phase='inventory',completed=len(results),total=len(plan))),flush=True)
    patterns=Counter(r['files'] for r in results);baseline=patterns.most_common(1)[0][0]
    common=set(baseline)
    for r in results:common.intersection_update(r['files'])
    variants=[dict(samples=n,file_count=len(files),extra=sorted(set(files)-set(baseline)),missing=sorted(set(baseline)-set(files)),files=list(files)) for files,n in patterns.most_common()]
    auxiliary=Counter()
    for r in results:
        for file in r['files']:
            if file.endswith('.lock') or Path(file).name in ['alignment_check.jsonl','annotations.json','frame_mapping.parquet']:
                auxiliary[file]+=1
    summary=dict(samples=len(results),dataset_groups=move['groups'],file_count_excludes_readme=True,
                 file_count_distribution=dict(sorted(Counter(len(r['files']) for r in results).items())),
                 readme_count_mismatches=sum(r['readme_count']!=len(r['files']) for r in results),
                 structure_variants=len(variants),common_files=len(common),auxiliary_files=dict(auxiliary),
                 variants=[{k:v for k,v in r.items() if k!='files'} for r in variants])
    save_json(REPORT/'file_structure_audit.json',dict(summary=summary,common_files=sorted(common),variants=variants))
    with (REPORT/'file_structure_audit.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_path','readme_file_count','actual_file_count','extra_vs_most_common','missing_vs_most_common'])
        for r in sorted(results,key=lambda r:r['path']):
            writer.writerow([r['path'],r['readme_count'],len(r['files']),' | '.join(sorted(set(r['files'])-set(baseline))),' | '.join(sorted(set(baseline)-set(r['files'])))])
    save_json(REPORT/'file_count_summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
