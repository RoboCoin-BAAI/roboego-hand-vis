"""Build deterministic task-name mapping previews. Never writes annotations."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

BASE=Path(__file__).resolve().parent


def normalize(name,rules,verbs):
    formatted=' '.join(name.lower().split())
    out=dict(original=name,formatted=formatted,object_stage=formatted,proposed=formatted,
             status='ready',rules=[],verb=None,object_before=None,object_after=None,candidates=[])
    if formatted!=name:out['rules'].append('format')
    # Only known base verbs and spelling aliases are parsed. Inflected or long
    # legacy descriptions are review items, never silently rewritten as goals.
    parts=formatted.split(' ',1)
    if len(parts)!=2 or parts[0] not in verbs or not parts[1]:
        out['status']='review_unparsed';return out
    verb,noun=parts
    if re.search(r'\b(and|or|then|with|to|in|on|into|from|for|using|after|before|by|of|onto|around|through|over|under)\b|[;,/&+]',noun):
        out['status']='review_compound';return out
    out.update(verb=verb,object_before=noun)
    noun_after=rules['object_aliases'].get(noun,noun)
    if noun_after!=noun:out['rules'].append('object_alias')
    if noun_after not in rules['preserve_nouns']:
        words=noun_after.split()
        tail=rules['plural_heads'].get(words[-1],words[-1])
        if tail!=words[-1]:
            words[-1]=tail;noun_after=' '.join(words);out['rules'].append('object_plural')
    out['object_after']=noun_after
    out['object_stage']=verb+' '+noun_after
    mapped=rules['verb_aliases'].get(verb,{})
    final_verb=mapped.get('target',verb)
    if final_verb!=verb:out['rules'].append('verb_alias:'+verb+'->'+final_verb)
    out['proposed']=final_verb+' '+noun_after
    out['candidates']=[c for c in rules['candidate_verb_merges'] if final_verb in c['sources']]
    return out


def write_csv(path,rows,columns):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for row in rows:w.writerow({k:row.get(k,'') for k in columns})


def build(inventory,rules,verb_table,output):
    verbs={r['verb'] for rows in verb_table['groups'].values() for r in rows}
    verbs.update(rules['verb_aliases'])
    for c in rules['candidate_verb_merges']:verbs.update(c['sources'])
    # Existing service verbs may legitimately be outside the old reference.
    verbs.update(['massage','perm','braid','straighten','curl','comb','style','dye','shampoo',
                  'condition','exfoliate','moisturize','wax','depilate','manicure','pedicure',
                  'vacuum','restock','replenish','store','reassemble','staple','coil','wind'])
    grouped=defaultdict(list)
    for r in inventory['records']:
        n=r.get('task_name')
        if isinstance(n,str) and n.strip():grouped[n].append(r)
    mappings=[]
    for name in sorted(grouped):
        r=normalize(name,rules,verbs);r['counts']=dict(Counter(v['dataset'] for v in grouped[name]))
        r['examples']=[dict(dataset=v['dataset'],path=v['path'],task_description=v.get('task_description')) for v in grouped[name][:2]]
        mappings.append(r)
    output.mkdir(parents=True,exist_ok=True)
    (output/'mapping.json').write_text(json.dumps(dict(rule_version=rules['version'],annotation_modified=False,
                                                      mappings=mappings),ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[dict(original=r['original'],object_stage=r['object_stage'],proposed=r['proposed'],status=r['status'],
               rules=';'.join(r['rules']),candidate_merges=';'.join(c['target'] for c in r['candidates']),
               jianzhi=r['counts'].get('jianzhi',0),guanglun=r['counts'].get('guanglun',0)) for r in mappings]
    write_csv(output/'task_name_mapping.csv',flat,['original','object_stage','proposed','status','rules','candidate_merges','jianzhi','guanglun'])
    write_csv(output/'all_sample_task_names.csv',inventory['records'],
              ['dataset','path','task_name','common_task_name','task_description'])
    write_csv(output/'missing_task_names.csv',[r for r in inventory['records']
              if not isinstance(r.get('task_name'),str) or not r['task_name'].strip()],['dataset','path','task_name','task_description'])
    write_csv(output/'review_tasks.csv',[r for r in flat if r['status']!='ready'],list(flat[0]) if flat else ['original'])
    nouns=defaultdict(Counter)
    for r in mappings:
        if r['object_before']:nouns[(r['object_before'],r['object_after'])].update(r['counts'])
    write_csv(output/'object_mapping.csv',[dict(original=a,normalized=b,jianzhi=c['jianzhi'],guanglun=c['guanglun'])
                                         for (a,b),c in sorted(nouns.items())],['original','normalized','jianzhi','guanglun'])
    summary={}
    for ds in ('jianzhi','guanglun','combined'):
        rs=[r for r in mappings if ds=='combined' or r['counts'].get(ds)]
        count=lambda r:sum(r['counts'].values()) if ds=='combined' else r['counts'].get(ds,0)
        summary[ds]=dict(samples=sum(count(r) for r in rs),original_unique=len(rs),
                        after_object_unique=len({r['object_stage'] for r in rs}),
                        after_verb_unique=len({r['proposed'] for r in rs}),
                        changed_samples=sum(count(r) for r in rs if r['original']!=r['proposed']),
                        review_unique=sum(r['status']!='ready' for r in rs))
    (output/'summary.json').write_text(json.dumps(dict(counts=summary,collection_errors=inventory['errors'],
       missing_task_names={ds:v['missing'] for ds,v in inventory['summary'].items()}),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return mappings,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rules',type=Path,default=BASE/'task_name_normalization_rules.json')
    p.add_argument('--vocabulary',type=Path,default=BASE/'action_vocabulary.json')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    build(load(a.inventory),load(a.rules),load(a.vocabulary),a.output)


if __name__=='__main__':main()
