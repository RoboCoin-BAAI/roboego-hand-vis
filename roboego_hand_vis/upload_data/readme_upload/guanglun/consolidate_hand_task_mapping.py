"""Consolidate existing names and cached model proposals. Review-only, no API/writes."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from build_task_name_mapping import write_csv
from review_main_hand_tasks import canonical_noun, phrase_present, text_values

BASE=Path(__file__).resolve().parent
OUTPUT=BASE/'hand_task_consolidation_20261119'
# These are task-level proposed grouping rules, not claims that the physical
# motions are identical. Subtasks and old task names remain untouched.
BODY={'face','head','scalp','back','neck','shoulder','arm','leg','foot','hand','body','abdomen','skin','eye','eyebrow','hair','wig','nail','lip','ear'}
ABSTRACT={'room','bathroom','bedroom','kitchen','shop','store','workspace','work area','kitchen work area','facial care','head care','operation','treatment','service'}
STORAGE_OBJECTS=set()
# Explicit phrases, not a catch-all noun suffix heuristic.
STORAGE_OBJECTS.update('box,bottle,chair,book,cup,can,bag,item,object,package,parcel,product,cosmetics,toiletries,medicine,medicine box,medicine product,container,paper,stationery,pen,pencil,marker,eraser,folder,document,card,tag,label,shelf,rack,cabinet,drawer,table,desk,counter,nightstand,tray,basket,dish,dishware,tableware,utensil,kitchenware,clothing,bedding,pillow,cushion,towel,cloth,coaster,placemat,napkin,blanket,sheet,tissue,band,rod,stick,cable,wire,cord,lanyard,clip,hairpin,brush,comb,tool,accessory,supply,equipment,dumbbell,barbell,weight plate,fitness equipment,sports equipment,yoga mat,wristband,racket,ball,game,game component,chess piece,toy,snack,candy,bread,food,noodle,nut,drink,beverage,beer,milk,water,soda,cola,coffee ingredient,sugar,seasoning,powder,sample,fruit,herb,vegetable,soap,toothpaste,shower gel,shampoo,cream,bottle opener,cotton roll,cotton swab,wet wipe,hygiene product,bathroom supply,cleaning supply,hairdressing tool,glasses,nail polish,skin care product'.split(','))
NOUN_ALIASES={'cosmetic':'cosmetics','toiletry':'toiletries','object':'item','skin-care product':'skin care product','skincare product':'skin care product'}


def normalize_pair(verb,noun,rules):
    noun=canonical_noun(noun,rules)
    noun=re.sub(r"^(?:customer(?:'s)?|client(?:'s)?|person(?:'s)?)\s+",'',noun)
    noun=NOUN_ALIASES.get(noun,noun)
    ids=[]
    if verb=='wipe':verb='clean';ids.append('V_WIPE_CLEAN')
    if verb in ('arrange','tidy') and noun in STORAGE_OBJECTS:
        verb='organize';ids.append('V_ORDINARY_STORAGE')
    if verb=='scrub' and not any(phrase_present(b,noun) for b in BODY):
        verb='clean';ids.append('V_OBJECT_SCRUB')
    if verb in ('sweep','mop','vacuum') and noun in ('floor','ground','carpet','rug'):
        verb='clean';ids.append('V_FLOOR_CLEANING')
    if verb in ('organise','colour'):verb={'organise':'organize','colour':'color'}[verb]
    return verb,noun,ids


def valid_pair(verb,noun,verbs):
    return (verb in verbs and noun and noun not in ABSTRACT and
            not re.search(r'\b(and|or|then|with|to|into|from|for|using|operation|treatment)\b',noun)
            and bool(re.fullmatch('[a-z]+(?:[ -][a-z]+)*',noun)))


def phrase_rules(name):
    n=' '.join(name.lower().split())
    regions={'facial':'face','face':'face','head':'head','scalp':'scalp','back':'back','abdominal':'abdomen','abdomen':'abdomen','neck':'neck','shoulder':'shoulder','shoulders':'shoulder','arm':'arm','arms':'arm','leg':'leg','legs':'leg','foot':'foot','feet':'foot','hand':'hand','hands':'hand','body':'body','lower back':'lower back','waist':'waist'}
    pattern=r'(?:(?:perform|give|provide|administer)\s+)?(facial|head|scalp|back|abdominal|neck|shoulder|arm|leg|foot|hand|body) massage(?: treatment)?'
    match=re.fullmatch(pattern,n)
    if match:return 'massage',regions[match[1]],'P_MASSAGE_REGION'
    match=re.fullmatch(r'massage (?:customer(?:\x27s)? |client(?:\x27s)? )?([a-z ]+?)(?: with (?:essential oil|hands and massage hammer|massage hammer|hands))?',n)
    if match and match[1] in regions:return 'massage',regions[match[1]],'P_MASSAGE_REGION'
    match=re.fullmatch(r'inflate (?:(?:multiple|deflated|flat) )?basketballs?(?: with (?:an? )?(?:air )?pump)?',n)
    if match:return 'inflate','basketball','P_INFLATE_BALL'
    if re.fullmatch(r'file (?:customer(?:\x27s)? )?(?:finger)?nails?',n):return 'file','nail','P_FILE_NAIL'
    if re.fullmatch(r'(?:practice (?:hair|perm rod|perm) rolling|roll hair)(?: on (?:a )?(?:wig head model|wig head|mannequin|wig mannequin)| with perm rods)?',n):
        return 'roll','hair','P_HAIR_ROLLING'
    return None


def main():
    inventory=json.loads((BASE/'task_names_inventory_20261119.json').read_text(encoding='utf-8'))
    lexical={r['original']:r for r in json.loads((BASE/'task_name_mapping_20261119/mapping.json').read_text(encoding='utf-8'))['mappings']}
    audit={r['old_task_name']:r for r in json.loads((BASE/'main_hand_task_mapping_20261119/cache_audit.json').read_text(encoding='utf-8'))['records']}
    rules=json.loads((BASE/'task_name_normalization_rules.json').read_text(encoding='utf-8'))
    table=json.loads((BASE/'action_vocabulary.json').read_text(encoding='utf-8'))
    verbs={r['verb'] for group in table['groups'].values() for r in group}
    verbs.update(['clean','organize','massage','perm','braid','curl','comb','dye','shampoo','style','condition','exfoliate','moisturize','wax','vacuum','store','restock','replenish','inspect','arrange','inflate','file','tidy','scrub','mop','sweep'])
    results=[]
    for name,l in lexical.items():
        model=audit.get(name,{})
        row=dict(old_task_name=name,proposed=None,status='review',counts=l['counts'],basis='unresolved',rules=[],reason='No reliable single-operation mapping yet')
        phrase=phrase_rules(name)
        if phrase:
            verb,noun,key=phrase;verb,noun,keys=normalize_pair(verb,noun,rules)
            row.update(proposed=verb+' '+noun,status='mapped',basis='explicit_phrase_rule',rules=[key]+keys,
                       reason='Explicit service/action phrase simplified; body region and operation retained')
        elif l['status']=='ready' and l.get('verb') not in ('prepare','perform','give','provide'):
            verb,noun,keys=normalize_pair(l['verb'],l['object_after'],rules)
            if valid_pair(verb,noun,verbs) and not re.search(r'\b(?:roll|rolling|cutting|cleaning|organizing|perform|practice)\b',noun):
                row.update(proposed=verb+' '+noun,status='mapped',basis='existing_short_name',rules=l['rules']+keys,
                           reason='Keep original primary operation/object; apply explicit vocabulary rules only')
        if row['status']=='review' and model.get('status')=='candidate':
            parts=model['proposed'].split(' ',1)
            if len(parts)==2:
                verb,noun,keys=normalize_pair(*parts,rules)
                if valid_pair(verb,noun,verbs):
                    row.update(proposed=verb+' '+noun,status='mapped',basis='validated_model_candidate',rules=keys,
                               reason=model.get('decision',{}).get('reason','Existing cached model proposal'))
        # Strict field mismatches need not discard an otherwise verbatim noun:
        # salvage only exact old-name targets, not a new abstraction/destination.
        if row['status']=='review' and model.get('validation_error') in ('Unsupported evidence quote','Unsupported noun normalization'):
            d=model.get('decision',{});verb=d.get('verb');noun=d.get('object_name')
            if isinstance(noun,str) and verb in verbs and phrase_present(noun,name):
                verb,noun,keys=normalize_pair(verb,noun,rules)
                if valid_pair(verb,noun,verbs):
                    row.update(proposed=verb+' '+noun,status='mapped',basis='model_candidate_existing_noun',rules=keys,
                               reason='Cached candidate, noun is explicitly in old name; raw field inconsistency retained for review')
                    row['caution']=model['validation_error']
        if row['status']=='review':row['reason']=model.get('validation_error',model.get('error',model.get('decision',{}).get('reason',row['reason'])))
        # Every semantic/model result remains an external proposal, never auto-approved.
        row['approval']='proposed_only'
        results.append(row)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    save=lambda name,value:(OUTPUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('mapping.json',dict(annotation_modified=False,model_called=False,records=results))
    flat=[dict(old_task_name=r['old_task_name'],proposed=r['proposed'],status=r['status'],basis=r['basis'],
               jianzhi=r['counts'].get('jianzhi',0),guanglun=r['counts'].get('guanglun',0),
               rule_ids=';'.join(r['rules']),reason=r['reason'],caution=r.get('caution','')) for r in results]
    write_csv(OUTPUT/'mapping.csv',flat,list(flat[0]))
    write_csv(OUTPUT/'remaining_review.csv',[r for r in flat if r['status']=='review'],list(flat[0]))
    write_csv(OUTPUT/'caution_candidates.csv',[r for r in flat if r['caution']],list(flat[0]))
    lookup={r['old_task_name']:r for r in results}
    samples=[]
    for source in inventory['records']:
        old=source.get('task_name');mapping=lookup.get(old)
        samples.append(dict(dataset=source['dataset'],path=source['path'],old_task_name=old,
                            proposed=mapping.get('proposed') if mapping else None,
                            status=mapping['status'] if mapping else 'missing_task_name',
                            basis=mapping['basis'] if mapping else 'missing',
                            caution=mapping.get('caution','') if mapping else ''))
    write_csv(OUTPUT/'sample_preview.csv',samples,['dataset','path','old_task_name','proposed','status','basis','caution'])
    groups=defaultdict(list)
    for r in results:
        if r['status']=='mapped':groups[r['proposed']].append(r)
    write_csv(OUTPUT/'groups.csv',[dict(task_name=n,source_names=len(rs),jianzhi=sum(r['counts'].get('jianzhi',0) for r in rs),
       guanglun=sum(r['counts'].get('guanglun',0) for r in rs),old_names=' | '.join(r['old_task_name'] for r in rs))
       for n,rs in sorted(groups.items())],['task_name','source_names','jianzhi','guanglun','old_names'])
    summary={}
    for ds in ('jianzhi','guanglun','combined'):
        rr=[r for r in results if ds=='combined' or r['counts'].get(ds)]
        count=lambda r:sum(r['counts'].values()) if ds=='combined' else r['counts'].get(ds,0)
        summary[ds]=dict(original_names=len(rr),mapped_names=sum(r['status']=='mapped' for r in rr),
                        mapped_task_types=len({r['proposed'] for r in rr if r['status']=='mapped'}),
                        remaining_names=sum(r['status']=='review' for r in rr),mapped_samples=sum(count(r) for r in rr if r['status']=='mapped'),
                        retained_review_samples=sum(count(r) for r in rr if r['status']=='review'),
                        total_types_if_reviews_retained=len({r['proposed'] if r['status']=='mapped' else r['old_task_name'] for r in rr}))
    save('summary.json',summary)
    save('rules.json',dict(noun_aliases=NOUN_ALIASES,ordinary_storage_objects=sorted(STORAGE_OBJECTS),
                           verb_rules=['wipe -> clean','ordinary arrange/tidy -> organize','object scrub -> clean (exclude body targets)',
                                       'sweep/mop/vacuum floor/ground/carpet/rug -> clean'],
                           source_precedence=['explicit phrase','existing short name','validated model','candidate with existing noun'],
                           approval='All semantic changes are external proposals; no annotation writeback'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
