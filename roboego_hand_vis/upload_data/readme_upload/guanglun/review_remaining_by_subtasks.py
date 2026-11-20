"""Evidence-based second review: full subtask text, explicit targets, timing.

No API calls, no annotation writes. Time assigned within compound segments is an
estimate; all evidence and unresolved ties are exported, not hidden.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from build_task_name_mapping import write_csv
from consolidate_hand_task_mapping import normalize_pair, ABSTRACT
from review_main_hand_tasks import phrase_present, canonical_noun

BASE=Path(__file__).resolve().parent
OUT=BASE/'remaining_task_review_20261119'
ALIASES={
 'wipe':'clean','wiping':'clean','clean':'clean','cleaning':'clean','sweep':'clean','sweeping':'clean',
 'mop':'clean','mopping':'clean','vacuum':'clean','vacuuming':'clean','scrub':'scrub','scrubbing':'scrub',
 'wash':'wash','washing':'wash','rinse':'rinse','rinsing':'rinse','dry':'dry','drying':'dry','blow dry':'dry','blow-dry':'dry',
 'massage':'massage','massaging':'massage','knead':'massage','kneading':'massage','pound':'massage','pounding':'massage',
 'tap':'tap','tapping':'tap','press':'press','pressing':'press','roll':'roll','rolling':'roll',
 'cut':'cut','cutting':'cut','trim':'trim','trimming':'trim','shave':'shave','shaving':'shave',
 'comb':'comb','combing':'comb','brush':'brush','brushing':'brush','braid':'braid','braiding':'braid',
 'straighten':'straighten','straightening':'straighten','dye':'dye','dyeing':'dye','perm':'perm','perming':'perm',
 'apply':'apply','applying':'apply','spray':'spray','spraying':'spray','paint':'paint','painting':'paint',
 'scrape':'scrape','scraping':'scrape','file':'file','filing':'file','polish':'polish','polishing':'polish',
 'fold':'fold','folding':'fold','unfold':'unfold','unfolding':'unfold','iron':'iron','ironing':'iron',
 'organize':'organize','organizing':'organize','arrange':'organize','arranging':'organize','tidy':'organize',
 'sort':'sort','sorting':'sort','stack':'stack','stacking':'stack','align':'align',
 'pack':'pack','packing':'pack','package':'pack','packaging':'pack','seal':'seal','sealing':'seal',
 'wrap':'wrap','wrapping':'wrap','tie':'tie','tying':'tie','bind':'bind','binding':'bind',
 'fill':'fill','filling':'fill','refill':'fill','refilling':'fill','pour':'pour','pouring':'pour',
 'stir':'stir','stirring':'stir','mix':'mix','mixing':'mix','whip':'whip','whipping':'whip',
 'peel':'peel','peeling':'peel','slice':'slice','slicing':'slice','dice':'dice','dicing':'dice',
 'chop':'chop','chopping':'chop','grind':'grind','grinding':'grind','crush':'crush','crushing':'crush',
 'scoop':'scoop','scooping':'scoop','heat':'heat','heating':'heat','bake':'bake','baking':'bake',
 'assemble':'assemble','assembling':'assemble','install':'install','installing':'install',
 'attach':'attach','attaching':'attach','replace':'replace','replacing':'replace','repair':'repair',
 'drill':'drill','drilling':'drill','screw':'screw','tighten':'tighten','inflate':'inflate',
 'write':'write','writing':'write','type':'type','typing':'type','enter':'enter','input':'enter',
 'collect':'collect','collecting':'collect','gather':'collect','gathering':'collect','hang':'hang','hanging':'hang',
 'pick up':'pick','pick':'pick','take out':'take','take':'take','put back':'place','put':'place','place':'place',
 'store':'store','return':'return','throw':'discard','discard':'discard','move':'move','carry':'carry',
 'open':'open','close':'close','cover':'cover','remove':'remove','adjust':'adjust','retrieve':'retrieve',
 'water':'water','watering':'water','weigh':'weigh','weighing':'weigh','measure':'measure',
 'connect':'connect','disconnect':'disconnect','insert':'insert','pull':'pull','push':'push',
 'serve':'serve','deliver':'deliver','spread':'spread','lay':'lay','flatten':'flatten',
 'operate':'operate','check':'inspect','inspect':'inspect','select':'select','browse':'browse',
 'disinfect':'disinfect','extract':'extract','pluck':'pluck','tattoo':'tattoo','pump':'pump'
}
ACTION=re.compile(r'(?<![a-z])('+'|'.join(re.escape(s) for s in sorted(ALIASES,key=len,reverse=True))+r')(?![a-z])')
AUX={'pick','take','place','store','return','discard','move','carry','open','close','cover','retrieve','adjust','serve','deliver','pull','push'}
BODY={'back','head','face','scalp','neck','shoulder','arm','leg','hand','foot','abdomen','waist','chest','buttock','calf','forehead','eye'}
BODY_ALIASES={'cheek':'face','chin':'face','jaw':'face','jawline':'face','forehead':'face','upper back':'back','lower back':'back','upper arm':'arm','forearm':'arm','calf':'leg','thigh':'leg'}
VETTED_NOUNS=set(BODY)|set(BODY_ALIASES)|{'hair','nail','keyboard','coffee','water','tea','curtain','paper','food'}
LEADING=re.compile(r"^(?:(?:the|a|an|some|both|all|remaining|scattered|first|second|third|new|old|customer's|customer|client's|client|person's|person|red|blue|green|yellow|black|white|pink|brown|silver|orange|beige|golden|two|three|four|five|multiple|several|small|large|wooden|plastic|metal|beige|light|dark|[0-9]+)\s+)+")


def targets(text,objects,rules):
    text=text.strip(' ,.:;')
    text=re.split(r'\b(with|using|from|into|onto|on|at|to|in|for|by|until|toward|near|after|before)\b',text,maxsplit=1)[0].strip()
    parts=re.split(r'\s+and\s+|,',text)
    result=[]
    for part in parts:
        part=LEADING.sub('',part.strip(' .,:;'))
        part=re.sub(r'\s+(?:and|then|to|of)$','',part).strip()
        part=re.sub(r'\b(?:neatly|repeatedly|gently|carefully|flat|up|down|horizontally|vertically|evenly|tightly|together|again)\b\s*$','',part).strip()
        # Prefer an attested physical noun over an unbounded descriptive clause.
        def attested(noun):
            forms={noun}
            for plural,singular in rules['plural_heads'].items():
                if noun==singular or noun.endswith(' '+singular):forms.add(noun[:-len(singular)]+plural)
            return any(phrase_present(form,part) for form in forms)
        found=[n for n in objects if n and attested(n) and n not in ('customer','person','client')]
        if found:
            found.sort(key=lambda n:-len(n));chosen=[]
            for noun in found:
                if not any(phrase_present(noun,n) for n in chosen):chosen.append(noun)
        elif re.fullmatch(r'[a-z]+(?:[ -][a-z]+){0,4}',part) and canonical_noun(part,rules) in VETTED_NOUNS:chosen=[part]
        else:chosen=[]
        for noun in chosen:
            noun=canonical_noun(noun,rules)
            if noun and not set(noun.split()) & {'it','them','this','that','the','a','an','and','of','customer','person','client','up','down'} and noun not in ABSTRACT:
                result.append(noun)
    return list(dict.fromkeys(result))


def parse_segment(segment,objects,rules,context):
    text=segment.get('text') or segment.get('description') or ''
    if not isinstance(text,str):return []
    lower=text.lower()
    # A verb spelling inside a noun (nail polish, baking tray, water bottle)
    # must not split the clause or invent an operation.
    matches=[m for m in ACTION.finditer(lower) if m.start()==0 or
             re.search(r'(?:[,;:]|\b(?:and|then|to|while|before|after|by|continue|gently|carefully|repeatedly|slowly|again))\s*$',lower[:m.start()])]
    events=[];previous_nouns=[]
    for i,m in enumerate(matches):
        # Avoid reading a tool noun such as "spray bottle" as a verb.
        if i and lower[m.start()-1:m.start()]=='-' :continue
        verb=ALIASES[m[1]]
        stop=matches[i+1].start() if i+1<len(matches) else len(lower)
        tail=lower[m.end():stop].strip(' ,.;:')
        if verb=='spray' and tail.startswith(('bottle','bottles')):continue
        if verb=='massage' and tail.startswith(('bed','hammer')):continue
        if verb=='write':tail=re.sub(r'^on\s+','',tail)
        nouns=targets(tail,objects,rules)
        if not nouns and re.match(r'^(?:it|them)\b',tail):nouns=list(previous_nouns)
        # "Wipe stains on the table" operates on a table surface, not on the
        # cleaning tool. Only this explicit locative construction is resolved.
        if verb=='clean' and re.match(r'^(?:the )?(?:stains?|dust|dirt|smudges?)\b',tail):
            locative=re.search(r'\b(?:on|off|from)\s+(.+)',tail)
            if locative:nouns=targets(locative[1],objects,rules)
        if not nouns:continue
        previous_nouns=nouns if not re.search(r'\b(with|using)\b',tail) else []
        for noun in nouns:
            v,n,ids=normalize_pair(verb,noun,rules)
            if v=='massage':n=BODY_ALIASES.get(n,n)
            if v in ('roll','tap','press') and any(phrase_present(b,n) for b in BODY) and re.search(r'\b(massage|knead|scraping|therapy)\b',context):
                v='massage';ids.append('BODY_MASSAGE_METHOD')
            if v=='scrub' and not any(phrase_present(b,n) for b in BODY):v='clean'
            events.append(dict(verb=v,noun=n,auxiliary=verb in AUX,rule_ids=ids,text=text))
    active=[e for e in events if not e['auxiliary']]
    if active:return active
    # Repeated placement/storage may itself be the work, but never infer the
    # destination as the object. Descriptions must explicitly support storage.
    storage=bool(re.search(r'\b(organiz\w*|arrang\w*|stor\w*|stock\w*|shelv\w*|shelf)\b',context))
    for e in events:
        if storage and e['verb'] in ('place','store'):e['verb']='organize'
    return [e for e in events if e['verb'] not in ('pick','take','return','open','close','cover','retrieve','pull','push')]


def review(sample,rules):
    objects=[o.get('name','').lower() for o in sample.get('objects',[]) if isinstance(o,dict) and isinstance(o.get('name'),str)]
    context=(sample['old_task_name']+' '+(sample.get('task_description') or '')).lower()
    scores=Counter();segments=defaultdict(list);total=0;invalid=0;intervals=[]
    for sub in sample.get('subtasks',[]):
        start,end=sub.get('start_s'),sub.get('end_s')
        if not isinstance(start,(float,int)) or isinstance(start,bool) or not isinstance(end,(float,int)) or isinstance(end,bool) or end<=start:
            invalid+=1;continue
        duration=end-start;total+=duration;intervals.append((start,end))
        events=parse_segment(sub,objects,rules,context)
        pairs=list(dict.fromkeys((e['verb'],e['noun']) for e in events))
        if not pairs:continue
        # Equal allocation is conservative bookkeeping, NOT measured action time.
        for pair in pairs:
            scores[pair]+=duration/len(pairs)
            segments[pair].append(dict(start_s=start,end_s=end,allocated_s=round(duration/len(pairs),3),text=sub.get('text')))
    ranked=scores.most_common();base={k:sample[k] for k in ('dataset','path','old_task_name','sha256')}
    base.update(proposed=None,status='review',basis='full_subtask_timing',total_segment_s=round(total,3),
                candidates=[dict(task_name=v+' '+n,allocated_s=round(s,3)) for (v,n),s in ranked[:5]])
    intervals.sort()
    overlap=any(intervals[i][0]<intervals[i-1][1]-0.01 for i in range(1,len(intervals)))
    if invalid or overlap:
        base['reason']='Invalid or overlapping subtask intervals; cannot compare time reliably';return base
    if not ranked or not total:
        base['reason']='No supported physical action-object evidence extracted';return base
    (verb,noun),score=ranked[0];second=ranked[1][1] if len(ranked)>1 else 0
    fraction=score/total;ratio=score/second if second else 999
    base.update(top_fraction=round(fraction,4),top_to_second=round(ratio,3),evidence=segments[(verb,noun)])
    if noun in ABSTRACT or noun in ('thing','area','surface','customer','person','treatment','operation'):
        base['reason']='Main target remains too broad or non-object';return base
    if fraction<0.35 or ratio<1.6 or score<8:
        body_pairs=[(p,s) for p,s in ranked if p[0]=='massage' and p[1] in BODY]
        body_score=sum(s for p,s in body_pairs)
        other=max((s for p,s in ranked if p[0]!='massage' or p[1] not in BODY),default=0)
        if len(body_pairs)>=2 and body_score/total>=0.6 and (not other or body_score/other>=2):
            base.update(status='candidate',proposed='massage body',basis='same_body_multiple_regions',
                        reason='Same hand operation on multiple regions of one body; retain region details in subtasks',
                        evidence=[e for p,s in body_pairs for e in segments[p]])
            return base
        base['reason']='No clearly dominant operation-object pair under conservative timing thresholds';return base
    if len(noun.split())>4 or re.search(r'\b(and|then|with|to|of|it|them|the|against|toward|from|into)\b',noun):
        base['reason']='Target phrase may contain an action or tool-name ambiguity';return base
    if verb=='clean' and re.search(r'\b(scrap|scraps|trash|waste|debris|clipping|clippings)\b',noun):
        base['reason']='Waste removal versus surface cleaning needs semantic disambiguation';return base
    if verb=='clean' and (noun in ('garbage','coffee','rice','dust') or
                         noun in ('hair','paper') and re.search(r'\b(clippings?|scraps?|shredded|barbershop)\b',context)):
        base['reason']='Do not interpret collected food/waste as an object being washed';return base
    if verb in ('browse','inspect') or noun in ('order','instructions','information','record','data'):
        base['reason']='Visual/cognitive action is not established as a primary hand manipulation';return base
    if verb=='operate' and noun in ('eyebrow','face','skin'):
        base['reason']='Operate is not a valid physical treatment label for this target';return base
    if verb in ('place','carry','move','organize') and re.search(r'\b(curtain)\b',noun):
        base['reason']='Curtain opening/closing cannot be inferred as ordinary storage';return base
    base.update(status='candidate',proposed=verb+' '+noun,reason='Explicit subtask target dominates allocated time; compare evidence before writeback')
    return base


def main():
    data=json.loads((BASE/'remaining_task_evidence_1058_20261119.json').read_text(encoding='utf-8'))
    rules=json.loads((BASE/'task_name_normalization_rules.json').read_text(encoding='utf-8'))
    vocabulary=json.loads((BASE/'object_vocabulary.json').read_text(encoding='utf-8'))
    VETTED_NOUNS.update(canonical_noun(r['name'],rules) for r in vocabulary['entries'])
    VETTED_NOUNS.update(canonical_noun(o['name'].lower(),rules) for row in data['records'] for o in row['objects']
                        if isinstance(o,dict) and isinstance(o.get('name'),str))
    rows=[review(r,rules) for r in data['records']]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'review.json').write_text(json.dumps(dict(model_called=False,annotation_modified=False,records=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[dict(dataset=r['dataset'],path=r['path'],old_task_name=r['old_task_name'],proposed=r['proposed'],status=r['status'],
               top_fraction=r.get('top_fraction'),top_to_second=r.get('top_to_second'),reason=r['reason']) for r in rows]
    write_csv(OUT/'sample_candidates.csv',flat,list(flat[0]))
    summary=dict(selected=len(rows),statuses=dict(Counter(r['status'] for r in rows)),collection_errors=len(data['errors']),
                 allocation='Equal allocation within compound segments; estimates, not measured action durations')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))


if __name__=='__main__':main()
