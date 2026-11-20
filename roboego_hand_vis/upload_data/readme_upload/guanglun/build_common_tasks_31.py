"""Build 31-class common_task_name previews; no annotation writes or model calls."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import re

from build_task_name_mapping import write_csv

BASE=Path(__file__).resolve().parent
OUT=BASE/'common_tasks_31_20261119'


def snake(value):return re.sub('[^a-z0-9]+','_',value.lower()).strip('_')


def classify(noun,verb,context,taxonomy):
    cats=taxonomy['categories'];ctx=context.lower()
    has=lambda pattern:bool(re.search(pattern,ctx))
    if verb=='massage':return 'body','explicit_user_massage_group'
    if noun in taxonomy.get('standalone_categories',{}) and noun not in ('mask','glass','liquid'):
        return noun,'user_retained_independent_category'
    if noun=='liquid':
        if has(r'\b(nail tips?|nail pieces?|nail polish|manicure|uv lamp)\b'):return 'nail_polish','context_nail_coating'
        if has(r'\b(hand soap|soap dispenser)\b'):return 'toiletries','context_soap_refill'
        if has(r'\b(soup|broth)\b'):return 'food','context_soup'
        if has(r'\b(preparing drinks|serving drinks|beverage preparation|preparing and serving beverages|serving beverages|drinks into cups)\b'):
            return 'beverage','context_beverage_service'
    # Test narrow material meanings before general categories.
    if noun in ('dye','pigment') and has(r'\b(hair|wig|barber|hairdresser)\b'):
        return 'hair_dye','context_hair_coloring'
    if noun in ('polish','gel','paint') and has(r'\b(nails?|manicure|fingernails?|toenails?|nail polish|nail tips?|nail pieces?)\b'):
        return 'nail_polish','context_nail_coloring'
    if noun in ('mask','gel','mud','product','liquid','solution','mixture') and verb=='apply' and has(r'\b(facial|face mask|skin care|skincare|essence|makeup|cosmetic|cosmetics)\b'):
        return 'cosmetics','context_facial_skincare'
    if noun in ('brush','clip','rod','roller','rubber band','rubber bands','model','dummy','hairdressing_tools'):
        if has(r'\b(hair|wig|barber|hairdresser|perm|manicure|makeup|nail polish|eyelash)\b'):
            return 'beauty_tool','context_beauty_tool'
        if noun=='brush' and has(r'\b(scrub|toilet|cleaning brush|cleaning tool|cleaning tools|floor)\b'):
            return 'cleaning_tool','context_cleaning_brush'
        if noun in ('rod','clip','rubber band','rubber bands') and has(r'\b(screw|bolt|metal|hardware|wood|assemble|assembly|mechanical)\b'):
            return 'component','context_hardware'
        if noun=='brush' and has(r'\b(paint|painting|drawing|art)\b'):return 'stationery','context_art_brush'
        if noun in ('clip','rubber band','rubber bands') and has(r'\b(paper|papers|document|documents|stationery)\b'):
            return 'stationery','context_paper_fastener'
    if noun in ('mask','pad'):
        if noun=='mask' and verb=='clean' and has(r'\b(mask box|mask boxes|boxed masks|bagged mask)\b'):
            return 'container','context_clean_mask_packaging'
        if has(r'\b(facial masks?|mask sheets?|mask paste|facial care|skin care|skincare)\b'):return 'cosmetics','context_facial_mask'
        if has(r'\b(surgical|medical mask|disposable mask|sanitary|cotton pad)\b'):return 'toiletries','context_personal_hygiene'
    if noun in ('glue','gel','powder','paste','solution','liquid','foam','substance'):
        if has(r'\b(eyelash glue|nail glue|facial|skin care|skincare|cosmetic|cosmetics|makeup)\b'):
            return 'cosmetics','context_beauty_material'
        if has(r'\b(detergent|cleaning solution|disinfectant|laundry|dishwashing)\b'):
            return 'cleaning_agent','context_cleaning_material'
        if has(r'\b(juice|milk tea|coffee|tea|drink|beverage)\b') and noun in ('liquid','foam'):
            return 'beverage','context_beverage_material'
        if has(r'\b(flour|rice|sugar|ingredient|ingredients|cook|cooking|food|dough|dessert)\b'):
            return 'food','context_food_material'
    if noun=='glass':
        if has(r'\b(glass (?:display |area of the )?(?:shelf|shelves|counter|cabinet|display)|display cabinet|display case)\b'):
            return 'furniture','context_glass_furniture_surface'
        if has(r'\b(window|glass door|glass panel|pane)\b') and not has(r'\b(cup|cups|drink|drinking|wine)\b'):
            return 'glass','retained_architectural_glass'
        if has(r'\b(glass containers?)\b'):return 'container','context_glass_container'
        if has(r'\b(cup|cups|drink|drinks|drinking|wine|tea|coffee|tableware|bar counter|glasses|glassware|dish rack|drying rack|draining rack)\b'):
            return 'tableware','context_drinking_glass'
        return 'glass','retained_ambiguous_glass'
    if noun in ('nut','tablet','ball','block','band','mat'):
        if noun=='nut' and has(r'\b(bolt|screw|hardware|wrench|fasten)\b'):return 'component','context_fastener_nut'
        if noun=='tablet' and has(r'\b(ipad|screen|electronic|computer)\b'):return 'electronic_device','context_electronic_tablet'
        if noun=='ball' and has(r'\b(basketball|football|tennis|badminton|sport|sports|gym)\b'):return 'sports_equipment','context_sports_ball'
        if noun=='block' and has(r'\b(toy|toys|building blocks|children|game|lettered)\b'):return 'toy','context_play_block'
        if noun in ('band','mat') and has(r'\b(yoga|fitness|exercise|sport|sports|gym)\b'):return 'sports_equipment','context_fitness_item'
    # Full phrases take precedence: e.g. nail drill, hair dye, cup lid.
    exact=[key for key,c in cats.items() if noun in c.get('exact',[])]
    if len(exact)==1:return exact[0],'exact_noun'
    if len(exact)>1:return None,'overlapping_exact_rules'
    candidates=[(len(head),key) for key,c in cats.items() for head in c.get('heads',[])
                if noun==head or noun.endswith(' '+head)]
    if candidates:
        longest=max(n for n,k in candidates);found={k for n,k in candidates if n==longest}
        if len(found)==1:return found.pop(),'specific_noun_head'
        return None,'ambiguous_noun_head'
    # Specific functional categories win above. Only remaining generic labels
    # lose shape/location qualifiers at the common-task level.
    if re.search(r'\b(items?|objects?|products?|goods|supplies|supply|materials?|articles?|stuff|things?)$',noun):
        return 'item','generic_noun_family_item'
    if noun in ('mask','liquid') and noun in taxonomy.get('standalone_categories',{}):
        return noun,'retained_ambiguous_'+noun
    return None,'catalog_gap_or_ambiguous_noun'


def main():
    taxonomy=json.loads((BASE/'OBJECT_CATEGORIES_31.json').read_text(encoding='utf-8'))
    assert len(taxonomy['categories'])==31
    samples=list(csv.DictReader((BASE/'remaining_task_review_20261119/all_sample_preview.csv').open(encoding='utf-8-sig')))
    inventory=json.loads((BASE/'task_names_inventory_20261119.json').read_text(encoding='utf-8'))['records']
    evidence={(r['dataset'],r['path']):r for r in inventory}
    extra_path=OUT/'ambiguous_context.json'
    extra={(r['dataset'],r['path']):r for r in json.loads(extra_path.read_text(encoding='utf-8'))} if extra_path.exists() else {}
    all_categories={**taxonomy['categories'],**taxonomy.get('standalone_categories',{})}
    rows=[]
    for sample in samples:
        fine=sample['proposed'];source=evidence[(sample['dataset'],sample['path'])]
        ctx=(source.get('task_description') or '')+'\n'+'\n'.join(str(n) for n in source.get('objects',[]) if n)
        detail=extra.get((sample['dataset'],sample['path']))
        if detail:
            ctx+='\n'+str(detail.get('scene') or '')+'\n'+'\n'.join(detail.get('subtasks') or [])
        if fine==taxonomy['user_fallback_label']:
            noun='some object';category='item';rule='user_fallback_generic_common';common=taxonomy.get('user_fallback_common_name',fine)
        else:
            verb,noun=fine.split(' ',1)
            category,rule=classify(noun,verb,ctx,taxonomy)
            common=snake(verb+' '+(category or noun))
        retained_fine=category is None
        if retained_fine:
            category=snake(noun)
            all_categories.setdefault(category,dict(zh=noun+'（保留细类）'))
            rule='retained_original_type:'+rule
        assert re.fullmatch('[a-z][a-z0-9_]*',common)
        rows.append(dict(dataset=sample['dataset'],path=sample['path'],task_name=fine,object_noun=noun,
                         object_category=category or '',category_zh=all_categories[category]['zh'] if category else '',
                         category_kind='retained_fine' if retained_fine else 'main_31' if category in taxonomy['categories'] else 'retained_independent',
                         common_task_name=common,status='retained_fine' if retained_fine else 'mapped',rule=rule))
    assert len(rows)==16259 and len({(r['dataset'],r['path']) for r in rows})==16259
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/'sample_mapping.csv',rows,list(rows[0]))
    (OUT/'mapping.json').write_text(json.dumps(dict(version=1,annotation_modified=False,model_called=False,records=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    by_noun=defaultdict(list);by_common=defaultdict(list)
    for r in rows:by_noun[(r['object_noun'],r['object_category'],r['rule'])].append(r);by_common[r['common_task_name']].append(r)
    nouns=[dict(noun=n,category=c,category_zh=all_categories[c]['zh'] if c else '',rule=rule,
                jianzhi=sum(r['dataset']=='jianzhi' for r in rr),guanglun=sum(r['dataset']=='guanglun' for r in rr))
           for (n,c,rule),rr in sorted(by_noun.items())]
    write_csv(OUT/'noun_categories.csv',nouns,list(nouns[0]))
    write_csv(OUT/'unmapped_nouns.csv',[r for r in nouns if r['rule'].startswith('retained_original_type:')],list(nouns[0]))
    groups=[dict(common_task_name=n,jianzhi=sum(r['dataset']=='jianzhi' for r in rr),guanglun=sum(r['dataset']=='guanglun' for r in rr),
                 fine_task_names=' | '.join(sorted({r['task_name'] for r in rr})),
                 has_unmapped_noun=any(r['status']=='retained_fine' for r in rr)) for n,rr in sorted(by_common.items())]
    write_csv(OUT/'common_task_groups.csv',groups,list(groups[0]))
    counts=[dict(category=k,zh=c['zh'],jianzhi=sum(r['dataset']=='jianzhi' and r['object_category']==k for r in rows),
                 guanglun=sum(r['dataset']=='guanglun' and r['object_category']==k for r in rows)) for k,c in taxonomy['categories'].items()]
    write_csv(OUT/'category_counts.csv',counts,['category','zh','jianzhi','guanglun'])
    standalone=[dict(category=k,zh=c['zh'],jianzhi=sum(r['dataset']=='jianzhi' and r['object_category']==k for r in rows),
                    guanglun=sum(r['dataset']=='guanglun' and r['object_category']==k for r in rows))
                for k,c in taxonomy.get('standalone_categories',{}).items()]
    write_csv(OUT/'retained_category_counts.csv',standalone,['category','zh','jianzhi','guanglun'])
    summary={ds:dict(samples=len(rr),mapped=sum(r['status']=='mapped' for r in rr),needs_category=sum(r['status']=='needs_category' for r in rr),
                    fine_tasks=len({r['task_name'] for r in rr}),common_tasks_preview=len({r['common_task_name'] for r in rr}),
                    mapped_common_tasks=len({r['common_task_name'] for r in rr if r['status']=='mapped'}),
                    main_31_samples=sum(r['category_kind']=='main_31' for r in rr),
                    retained_independent_samples=sum(r['category_kind']=='retained_independent' for r in rr),
                    retained_fine_samples=sum(r['category_kind']=='retained_fine' for r in rr))
             for ds in ('jianzhi','guanglun','combined') for rr in [[r for r in rows if ds=='combined' or r['dataset']==ds]]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
