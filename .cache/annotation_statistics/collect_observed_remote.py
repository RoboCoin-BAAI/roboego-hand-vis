import json,math,time
from pathlib import Path
from collections import Counter,defaultdict
aliases={'trash bin':'trash can','coffee cup':'cup','paper cup':'cup','glass cup':'cup','water glass':'cup','plastic cup':'cup','water bottle':'bottle','plastic bottle':'bottle','clear bottle':'bottle','data cable':'cable','charging cable':'cable','usb cable':'cable','clear plastic storage box':'storage box','plastic storage box':'storage box'}
result={'object_aliases':aliases,'datasets':{}}
for ds in ['guanglun_depth','L3_lerobot']:
 root=Path('/share/project/liuyou/humancoin/data')/ds
 actions=Counter();objects=Counter();scenes=Counter();attributes=defaultdict(lambda:defaultdict(Counter));errors=[];files=0;seconds=0
 for batch in sorted(root.glob('batch_*')):
  if not batch.is_dir():continue
  for session in batch.iterdir():
   if not session.is_dir():continue
   path=session/'annotation/episode_000000_structured.json'
   if not path.is_file():continue
   try:
    ann=json.loads(path.read_text());info=json.loads((session/'meta/info.json').read_text())
    duration=float(info['total_frames'])/float(info['fps'])
    if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid metadata duration')
    scene=ann.get('scene_classification') or {};category=scene.get('category') or 'unknown';area=scene.get('area') or 'unknown'
    scenes[(category,area)]+=duration
    actions.update(set(x for x in (ann.get('actions') or {}).get('vocabulary',[]) if isinstance(x,str) and x.strip()))
    for obj in ann.get('objects',[]):
     name=obj.get('name')
     if not isinstance(name,str) or not name.strip():continue
     name=aliases.get(name.lower().strip(),name.lower().strip());objects[name]+=1
     attrs=obj.get('attributes') or {}
     if isinstance(attrs,dict):
      for field in ['color','material','shape','size']:
       v=attrs.get(field)
       if isinstance(v,str) and v.strip():attributes[name][field][v]+=1
    files+=1;seconds+=duration
   except Exception as e:errors.append({'session':str(session),'error':str(e)})
 result['datasets'][ds]={'files':files,'duration_s':seconds,'scenes':[{'category':c,'area':a,'hours':v/3600} for (c,a),v in scenes.items()],'actions':dict(actions),'objects':dict(objects),'attributes':{name:{field:dict(counts.most_common(5)) for field,counts in attrs.items()} for name,attrs in attributes.items()},'errors':errors}
print(json.dumps(result))
