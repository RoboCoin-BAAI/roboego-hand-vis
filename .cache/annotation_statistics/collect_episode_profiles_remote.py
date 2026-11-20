import json,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
result={}
def read_session(session):
 try:
  info=json.loads((session/'meta/info.json').read_text())
  annotation=json.loads((session/'annotation/episode_000000_structured.json').read_text())
  duration=float(info['total_frames'])/float(info['fps'])
  if not math.isfinite(duration) or duration<=0:raise ValueError('Invalid duration')
  subtasks=annotation.get('subtask')
  if not isinstance(subtasks,list):raise ValueError('subtask must be a list')
  return [duration,len(subtasks)],None
 except Exception as e:return None,{'session':str(session),'error':str(e)}
for ds in ['guanglun_depth','L3_lerobot']:
 root=Path('/share/project/liuyou/humancoin/data')/ds
 sessions=[]
 for batch in sorted(root.glob('batch_*')):
  if not batch.is_dir():continue
  for session in batch.iterdir():
   if session.is_dir() and (session/'annotation/episode_000000_structured.json').is_file():sessions.append(session)
 records=[];errors=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for record,error in pool.map(read_session,sessions):
   if error:errors.append(error)
   else:records.append(record)
 result[ds]={'records':records,'errors':errors}
print(json.dumps(result))
