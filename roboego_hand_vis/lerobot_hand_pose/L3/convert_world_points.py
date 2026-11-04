"""Convert L3 glove skeletons and wrist Trackers to world-space 21-point hands."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from roboego_hand_vis.lerobot_hand_pose.L3.world_points import (
    convert, NODES, R_TRACKER_SKELETON, R_TRACKER_SKELETON_RIGHT, T_TRACKER_WRIST, CONVERTER_VERSION, DEFAULT_MAX_GAP_S)


def discover(source):
    source=Path(source).expanduser().resolve()
    if (source/'meta/info.json').is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f'Source directory not found: {source}')
    found=set()
    for info in source.rglob('info.json'):
        if (info.parent.name=='meta' and
                not any(p.startswith('.') for p in info.relative_to(source).parts)):
            session=info.parent.parent.resolve()
            if session.is_relative_to(source):
                found.add(session)
    if not found:
        raise ValueError(f'No sessions with meta/info.json: {source}')
    return sorted(found)


def reusable(session,side,max_gap_s):
    """Reuse only complete, unchanged outputs produced by this converter."""
    hands=('left','right') if side=='both' else (side,)
    episodes=sorted(p for p in (session/'data').glob('chunk-*/episode_*') if p.is_dir())
    if not episodes:
        return False
    info=json.loads((session/'meta/info.json').read_text(encoding='utf-8-sig'))
    for ep in episodes:
        for hand in hands:
            feature=f'observation.hand_{hand}_points'
            output=ep/f'{feature}.parquet'
            try:
                p=json.loads((ep/f'hand_{hand}_points_provenance.json').read_text(encoding='utf-8'))
                nodes=[n+(25 if hand=='right' else 0) for n in NODES]
                rotation=R_TRACKER_SKELETON if hand=='left' else R_TRACKER_SKELETON_RIGHT
                if (output.is_symlink() or p.get('converter')!='lerobot_hand_pose.L3.world_points'
                        or p.get('converter_version')!=CONVERTER_VERSION or p.get('feature')!=feature
                        or p.get('source_nodes')!=nodes or p.get('max_gap_s')!=max_gap_s
                        or p.get('R_tracker_skeleton')!=rotation.tolist()
                        or p.get('t_tracker_wrist_m')!=T_TRACKER_WRIST.tolist()
                        or p.get('output_sha256')!=hashlib.sha256(output.read_bytes()).hexdigest()
                        or info.get('features',{}).get(feature,{}).get('coordinate_frame')
                           !='OpenVR TrackingUniverseStanding world'):
                    return False
                for name in ('observation.hand_skeleton.parquet',f'observation.{hand}_wrist_pose.parquet'):
                    if p.get('source_sha256',{}).get(name)!=hashlib.sha256((ep/name).read_bytes()).hexdigest():
                        return False
            except (OSError,ValueError,TypeError,AttributeError):
                return False
    return True


def worker(session,side,max_gap_s,overwrite):
    try:
        if not overwrite and reusable(session,side,max_gap_s):
            return dict(session=str(session),status='reused')
        reports=convert(session,max_gap_s,side,overwrite)
        return dict(session=str(session),status='done',episodes=reports)
    except Exception as exc:
        return dict(session=str(session),status='failed',error=f'{type(exc).__name__}: {exc}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='One session, batch directory, or batches root')
    parser.add_argument('--side',choices=('both','left','right'),default='both')
    parser.add_argument('--workers',type=int,default=1,help='Parallel sessions; each session publishes both hands together')
    parser.add_argument('--max-gap-s',type=float,default=DEFAULT_MAX_GAP_S,help='Maximum pose interpolation bracket in seconds (default 0.03)')
    parser.add_argument('--overwrite',action='store_true',help='Regenerate this converter\'s previous outputs')
    parser.add_argument('--dry-run',action='store_true',help='List sessions and outputs; no writes')
    args=parser.parse_args(argv)
    if args.workers<1 or not math.isfinite(args.max_gap_s) or args.max_gap_s<=0:
        parser.error('workers and max-gap-s must be positive')
    try:
        sessions=discover(args.source)
    except (OSError,ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for session in sessions:
            print(json.dumps(dict(session=str(session),side=args.side,
                features=[f'observation.hand_{h}_points' for h in
                          (('left','right') if args.side=='both' else (args.side,))]),ensure_ascii=False))
        return 0
    failed=0
    if args.workers==1:
        results=(worker(s,args.side,args.max_gap_s,args.overwrite) for s in sessions)
        for item in results:
            failed+=item['status']=='failed'
            print(json.dumps(item,ensure_ascii=False),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(worker,s,args.side,args.max_gap_s,args.overwrite):s for s in sessions}
            for future in as_completed(futures):
                try:
                    item=future.result()
                except Exception as exc:
                    item=dict(session=str(futures[future]),status='failed',error=str(exc))
                failed+=item['status']=='failed'
                print(json.dumps(item,ensure_ascii=False),flush=True)
    print(json.dumps(dict(total=len(sessions),failed=failed)))
    return int(bool(failed))


if __name__=='__main__':
    raise SystemExit(main())
