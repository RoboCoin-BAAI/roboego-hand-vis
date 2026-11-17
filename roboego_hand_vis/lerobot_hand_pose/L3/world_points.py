"""Convert MANUS nodes to OpenVR world points using a per-hand mounting hypothesis."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

NODES = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
LANDMARKS = ['wrist', 'thumbCMC', 'thumbMCP', 'thumbIP', 'thumbTip',
             'indexMCP', 'indexPIP', 'indexDIP', 'indexTip',
             'middleMCP', 'middlePIP', 'middleDIP', 'middleTip',
             'ringMCP', 'ringPIP', 'ringDIP', 'ringTip',
             'pinkyMCP', 'pinkyPIP', 'pinkyDIP', 'pinkyTip']
NAMES = [f'{name}_{axis}' for name in LANDMARKS for axis in 'xyz']
FEATURE = 'observation.hand_left_points'
R_TRACKER_SKELETON = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
# MANUS already encodes hand shape mirroring. Both mounts share the axis alignment.
R_TRACKER_SKELETON_RIGHT = R_TRACKER_SKELETON.copy()
T_TRACKER_WRIST = np.array([0., -.085, -.058])
CONVERTER_VERSION = 2
DEFAULT_MAX_GAP_S = .03


def normalize_tracker_times(pose_times, positions, rpy_deg):
    """Stable sort finite timestamps; last original row wins at duplicate times."""
    finite = np.isfinite(pose_times)
    order = np.flatnonzero(finite)
    order = order[np.argsort(pose_times[order], kind='stable')]
    sorted_times = pose_times[order]
    keep = np.r_[np.diff(sorted_times) != 0, True] if len(order) else np.array([], dtype=bool)
    selected = order[keep]
    report = dict(nonfinite_timestamp_rows=int((~finite).sum()),
                  duplicate_timestamp_rows=int(len(order)-len(selected)),
                  backward_steps=int((np.diff(pose_times[finite]) < 0).sum()),
                  usable_timestamp_rows=len(selected),
                  policy='stable timestamp sort; last original row per timestamp; discard nonfinite timestamps')
    return pose_times[selected], positions[selected], rpy_deg[selected], report


def rpy_quaternion(degrees):
    r,p,y=np.moveaxis(np.deg2rad(degrees)/2,-1,0)
    cr,cp,cy=np.cos(r),np.cos(p),np.cos(y)
    sr,sp,sy=np.sin(r),np.sin(p),np.sin(y)
    return np.stack([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy],axis=-1)


def rotate_vectors(q, vectors):
    xyz=q[:,None,:3]
    cross=2*np.cross(xyz,vectors)
    return vectors+q[:,None,3:4]*cross+np.cross(xyz,cross)


def world_points(times, points, pose_times, positions, rpy_deg, max_gap_s=DEFAULT_MAX_GAP_S,
                 alignment=None):
    """Interpolate the Tracker pose; never extrapolate or bridge long gaps."""
    times, pose_times = np.asarray(times), np.asarray(pose_times)
    points, positions, rpy_deg = map(np.asarray, (points, positions, rpy_deg))
    if (times.ndim != 1 or pose_times.ndim != 1 or not len(times)
            or points.ndim != 3 or points.shape[0] != len(times) or points.shape[2] != 3
            or positions.shape != (len(pose_times),3) or rpy_deg.shape != positions.shape):
        raise ValueError('Inconsistent skeleton or Tracker array shapes')
    pose_times, positions, rpy_deg, _ = normalize_tracker_times(pose_times, positions, rpy_deg)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s must be positive')
    if not len(pose_times):
        return np.full(points.shape, np.nan), np.zeros(len(times), dtype=bool)
    upper = np.searchsorted(pose_times, times, side='left')
    safe = np.clip(upper, 0, len(pose_times)-1)
    exact = times == pose_times[safe]
    hi = np.clip(upper, 1, len(pose_times)-1)
    lo = hi-1
    lo = np.where(exact, safe, lo)
    hi = np.where(exact, safe, hi)
    valid = (np.isfinite(times) & (times >= pose_times[0]) & (times <= pose_times[-1])
             & (pose_times[hi]-pose_times[lo] <= max_gap_s)
             & np.isfinite(points).all(axis=(1, 2))
             & np.isfinite(positions[lo]).all(axis=1) & np.isfinite(positions[hi]).all(axis=1)
             & np.isfinite(rpy_deg[lo]).all(axis=1) & np.isfinite(rpy_deg[hi]).all(axis=1))
    result = np.full(points.shape, np.nan)
    l,h=lo[valid],hi[valid]
    duration=pose_times[h]-pose_times[l]
    alpha=np.divide(times[valid]-pose_times[l],duration,
                    out=np.zeros(len(l)),where=duration>0)[:,None]
    q0,q1=rpy_quaternion(rpy_deg[l]),rpy_quaternion(rpy_deg[h])
    dot=np.sum(q0*q1,axis=1,keepdims=True)
    q1=np.where(dot<0,-q1,q1)
    dot=np.clip(abs(dot),0,1)
    theta=np.arccos(dot)
    sine=np.sin(theta)
    safe_sine=np.where(sine>1e-8,sine,1.)
    spherical=(np.sin((1-alpha)*theta)*q0+np.sin(alpha*theta)*q1)/safe_sine
    q=np.where(dot>.9995,(1-alpha)*q0+alpha*q1,spherical)
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    position=positions[l]+alpha*(positions[h]-positions[l])
    alignment=R_TRACKER_SKELETON if alignment is None else np.asarray(alignment)
    if (alignment.shape!=(3,3) or not np.isfinite(alignment).all()
            or not np.allclose(alignment.T@alignment,np.eye(3))
            or not np.isclose(np.linalg.det(alignment),1.)):
        raise ValueError('Alignment must be a proper right-handed rotation')
    local=points[valid]@alignment.T+T_TRACKER_WRIST
    result[valid]=rotate_vectors(q,local)+position[:,None,:]
    return result, valid


def _convert_side(session, max_gap_s, side, overwrite, staging):
    import pyarrow as pa
    import pyarrow.parquet as pq

    session = Path(session).resolve()
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    feature_name=f'observation.hand_{side}_points'
    nodes=[n+(25 if side=='right' else 0) for n in NODES]
    alignment=R_TRACKER_SKELETON if side=='left' else R_TRACKER_SKELETON_RIGHT
    info_path=staging/'meta/info.json'
    if not info_path.exists():
        info_path=session/'meta/info.json'
    info = json.loads(info_path.read_text(encoding='utf-8-sig'))
    episodes = sorted(p for p in (session/'data').glob('chunk-*/episode_*') if p.is_dir())
    if not episodes:
        raise ValueError('No episodes found')
    for episode in episodes:
        if (episode/f'{feature_name}.parquet').exists():
            if not overwrite:
                raise FileExistsError(f'Refusing to overwrite: {episode / (feature_name + ".parquet")}')
    summaries = []
    for episode in episodes:
        skeleton = episode/'observation.hand_skeleton.parquet'
        tracker = episode/f'observation.{side}_wrist_pose.parquet'
        s, p = pq.read_table(skeleton), pq.read_table(tracker)
        times = s['timestamp'].to_numpy()
        points = np.stack([np.stack([s[f'node{node:02d}_{axis}'].to_numpy()
                                    for axis in 'xyz'],axis=1) for node in nodes],axis=1)
        # Express nodes relative to the recorded root before applying mounting extrinsics.
        points = points-points[:, :1]
        xyz = np.stack([p[a].to_numpy() for a in 'xyz'],axis=1)
        rpy = np.stack([p[a].to_numpy() for a in ('roll','pitch','yaw')],axis=1)
        world, valid = world_points(times, points, p['timestamp'].to_numpy(), xyz, rpy,
                                    max_gap_s, alignment=alignment)
        _, _, _, tracker_time_report = normalize_tracker_times(p['timestamp'].to_numpy(), xyz, rpy)
        finite_times = times[np.isfinite(times)]
        timestamp_processing = dict(tracker=tracker_time_report, skeleton=dict(
            nonfinite_timestamp_rows=int((~np.isfinite(times)).sum()),
            duplicate_timestamp_rows=int(len(finite_times)-len(np.unique(finite_times))),
            backward_steps=int((np.diff(finite_times)<0).sum()),
            policy='preserve original rows and timestamps; nonfinite queries produce NaN points'))
        if not valid.any():
            raise ValueError(f'No synchronized valid poses: {episode}')
        dest = staging/episode.relative_to(session)
        if not np.all(s['episode_index'].to_numpy()==s['episode_index'][0].as_py()):
            raise ValueError(f'Mixed episode indices in skeleton: {episode}')
        if not np.all(p['episode_index'].to_numpy()==s['episode_index'][0].as_py()):
            raise ValueError(f'Tracker/skeleton episode mismatch: {episode}')
        dest.mkdir(parents=True,exist_ok=True)
        feature = pa.table(dict(timestamp=s['timestamp'], episode_index=s['episode_index'],
            **{name:pa.array(world.reshape(len(times),63)[:,i],type=pa.float32()) for i,name in enumerate(NAMES)}))
        feature = feature.replace_schema_metadata({
            b'coordinate_frame':b'OpenVR TrackingUniverseStanding world', b'unit':b'meter',
            b'extrinsic_status':b'user-confirmed mounting hypothesis; not independently calibrated',
            b'invalid_value':b'NaN: no finite pose bracket within max_gap_s'})
        pq.write_table(feature,dest/f'{feature_name}.parquet')
        summaries.append(dict(side=side, chunk=episode.parent.name, episode=episode.name,
                              samples=len(times), valid=int(valid.sum()),invalid=int((~valid).sum())))
        # Add only valid world coordinates to derived-feature statistics.
        stats_path=staging/'meta/episodes_stats.jsonl'
        if not stats_path.exists():
            stats_path=session/'meta/episodes_stats.jsonl'
        records=[json.loads(line) for line in stats_path.read_text(encoding='utf-8-sig').splitlines()
                 if line.strip()] if stats_path.exists() else []
        episode_index=s['episode_index'][0].as_py()
        record=next((r for r in records if r['episode_index']==episode_index),None)
        if record is None:
            record=dict(episode_index=episode_index,stats={})
            records.append(record)
        values=world[valid].reshape(-1,63)
        record.setdefault('stats',{})[feature_name]=dict(
            min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist(),
            mean=values.mean(axis=0).tolist(), std=values.std(axis=0).tolist(),count=[len(values)])
        (staging/'meta').mkdir(exist_ok=True)
        (staging/'meta/episodes_stats.jsonl').write_text(
            ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    spec = dict(dtype='float32',shape=[63],names=NAMES,landmark_order=LANDMARKS,
                coordinate_frame='OpenVR TrackingUniverseStanding world',translation_unit='m',
                source=f'MANUS skeleton + VIVE {side} wrist pose + fixed mounting transform',
                invalid_value='NaN',extrinsic_status='mounting hypothesis; not independently calibrated')
    for key in ('fps','tolerance_s','window'):
        if key in info['features']['observation.hand_skeleton']:
            spec[key]=info['features']['observation.hand_skeleton'][key]
    info['features'][feature_name]=spec
    (staging/'meta').mkdir(exist_ok=True)
    (staging/'meta/info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return summaries


def convert(session, max_gap_s=DEFAULT_MAX_GAP_S, side='both', overwrite=False):
    """Stage all requested sides/episodes before publishing any output."""
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import publish_files

    if side not in ('left','right','both'):
        raise ValueError('side must be left, right or both')
    session=Path(session).resolve()
    staging=session/f'.hand-world-{uuid4().hex}'
    try:
        summaries=[]
        for hand in (('left','right') if side=='both' else (side,)):
            summaries.extend(_convert_side(session,max_gap_s,hand,overwrite,staging))
        publish_files(staging,session)
        return summaries
    finally:
        if staging.exists():
            if staging.is_symlink() or staging.resolve().parent!=session:
                raise ValueError(f'Unsafe staging cleanup: {staging}')
            shutil.rmtree(staging)


if __name__=='__main__':
    from roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points import main
    raise SystemExit(main())
