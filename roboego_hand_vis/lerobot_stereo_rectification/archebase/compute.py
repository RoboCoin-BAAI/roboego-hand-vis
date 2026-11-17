"""Undistort Archebase cameras and project both hands into the new left image.

This is independent monocular undistortion, not stereo epipolar rectification.
The virtual camera keeps the source optical axes (R=I), so XYZ is unchanged.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from roboego_hand_vis.lerobot_headband_integrity.common import Context
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import Encoder, publish_files


def geometry(camera_info, balance=0.):
    if not np.isfinite(balance) or not 0 <= balance <= 1:
        raise ValueError('balance must be between 0 and 1')
    if camera_info['distortion_model'] != 'equidistant':
        raise ValueError('Expected Archebase equidistant camera')
    k = np.asarray(camera_info['k'], dtype=np.float64).reshape(3, 3)
    d = np.asarray(camera_info['d'], dtype=np.float64)
    size = (int(camera_info['width']), int(camera_info['height']))
    if (d.shape != (4,) or not np.isfinite(k).all() or not np.isfinite(d).all()
            or min(size) <= 0 or k[0,0] <= 0 or k[1,1] <= 0):
        raise ValueError('Invalid camera calibration')
    rotation = np.eye(3)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, size, rotation, balance=balance)
    maps = cv2.fisheye.initUndistortRectifyMap(k, d, rotation, new_k, size, cv2.CV_32FC1)
    return dict(K=k, D=d, R=rotation, K_new=new_k, size=size, maps=maps)


def project_points(xyz, rotation, intrinsics):
    """Return transformed XYZ and pinhole UV; unavailable/behind points are NaN."""
    transformed = np.asarray(xyz, dtype=np.float64) @ np.asarray(rotation).T
    homogeneous = transformed @ np.asarray(intrinsics).T
    uv = np.full(transformed.shape[:-1]+(2,), np.nan)
    valid = np.isfinite(transformed).all(axis=-1) & (transformed[...,2] > 0)
    uv[valid] = homogeneous[valid,:2] / homogeneous[valid,2:3]
    return transformed, uv


def numeric_stats(values):
    a = np.asarray(values, dtype=float)
    # Missing projections remain NaN in Parquet, but JSON stats must be finite.
    result = {}
    for name, function in [('min',np.min),('max',np.max),('mean',np.mean),('std',np.std)]:
        result[name] = [float(function(v[np.isfinite(v)])) if np.isfinite(v).any() else None for v in a.T]
    result['count'] = [len(a)]
    return result


def transform_hands(ctx, episode, staging, left_geometry, info):
    updates = {}
    for side in ('left','right'):
        source_key = f'observation.hand_{side}_points'
        table = pq.read_table(episode/f'{source_key}.parquet')
        names = list(ctx.info['features'][source_key]['names'])
        xyz = np.column_stack([table[name].to_numpy() for name in names]).reshape(-1,21,3)
        transformed, uv = project_points(xyz, left_geometry['R'], left_geometry['K_new'])
        for key, values, columns in (
                (source_key+'_undistorted', transformed.reshape(-1,63), names),
                (f'observation.hand_{side}_points_2d_undistorted', uv.reshape(-1,42),
                 [name[:-2]+'_'+a for name in names[::3] for a in ('u','v')])):
            arrays = {n:table[n] for n in ('timestamp','episode_index')}
            arrays.update({name:pa.array(values[:,i], type=pa.float32()) for i,name in enumerate(columns)})
            path = staging/episode.relative_to(ctx.root)/f'{key}.parquet'
            path.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.table(arrays),path,compression='snappy')
            spec = deepcopy(ctx.info['features'][source_key])
            spec.update(dtype='float32', shape=[values.shape[1]], names=columns,
                        source_feature=source_key, source_frame_id='head_left_undistorted_optical',
                        coordinate_frame='head_left_undistorted_optical',
                        coordinate_policy='X_new = R @ X_source; virtual optical axes unchanged (R=I)',
                        image_feature='observation.images.head_left_undistorted')
            if values.shape[1] == 42:
                spec.update(projection='pinhole K_new, zero distortion',units='pixel',
                            missing_policy='nonfinite or behind-camera points produce NaN; outside-image pixels retained')
            info['features'][key] = spec
            updates[key] = dict(stats=numeric_stats(values), length=len(table),
                                from_timestamp=float(table['timestamp'][0].as_py()),
                                to_timestamp=float(table['timestamp'][-1].as_py()))
    return updates


def process_session(session, *, balance=0., ffmpeg='ffmpeg', dry_run=False):
    ctx = Context(Path(session))
    path = ctx.root/'calibrations/camera_info.json'
    raw = path.read_bytes()
    source = json.loads(raw)
    cameras = {}
    for camera in ('head_left','head_right'):
        entries = source[camera]
        if len(entries) != 1:
            raise ValueError('Time-varying calibration needs per-frame maps: '+camera)
        cameras[camera] = geometry(entries[0]['camera_info'],balance)
        if ctx.info['features'][f'observation.images.{camera}']['shape'][:2] != list(cameras[camera]['size'][::-1]):
            raise ValueError('Video/calibration resolution mismatch: '+camera)
    plan = dict(mode='independent_fisheye_undistortion',stereo_epipolar_rectification=False,
                coordinate_policy='Both hands use the source LEFT camera coordinates; R=I, XYZ unchanged',
                cameras={name:dict(K_new=g['K_new'].tolist(),R=g['R'].tolist(),resolution=list(g['size']))
                         for name,g in cameras.items()})
    if dry_run:
        return plan
    for camera in cameras:
        if f'observation.images.{camera}_undistorted' in ctx.info['features']:
            raise FileExistsError('Undistorted features already registered; refusing overwrite')
    staging = ctx.root/f'.archebase-undistortion.partial-{uuid4().hex}'
    staging.mkdir()
    info = deepcopy(ctx.info)
    stats_path = ctx.root/'meta/stats.json'
    stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.exists() else {}
    episode_updates = {}
    counts = {}
    try:
        for episode in ctx.episodes:
            record = {}
            for camera,g in cameras.items():
                key = f'observation.images.{camera}_undistorted'
                relative = Path('videos')/episode.parent.name/key/f'{episode.name}.mp4'
                if (ctx.root/relative).exists():
                    raise FileExistsError(ctx.root/relative)
                times = pq.read_table(episode/f'observation.images.{camera}.parquet')
                cap = cv2.VideoCapture(str(ctx.video(episode,camera)))
                encoder = None
                count = 0
                try:
                    if not cap.isOpened():
                        raise ValueError('Cannot open video: '+camera)
                    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    if actual != g['size']:
                        raise ValueError('Actual video/calibration dimensions differ')
                    encoder = Encoder(staging/relative,g['size'],ctx.fps(camera),ffmpeg,preset='veryfast',threads=4)
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        mapped = cv2.remap(frame,*g['maps'],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                        encoder.write(mapped)
                        if count == 0:
                            preview = staging/'preview'/f'{episode.name}_{camera}_undistorted.jpg'
                            preview.parent.mkdir(parents=True,exist_ok=True)
                            cv2.imwrite(str(preview),mapped)
                        count += 1
                        if count == 1 or count % 500 == 0:
                            print(f'[{ctx.root.name}/{camera}] {count}',flush=True)
                    encoder.close()
                    encoder = None
                finally:
                    cap.release()
                    if encoder is not None:
                        encoder.close()
                if count != len(times):
                    raise ValueError(f'{camera}: output count {count} != source timestamps {len(times)}')
                destination = staging/episode.relative_to(ctx.root)/f'{key}.parquet'
                destination.parent.mkdir(parents=True,exist_ok=True)
                pq.write_table(times,destination,compression='snappy')
                spec = deepcopy(ctx.info['features'][f'observation.images.{camera}'])
                spec.update(camera_model='pinhole', distortion_model='none',distortion_coeffs=[],
                            intrinsics=[float(g['K_new'][i,j]) for i,j in ((0,0),(1,1),(0,2),(1,2))],
                            derivation='independent_fisheye_undistortion')
                spec.setdefault('info',{}).update({'video.codec':'h264','video.pix_fmt':'yuv420p'})
                info['features'][key] = spec
                counts[f'{episode.name}/{camera}'] = count
            record.update(transform_hands(ctx,episode,staging,cameras['head_left'],info))
            episode_updates[int(episode.name.split('_')[-1])] = record
            # Overlay each hand on the first undistorted LEFT image for review.
            image = cv2.imread(str(staging/'preview'/f'{episode.name}_head_left_undistorted.jpg'))
            for side,color in [('left',(0,255,0)),('right',(0,160,255))]:
                table = pq.read_table(staging/episode.relative_to(ctx.root)/f'observation.hand_{side}_points_2d_undistorted.parquet')
                ts = pq.read_table(episode/'observation.images.head_left.parquet')['timestamp'][0].as_py()
                row = int(np.argmin(abs(table['timestamp'].to_numpy()-ts)))
                uv = np.array([table[n][row].as_py() for n in table.column_names[2:]]).reshape(21,2)
                for chain in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)]:
                    for a,b in zip(chain,chain[1:]):
                        if np.isfinite(uv[[a,b]]).all() and np.max(abs(uv[[a,b]]))<1e6:
                            cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,2)
                for i,p in enumerate(uv):
                    if np.isfinite(p).all() and 0<=p[0]<image.shape[1] and 0<=p[1]<image.shape[0]:
                        cv2.circle(image,tuple(np.rint(p).astype(int)),4,color,-1)
            cv2.imwrite(str(staging/'preview'/f'{episode.name}_hands_undistorted.jpg'),image)
        for records in episode_updates.values():
            for key,value in records.items():
                # Aggregate across episodes below when multiple episodes are present.
                stats.setdefault(key,value['stats'])
        if len(ctx.episodes)>1:
            for key in next(iter(episode_updates.values())):
                arrays=[]
                for episode in ctx.episodes:
                    table=pq.read_table(staging/episode.relative_to(ctx.root)/f'{key}.parquet')
                    arrays.append(np.column_stack([table[n].to_numpy() for n in table.column_names[2:]]))
                stats[key]=numeric_stats(np.concatenate(arrays))
        (staging/'meta').mkdir(exist_ok=True)
        for filename in ('episodes.jsonl','episodes_stats.jsonl'):
            source_path = ctx.root/'meta'/filename
            if not source_path.exists():
                continue
            rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            for row in rows:
                for key,value in episode_updates.get(row['episode_index'],{}).items():
                    if filename=='episodes_stats.jsonl':
                        row.setdefault('stats',{})[key]=value['stats']
                    else:
                        for field in ('length','from_timestamp','to_timestamp'):
                            row[f'data/{key}/{field}']=value[field]
            (staging/'meta'/filename).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
        info['total_videos'] = ctx.info.get('total_videos',0)+2*len(ctx.episodes)
        (staging/'meta/info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (staging/'meta/stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        plan.update(balance=balance,frame_counts=counts,calibration_sha256=hashlib.sha256(raw).hexdigest(),
                    encoding='libx264 veryfast CRF18, no B frames; required image remap; no extra decode QC')
        (staging/'calibrations').mkdir(exist_ok=True)
        (staging/'calibrations/archebase_undistortion.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
        publish_files(staging,ctx.root)
        return plan
    except BaseException:
        print(f'Incomplete output retained: {staging}',flush=True)
        raise
