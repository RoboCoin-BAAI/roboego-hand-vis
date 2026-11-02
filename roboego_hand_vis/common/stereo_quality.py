"""First-frame rectification gate shared by depth and VIO."""
import json
import os
from pathlib import Path
from uuid import uuid4

POLICY = dict(version=1, min_matches=30, min_cells=4, max_median_px=20., max_p95_px=40.)
GUANGLUN_POLICY = dict(version=3, min_matches=15, min_cells=0, max_median_px=20., max_p95_px=None)


def evaluate(left, right, policy=None):
    policy = POLICY if policy is None else policy
    import cv2
    import numpy as np
    if left.shape != right.shape:
        return dict(status='warning', reason='image_dimensions_differ', matches=0)
    detector = cv2.SIFT_create(nfeatures=3000)
    k0,d0 = detector.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),None)
    k1,d1 = detector.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),None)
    if d0 is None or d1 is None or min(len(d0),len(d1)) < 2:
        return dict(status='inconclusive',reason='insufficient_features',matches=0)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def good(a,b):
        return {m.queryIdx:m.trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance}
    forward,backward = good(d0,d1),good(d1,d0)
    pairs = [(k0[i].pt,k1[j].pt) for i,j in forward.items() if backward.get(j)==i]
    if not pairs:
        return dict(status='inconclusive',reason='insufficient_matches',matches=0)
    points = np.asarray(pairs)
    # Never filter by vertical error or fit it away before measuring it.
    dy = np.abs(points[:,0,1]-points[:,1,1])
    h,w = left.shape[:2]
    cells = {(min(3,int(x*4/w)),min(2,int(y*3/h))) for x,y in points[:,0]}
    median,p95 = float(np.median(dy)),float(np.percentile(dy,95))
    enough = len(pairs)>=policy['min_matches'] and len(cells)>=policy['min_cells']
    status = ('inconclusive' if not enough else 'passed'
              if median<=policy['max_median_px'] and
              (policy['max_p95_px'] is None or p95<=policy['max_p95_px']) else 'warning')
    return dict(status=status,reason='insufficient_matches_or_distribution' if not enough else
                'vertical_error_within_limits' if status=='passed' else 'excessive_vertical_error',
                matches=len(pairs),occupied_cells=len(cells),median_vertical_error_px=median,
                p95_vertical_error_px=p95,image_size=[w,h])


def require_stereo_quality(session, episode, left_video, right_video, *, policy=None):
    import cv2
    session,episode = Path(session),Path(episode)
    paths = [Path(left_video),Path(right_video)]
    identity = dict(policy=POLICY if policy is None else policy,videos=[dict(path=str(p.resolve()),size=p.stat().st_size,
                    mtime_ns=p.stat().st_mtime_ns) for p in paths])
    path = session/'annotation/stereo_quality'/f'{episode.parent.name}_{episode.name}.json'
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        report = {}
    if not isinstance(report,dict) or report.get('identity') != identity:
        images=[]
        for video in paths:
            cap=cv2.VideoCapture(str(video))
            try:
                ok,frame=cap.read()
                if not ok:
                    raise ValueError(f'Cannot read first stereo frame: {video}')
                images.append(frame)
            finally:
                cap.release()
        metrics = evaluate(*images) if policy is None else evaluate(*images, policy=policy)
        report=dict(identity=identity,frame_index=0,**metrics)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name('.'+path.name+'-'+uuid4().hex)
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    if report.get('status') != 'passed':
        raise ValueError(f"Stereo first-frame quality {report.get('status')}: {report.get('reason')}; report: {path}")
    return report
