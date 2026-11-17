"""Diagnostic coordinate hypotheses; input hand coordinates are not confirmed."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from roboego_hand_vis.lerobot_stereo_rectification.jianzhi.compute import load_geometry
from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import crop_output_maps
from roboego_hand_vis.stereo_rectification.double_sphere import DoubleSphereCameraModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[1, 150, 900])
    parser.add_argument('--point-frame', choices=['body', 'left-camera', 'world-left-camera'], default='body')
    parser.add_argument('--axis-signs', type=int, nargs=3, choices=[-1, 1], default=[1, 1, 1])
    parser.add_argument('--axis-order', choices=['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'], default='xyz')
    parser.add_argument('--hands', choices=['left', 'right', 'both'], default='both')
    parser.add_argument('--rotate-image-ccw', type=float, default=0., help='Rotate points about source camera optical Z; positive is visually counterclockwise')
    args = parser.parse_args()
    root, out = args.source, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    episode = root/'data/chunk-000/episode_000000'
    cal = json.loads((root/'calibrations/jianzhi-camera-calibration.json').read_text())
    (left, right, transform), provenance = load_geometry(root)
    models = [DoubleSphereCameraModel(*c['intrinsics'][2:], *c['intrinsics'][:2]) for c in [left, right]]
    size = tuple(left['resolution'])
    params = DoubleSphereCameraModel.stereo_rectify(*models, size, transform[:3,:3], transform[:3,3])
    maps = [m.generate_stereo_remap(params[f'R{i}'], params[f'P{i}'], *size) for i,m in enumerate(models,1)]
    maps, params, size, roi = crop_output_maps(maps, params, size)
    selected_hands = ['left', 'right'] if args.hands == 'both' else [args.hands]
    hands = {s:pq.read_table(episode/f'observation.hand_{s}_points.parquet').to_pydict() for s in selected_hands}
    head = (pq.read_table(episode/'observation.head_pose.parquet').to_pydict()
            if args.point_frame == 'world-left-camera' else None)
    label = f'{args.point_frame} {args.axis_order} signs={args.axis_signs} ccw={args.rotate_image_ccw:g} UNCONFIRMED'
    report = dict(coordinate_hypothesis=label, calibration_source=provenance, frames=[])
    chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
    for frame_index in args.frames:
        panels = [[],[]]
        record = dict(frame=frame_index, cameras={})
        for index,side in enumerate(['left','right']):
            camera = 'head_'+side
            times = pq.read_table(episode/f'observation.images.{camera}.parquet')['timestamp'].to_numpy()
            if not 0 <= frame_index < len(times):
                raise ValueError('Frame out of range')
            timestamp = float(times[frame_index])
            head_rotation = head_position = None
            if head is not None:
                hindex = int(np.argmin(abs(np.asarray(head['timestamp'])-timestamp)))
                if abs(head['timestamp'][hindex]-timestamp) > .05:
                    raise ValueError('No head pose within 50 ms')
                x,y,z,w = np.array([head[k][hindex] for k in ['qx','qy','qz','qw']], dtype=float)
                norm = np.linalg.norm([x,y,z,w])
                if not np.isfinite(norm) or norm < 1e-9:
                    raise ValueError('Invalid head quaternion')
                x,y,z,w = np.array([x,y,z,w])/norm
                head_rotation = np.array([
                    [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                head_position = np.array([head[k][hindex] for k in ['x','y','z']])
                record.setdefault('head_pose_matches', {})[side] = dict(row=hindex,
                    timestamp_delta_s=float(head['timestamp'][hindex]-timestamp))
            cap = cv2.VideoCapture(str(root/f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, original = cap.read(); cap.release()
            if not ok: raise ValueError('Cannot decode requested frame')
            rectified = cv2.remap(original, *maps[index], cv2.INTER_LINEAR)
            pose = np.array(cal[camera]['T_b_c_matrix'])
            timing = {}
            for hand,color in [('left',(0,255,0)),('right',(0,140,255))]:
                if hand not in hands:
                    continue
                table = hands[hand]
                row = int(np.argmin(abs(np.asarray(table['timestamp'])-timestamp)))
                delta = float(table['timestamp'][row]-timestamp)
                timing[hand] = dict(row=row, timestamp_delta_s=delta)
                if abs(delta) > .05: continue
                columns = [k for k in table if k not in ['timestamp','episode_index']]
                points = np.array([table[k][row] for k in columns]).reshape(21,3)
                points = points[:, ['xyz'.index(a) for a in args.axis_order]] * np.asarray(args.axis_signs)
                theta = np.deg2rad(args.rotate_image_ccw)
                # Optical Y points downward: image CCW uses negative Z rotation.
                rotation = np.array([[np.cos(theta), np.sin(theta), 0.],
                                     [-np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
                points = points@rotation.T
                if head is not None:
                    # T_world_left assumed: p_left = R_world_left.T (p_world - t).
                    points = (points-head_position)@head_rotation
                if args.point_frame == 'body':
                    points = (points-pose[:3,3])@pose[:3,:3]
                elif side == 'right':
                    points = points@transform[:3,:3].T + transform[:3,3]
                raw_uv = models[index].double_sphere_project(points)
                rays = points@params[f'R{index+1}'].T
                pixels = rays@params[f'P{index+1}'][:3,:3].T
                rect_uv = pixels[:,:2]/pixels[:,2:3]
                for image,uv,visible in [(original,raw_uv,np.isfinite(raw_uv).all(axis=1)&(raw_uv>=0).all(axis=1)),(rectified,rect_uv,(rays[:,2]>0)&np.isfinite(rect_uv).all(axis=1))]:
                    h,w = image.shape[:2]
                    visible &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                    for chain in chains:
                        for a,b in zip(chain,chain[1:]):
                            if visible[a] and visible[b]:cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,3)
                    for i in np.flatnonzero(visible):cv2.circle(image,tuple(np.rint(uv[i]).astype(int)),6,color,-1)
            for row,image in enumerate([original,rectified]):
                cv2.imwrite(str(out/f'frame_{frame_index:06d}_{side}_{["raw","rectified"][row]}.jpg'),image)
                image = cv2.resize(image,(800,650))
                cv2.putText(image,f'{side} {["raw","rectified"][row]} | {label}',(15,30),0,.5,(0,255,255),1)
                panels[row].append(image)
            record['cameras'][side] = dict(timestamp=timestamp,hands=timing)
        cv2.imwrite(str(out/f'frame_{frame_index:06d}_comparison.jpg'),cv2.vconcat([cv2.hconcat(p) for p in panels]))
        report['frames'].append(record)
    (out/'projection_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
