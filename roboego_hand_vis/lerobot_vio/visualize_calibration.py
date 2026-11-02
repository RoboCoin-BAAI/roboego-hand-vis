"""Visualize Kalibr camera/IMU frames and cross-check calibration chains."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def load(session, episode_index):
    source = session / 'meta/calibrations.jsonl'
    if source.is_file():
        rows = [json.loads(s) for s in source.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        rows = [r for r in rows if r.get('episode_index') == episode_index]
        if len(rows) != 1:
            raise ValueError('Expected exactly one matching calibration episode')
        head = rows[0]['calibrations']['ego_headband']
        return head['camera_imu-camchain-imucam'], head['camera-camchain'], str(source)
    def read(name):
        return yaml.safe_load((session / 'calibrations' / name).read_text(encoding='utf-8'))
    return read('camera_imu-camchain-imucam.yaml'), read('camera-camchain.yaml'), str(session / 'calibrations')


def rigid(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or
            not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return t


def visualize(session, output, episode_index=0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    imucam, chain, source = load(session, episode_index)
    keys = sorted((k for k in imucam if k.startswith('cam')), key=lambda k: int(k[3:]))
    poses = {k: np.linalg.inv(rigid(imucam[k]['T_cam_imu'])) for k in keys}
    # T_cam_imu maps IMU points into camera coordinates. Its inverse is the camera pose in IMU.
    left_keys = [k for k in keys if '/left/' in imucam[k].get('rostopic', '')]
    anchor = left_keys[0] if len(left_keys) == 1 else keys[0]
    anchor_topic = imucam[anchor].get('rostopic')
    chain_anchor = next((k for k, c in chain.items() if c.get('rostopic') == anchor_topic), anchor)
    predicted = {chain_anchor: poses[anchor]}
    chain_keys = sorted((k for k in chain if k.startswith('cam')), key=lambda k: int(k[3:]))
    for prev, key in zip(chain_keys, chain_keys[1:]):
        if prev in predicted and 'T_cn_cnm1' in chain[key]:
            predicted[key] = predicted[prev] @ np.linalg.inv(rigid(chain[key]['T_cn_cnm1']))
    metrics = []
    for key in keys:
        camera = imucam[key]
        ck = next((k for k, c in chain.items() if c.get('rostopic') == camera.get('rostopic')), key)
        row = dict(camera=key, topic=camera.get('rostopic'), position_in_imu_m=poses[key][:3, 3].tolist(),
                   distance_to_imu_m=float(np.linalg.norm(poses[key][:3, 3])),
                   timeshift_cam_imu_s=camera.get('timeshift_cam_imu'))
        if ck in predicted:
            delta = poses[key][:3, :3].T @ predicted[ck][:3, :3]
            row.update(chain_position_difference_mm=float(np.linalg.norm(poses[key][:3, 3]-predicted[ck][:3, 3])*1000),
                       chain_rotation_difference_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))))
        metrics.append(row)
    right_keys = [k for k in keys if '/right/' in imucam[k].get('rostopic', '')]
    baseline = float(np.linalg.norm(poses[right_keys[0]][:3, 3]-poses[anchor][:3, 3])) if right_keys else None
    output.mkdir(parents=True, exist_ok=True)
    report = dict(source=source, episode_index=episode_index, anchor=anchor,
                  convention='T_cam_imu maps IMU to camera; plotted poses are its inverse. Camera X right, Y down, Z forward.',
                  chain_comparison='Stereo chain anchored at the same left camera pose; this checks internal consistency, not ground-truth accuracy.',
                  baseline_from_imucam_m=baseline, cameras=metrics)
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig = plt.figure(figsize=(16, 8))
    for panel, (reference, title) in enumerate([(np.eye(4), 'IMU reference frame'),
                                                (np.linalg.inv(poses[anchor]), 'Left camera reference frame')], 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        positions = []
        def draw(name, pose, dashed=False):
            pose = reference @ pose
            o = pose[:3, 3]; positions.append(o)
            ax.scatter(*o, s=30, c='black' if not dashed else 'orange')
            if not dashed:
                ax.text(*o, name, fontsize=8)
            for i, color in enumerate(['red', 'green', 'blue']):
                end = o + pose[:3, i]*.045
                ax.plot(*np.stack([o, end]).T, color=color, linestyle='--' if dashed else '-', alpha=.65 if dashed else 1)
                if not dashed:
                    ax.text(*end, 'XYZ'[i], color=color, fontsize=7)
        draw('IMU', np.eye(4))
        for key in keys:
            topic = imucam[key].get('rostopic', key).split('/')
            label = topic[2] if len(topic)>2 else key
            draw(f'{key} {label}', poses[key])
        for key, pose in predicted.items():
            if key != chain_anchor:
                draw(f'{key} stereo-chain', pose, True)
        pts = np.asarray(positions); center = (pts.min(0)+pts.max(0))/2
        radius = max(.12, float(np.ptp(pts, axis=0).max())*.65)
        for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            setter(center[dim]-radius, center[dim]+radius)
        ax.set_box_aspect((1, 1, 1)); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(title); ax.view_init(elev=22, azim=-55)
    fig.suptitle(session.name+'\nAxes: X red / Y green / Z blue. Solid: camera-IMU calibration. Dashed: stereo chain.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / 'calibration.png', dpi=180)
    fig.savefig(output / 'calibration.pdf')
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print('Output:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--episode-index', type=int, default=0)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    session = args.source.resolve()
    visualize(session, args.output or session / 'annotation/calibration_visualization', args.episode_index)
