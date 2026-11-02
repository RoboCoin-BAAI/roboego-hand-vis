"""Independent sampled stereo geometry check; does not run VIO or edit datasets."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def intrinsics(camera):
    fx, fy, cx, cy = camera['intrinsics']
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])


def matches(left, right):
    detector = cv2.ORB_create(nfeatures=4000)
    a, da = detector.detectAndCompute(left, None)
    b, db = detector.detectAndCompute(right, None)
    if da is None or db is None:
        raise ValueError('No descriptors')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = [pair[0] for pair in matcher.knnMatch(da, db, k=2)
               if len(pair) == 2 and pair[0].distance < .65*pair[1].distance]
    reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(db, da, k=2)
               if len(pair) == 2 and pair[0].distance < .65*pair[1].distance}
    good = [m for m in forward if reverse.get(m.trainIdx) == m.queryIdx]
    if len(good) < 8:
        raise ValueError('Fewer than eight mutually matched descriptors')
    return (np.array([a[m.queryIdx].pt for m in good], dtype=float),
            np.array([b[m.trainIdx].pt for m in good], dtype=float))


def geometry(points, cameras, transform):
    normalized = [cv2.fisheye.undistortPoints(p.reshape(-1, 1, 2), intrinsics(c),
                  np.array(c['distortion_coeffs'], dtype=float)).reshape(-1, 2) for p, c in zip(points, cameras)]
    a, b = [np.column_stack([p, np.ones(len(p))]) for p in normalized]
    rotation, translation = transform[:3, :3], transform[:3, 3]
    x, y, z = translation
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    essential = cross @ rotation
    line_b, line_a = a @ essential.T, b @ essential
    numerator = np.abs(np.sum(b*line_b, axis=1))
    f = np.mean([c['intrinsics'][0] for c in cameras])
    error = .5*numerator*(1/np.maximum(np.linalg.norm(line_b[:, :2], axis=1), 1e-12)
                          +1/np.maximum(np.linalg.norm(line_a[:, :2], axis=1), 1e-12))*f
    homogeneous = cv2.triangulatePoints(np.eye(3, 4), transform[:3], normalized[0].T, normalized[1].T)
    with np.errstate(divide='ignore', invalid='ignore'):
        xyz = (homogeneous[:3]/homogeneous[3]).T
    xyz_right = xyz @ rotation.T + translation
    positive = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz_right[:, 2] > 0)
    reprojection = np.full(len(a), np.nan)
    if positive.any():
        errors = []
        for position, p, c in zip((xyz, xyz_right), points, cameras):
            projected, _ = cv2.fisheye.projectPoints(position[positive].reshape(-1, 1, 3),
                             np.zeros(3), np.zeros(3), intrinsics(c), np.array(c['distortion_coeffs'], dtype=float))
            errors.append(np.linalg.norm(projected.reshape(-1, 2)-p[positive], axis=1))
        reprojection[positive] = np.mean(errors, axis=0)
    def median(v):
        return float(np.median(v)) if len(v) else None
    return dict(match_count=len(a), epipolar_median_px_equivalent=median(error),
                epipolar_p90_px_equivalent=float(np.quantile(error, .9)),
                epipolar_under_2px_ratio=float(np.mean(error < 2)), positive_depth_ratio=float(positive.mean()),
                reprojection_median_px=median(reprojection[positive]),
                positive_depth_median_m=median(xyz[positive, 2]))


def frame(session, side, index, rectified=False):
    suffix = '_rectified' if rectified else ''
    path = session / 'videos/chunk-000' / f'observation.images.head_{side}{suffix}' / 'episode_000000.mp4'
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode frame {index}: {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def check(session, frames):
    session = Path(session)
    stereo = yaml.safe_load((session / 'calibrations/camera-camchain.yaml').read_text())
    imucam = yaml.safe_load((session / 'calibrations/camera_imu-camchain-imucam.yaml').read_text())
    transforms = dict(stereo=np.array(stereo['cam1']['T_cn_cnm1']),
                      imucam=np.array(imucam['cam1']['T_cam_imu']) @ np.linalg.inv(np.array(imucam['cam0']['T_cam_imu'])))
    rows = []
    for index in frames:
        images = [frame(session, side, index) for side in ('left', 'right')]
        # Same confirmed center ROI used by the existing rectifier and raw runner.
        cropped = []
        for image, c in zip(images, (stereo['cam0'], stereo['cam1'])):
            w, h = c['resolution']
            if image.shape[0] < h or image.shape[1] < w:
                raise ValueError('Raw video smaller than calibration')
            y, x = (image.shape[0]-h)//2, (image.shape[1]-w)//2
            cropped.append(image[y:y+h, x:x+w])
        points = matches(*cropped)
        row = dict(frame_index=index)
        for name, calibration in (('stereo', stereo), ('imucam', imucam)):
            row[name] = geometry(points, [calibration['cam0'], calibration['cam1']], transforms[name])
        try:
            a, b = matches(*[frame(session, side, index, True) for side in ('left', 'right')])
            vertical = np.abs(a[:, 1]-b[:, 1])
            row['rectified'] = dict(match_count=len(a), vertical_median_px=float(np.median(vertical)),
                                    vertical_p90_px=float(np.quantile(vertical, .9)))
        except ValueError as exc:
            row['rectified'] = dict(error=str(exc))
        rows.append(row)
    return dict(session=str(session), frames=rows,
                method='ORB 0.65 ratio and mutual matching; identical raw matches for both calibrations. '
                       'No VIO pose or calibration-based match filtering. Epipolar pixel equivalent uses normalized '
                       'coordinates times mean fx, not exact fisheye pixel distance. False matches can remain. '
                       'Center crop assumed; sampled geometry only, not proof of calibration accuracy.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--frames', default='100,258,259,260,349,438,506,536')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = check(args.source.resolve(), [int(v) for v in args.frames.split(',')])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
