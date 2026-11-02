import cv2
import numpy as np
from roboego_hand_vis.lerobot_vio.L1.check_stereo import geometry, intrinsics


def test_known_fisheye_stereo_geometry():
    camera = dict(intrinsics=[500., 500., 320., 240.], distortion_coeffs=[.1, -.01, .001, 0.])
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-1., -.5, 2.], [1., .5, 5.], (40, 3))
    transform = np.eye(4)
    transform[0, 3] = -.1
    points = []
    for position in (xyz, xyz+transform[:3, 3]):
        p, _ = cv2.fisheye.projectPoints(position.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                         intrinsics(camera), np.array(camera['distortion_coeffs']))
        points.append(p.reshape(-1, 2))
    report = geometry(points, [camera, camera], transform)
    assert report['positive_depth_ratio'] == 1.
    assert report['epipolar_median_px_equivalent'] < 1e-8
    assert report['reprojection_median_px'] < 1e-8
    points[1][:, 1] += 5
    assert geometry(points, [camera, camera], transform)['epipolar_median_px_equivalent'] > 4
