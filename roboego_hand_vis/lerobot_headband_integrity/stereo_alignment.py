"""Shared L1/L3 first-frame rectified stereo alignment checks."""
from .common import result


def check_stereo_alignment(ctx, episode, *, geometry, provenance, camera_names,
                           frame_index=0, policy=None):
    """Rectify one selected zero-based frame using the depth input geometry."""
    import cv2
    from roboego_hand_vis.common.stereo_quality import evaluate, POLICY
    policy = dict(POLICY if policy is None else policy)
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import crop_roi, crop_output_maps
    from roboego_hand_vis.stereo_rectification.equidistant import EquidistantCameraModel
    from roboego_hand_vis.stereo_rectification.double_sphere import DoubleSphereCameraModel

    left, right, transform = geometry
    cameras = (left, right)
    if left['camera_model'] == 'ds':
        models = [DoubleSphereCameraModel(*c['intrinsics'][2:], *c['intrinsics'][:2])
                  for c in cameras]
    else:
        models = [EquidistantCameraModel(*c['intrinsics'], *c['distortion_coeffs'])
                  for c in cameras]
    size = tuple(left['resolution'])
    params = type(models[0]).stereo_rectify(
        *models, size, transform[:3, :3], transform[:3, 3], balance=0.)
    if (params['P1'][0, 0] <= 0 or params['P2'][0, 3] >= 0
            or abs(params['P2'][1, 3]) > 1e-6):
        raise ValueError('Rectification is not positive-disparity horizontal stereo')
    maps = [model.generate_stereo_remap(params[f'R{i}'], params[f'P{i}'], *size)
            for i, model in enumerate(models, 1)]
    maps, _, _, roi = crop_output_maps(maps, params, size)
    images = []
    for camera, remap in zip(camera_names, maps):
        path = ctx.video(episode, camera)
        cap = cv2.VideoCapture(str(path))
        try:
            for _ in range(frame_index + 1):
                ok, frame = cap.read()
                if not ok:
                    break
            if not ok:
                raise ValueError(f'Cannot read stereo frame index {frame_index}: {path}')
        finally:
            cap.release()
        x0, y0, x1, y1 = crop_roi((frame.shape[1], frame.shape[0]), size)
        images.append(cv2.remap(frame[y0:y1, x0:x1], *remap, cv2.INTER_LINEAR))
    metrics = evaluate(*images, policy=policy)
    return [result('horizontal_alignment', metrics['status'] == 'passed',
                   frame_index=frame_index, policy=policy, calibration_source=provenance,
                   crop_roi=list(roi), **metrics)]

