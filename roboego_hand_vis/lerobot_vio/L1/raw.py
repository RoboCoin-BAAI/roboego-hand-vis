"""Raw fisheye input preparation, with explicit center-crop consent."""
import json
import cv2
import yaml
from roboego_hand_vis.vio.calibration import prepare_openvins_calibration


def prepare_raw(plan, work, allow_center_crop=False, downsample_cameras=False, camera_sides=("left", "right")):
    source = yaml.safe_load(plan['camera_imu'].read_text(encoding='utf-8'))
    videos, crops = {}, {}
    for index, side in enumerate(camera_sides):
        camera = source[f'cam{index}']
        if camera['camera_model'] != 'pinhole' or camera['distortion_model'] not in ('equidistant', 'radtan'):
            raise ValueError('Unsupported raw camera model')
        target = tuple(camera['resolution'])
        capture = cv2.VideoCapture(str(plan[f'{side}_video']))
        writer = None
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open raw {side} video')
            actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if actual == target:
                videos[side] = plan[f'{side}_video']
                crops[side] = [0, 0, *target]
                continue
            if not allow_center_crop:
                raise ValueError(f'Raw {side} video {actual} differs from calibration {target}; '
                                 'confirm the same center-crop assumption as rectification with --raw-center-crop-calibration')
            if actual[0] < target[0] or actual[1] < target[1] or any(v % 2 for v in target):
                raise ValueError('Raw video cannot be center-cropped to calibration resolution')
            x, y = (actual[0]-target[0])//2, (actual[1]-target[1])//2
            crops[side] = [x, y, x+target[0], y+target[1]]
            destination = work / f'{side}_raw_cropped.mkv'
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'FFV1'),
                                     capture.get(cv2.CAP_PROP_FPS), target)
            if not writer.isOpened():
                raise ValueError('OpenCV FFV1 encoder unavailable for lossless raw crop')
            count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame[y:y+target[1], x:x+target[0]])
                count += 1
            writer.release()
            writer = None
            import pyarrow.parquet as pq
            if count != pq.read_metadata(plan[f'{side}_timestamps']).num_rows:
                raise ValueError('Raw decoded frame count differs from sensor timestamps')
            verify = cv2.VideoCapture(str(destination))
            try:
                if not verify.isOpened() or int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) != count:
                    raise ValueError('Lossless crop output frame count mismatch')
            finally:
                verify.release()
            videos[side] = destination
        finally:
            capture.release()
            if writer is not None:
                writer.release()
    calibration = prepare_openvins_calibration(plan['camera_imu'], plan['imu_calibration'],
                                               work / 'config', downsample_cameras=downsample_cameras, camera_ids=tuple(range(len(camera_sides))))
    (work / 'raw_input.json').write_text(json.dumps(dict(crops=crops, videos={k:str(v) for k,v in videos.items()},
        note='No undistortion or resizing; optional lossless center crop assumes calibration describes the central ROI.'), indent=2), encoding='utf-8')
    return calibration, videos
