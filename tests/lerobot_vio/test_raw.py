import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from roboego_hand_vis.lerobot_vio.L1 import raw


def test_raw_crop_requires_consent_and_preserves_frames(tmp_path, monkeypatch):
    source = tmp_path / 'camera.yaml'
    source.write_text(yaml.safe_dump({f'cam{i}': dict(camera_model='pinhole', distortion_model='equidistant',
                                                   resolution=[32, 16]) for i in range(2)}))
    plan = dict(camera_imu=source, imu_calibration=source)
    for side in ('left', 'right'):
        video = tmp_path / f'{side}.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 30., (32, 24))
        assert writer.isOpened()
        for i in range(3):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[4:20] = 30+i
            writer.write(frame)
        writer.release()
        timestamps = tmp_path / f'{side}.parquet'
        pq.write_table(pa.table(dict(timestamp=[0., 1/30, 2/30])), timestamps)
        plan[f'{side}_video'], plan[f'{side}_timestamps'] = video, timestamps
    with pytest.raises(ValueError, match='confirm'):
        raw.prepare_raw(plan, tmp_path)
    sentinel = object()
    monkeypatch.setattr(raw, 'prepare_openvins_calibration', lambda *a, **kw: sentinel)
    calibration, videos = raw.prepare_raw(plan, tmp_path, allow_center_crop=True)
    assert calibration is sentinel
    for video in videos.values():
        capture = cv2.VideoCapture(str(video))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            for i in range(3):
                ok, image = capture.read()
                assert ok and image.shape == (16, 32, 3)
                assert np.all(image == 30+i)
        finally:
            capture.release()


def test_raw_plan_without_rectification(tmp_path):
    from roboego_hand_vis.lerobot_vio.L1.compute import plan_session
    episode = tmp_path / 'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    (episode / 'observation.imu.parquet').touch()
    calibration = tmp_path / 'calibrations'
    calibration.mkdir()
    for name in ('camera_imu-camchain-imucam.yaml', 'camera_imu-imu.yaml'):
        (calibration / name).touch()
    for side in ('left', 'right'):
        key = f'observation.images.head_{side}'
        (episode / f'{key}.parquet').touch()
        folder = tmp_path / 'videos/chunk-000' / key
        folder.mkdir(parents=True)
        (folder / 'episode_000000.mp4').touch()
    plan = plan_session(tmp_path, input_mode='raw')[0]
    assert 'rectify' not in plan and 'manifest' not in plan
    assert plan['output'] == tmp_path / 'vio_raw/chunk-000/episode_000000'
