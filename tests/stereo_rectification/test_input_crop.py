import cv2
import numpy as np
import pytest
import yaml

from roboego_hand_vis.stereo_rectification import rectify_video as rv


def write_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 30, (w, h))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_center_crop_and_noop():
    assert rv.calibration_crop_roi((1920, 1200), [1920, 1088]) == (0, 56, 1920, 1144)
    assert rv.calibration_crop_roi((1920, 1088), [1920, 1088]) == (0, 0, 1920, 1088)
    assert rv.calibration_crop_roi((1923, 1201), [1920, 1088]) == (1, 56, 1921, 1144)


@pytest.mark.parametrize('resolution', [None, [], [1920], [0, 1088], [-1, 1088],
                                        [1920.5, 1088], [True, 1088]])
def test_bad_calibration_resolution(resolution):
    with pytest.raises(ValueError, match='resolution'):
        rv.calibration_crop_roi((1920, 1200), resolution)


def test_smaller_video_rejected():
    with pytest.raises(ValueError, match='小于'):
        rv.calibration_crop_roi((960, 600), [1920, 1088])


def test_frame_is_cropped_before_remap(tmp_path):
    frame = np.zeros((80, 96, 3), dtype=np.uint8)
    frame[8:72] = (40, 120, 200)
    source = tmp_path / 'input.mkv'
    write_video(source, [frame, frame])
    mx, my = np.meshgrid(np.arange(96, dtype=np.float32), np.arange(64, dtype=np.float32))
    result = tmp_path / 'output.mp4'
    assert rv.process_video(str(source), str(result), mx, my, 30,
                            input_roi=(0, 8, 96, 72)) == 2
    cap = cv2.VideoCapture(str(result))
    ok, decoded = cap.read()
    cap.release()
    assert ok and decoded.shape == (64, 96, 3)
    np.testing.assert_allclose(decoded.mean(axis=(0, 1)), (40, 120, 200), atol=4)


def test_pipeline_matches_precropped_input(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)
    raw, cropped = tmp_path / 'raw', tmp_path / 'cropped'
    write_video(raw / 'videos/left.mkv', [frame])
    # Right stream has a different original size; it needs its own ROI.
    write_video(raw / 'videos/right.mkv', [frame[4:76]])
    for side in ('left', 'right'):
        write_video(cropped / 'videos' / f'{side}.mkv', [frame[8:72]])
    model = rv.EquidistantCameraModel(35, 35, 48, 32, 0.1, 0, 0, 0)
    rendered = []

    def capture(inputs, output, mlx, mly, mrx, mry, fps, input_rois=None):
        pair = []
        for index, (side, mx, my) in enumerate((('left', mlx, mly), ('right', mrx, mry))):
            cap = cv2.VideoCapture(str(inputs / 'videos' / f'{side}.mkv'))
            ok, image = cap.read()
            cap.release()
            assert ok
            if input_rois is not None:
                x0, y0, x1, y1 = input_rois[index]
                image = image[y0:y1, x0:x1]
            pair.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
        rendered.append(pair)

    monkeypatch.setattr(rv, '_process_videos', capture)
    params = []
    for inputs in (raw, cropped):
        result, size = rv.run_equidistant_pipeline(
            model, model, np.eye(3), np.array([-0.1, 0, 0]),
            96, 64, 0.0, inputs, inputs / 'output', 30,
            calibration_resolutions=([96, 64], [96, 64]))
        params.append(result)
    for a, b in zip(*rendered):
        np.testing.assert_array_equal(a, b)
    for key in ('P1', 'P2', 'Q'):
        np.testing.assert_array_equal(params[0][key], params[1][key])
    saved = yaml.safe_load((raw / 'output/calibrations/rectify_params.yaml').read_text())
    assert saved['input_crops']['left']['roi'] == [0, 8, 96, 72]
    assert saved['input_crops']['right']['roi'] == [0, 4, 96, 68]
    assert saved['input_crops']['left']['source_resolution'] == [96, 80]


def test_different_calibration_sizes_rejected(tmp_path):
    model = rv.EquidistantCameraModel(35, 35, 48, 32, 0.1, 0, 0, 0)
    with pytest.raises(ValueError, match='左右.*分辨率'):
        rv.run_equidistant_pipeline(
            model, model, np.eye(3), np.array([-0.1, 0, 0]),
            96, 64, 0.0, tmp_path, tmp_path / 'out', 30,
            calibration_resolutions=([96, 64], [96, 60]))
