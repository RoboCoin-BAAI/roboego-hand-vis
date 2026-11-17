"""Validate published rectification before reusing its videos."""
import json
from pathlib import Path


def validate_existing(ctx, episode, *, manifest_path, params_path, mapping_path,
                      video_relative, timestamp_relative, params, size, crops,
                      output_roi, transform, balance, camera_names=None, calibration_source=None, full_decode_validation=True):
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import yaml

    camera_names = camera_names or {'left': 'head_left', 'right': 'head_right'}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    saved = yaml.safe_load(params_path.read_text(encoding='utf-8'))
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(saved['balance'] == balance, 'balance changed')
    require(saved['resolution'] == list(size), 'resolution changed')
    require(saved['input_crops'] == [list(c) for c in crops], 'input crop changed')
    require(saved['crop_roi'] == dict(zip(('x_min', 'y_min', 'x_max', 'y_max'), output_roi)), 'output crop changed')
    require(saved['frame_policy'] == 'source_order' and manifest['frame_policy'] == 'source_order', 'frame policy changed')
    require(manifest['reference_camera'] == 'left', 'reference camera changed')
    for key, value in {**params, 'left_to_right': transform}.items():
        actual = np.asarray(saved[key])
        require(actual.shape == value.shape and np.allclose(actual, value, rtol=1e-7, atol=1e-9), f'{key} changed')
    if calibration_source is not None:
        require(manifest.get('calibration_source') == calibration_source, 'calibration source changed')
    mapping = pq.read_table(mapping_path).to_pydict() if mapping_path is not None else None
    counts = {}
    for side in ('left', 'right'):
        video = ctx.root / video_relative[side]
        meta = manifest['videos'][side]
        require((meta['width'], meta['height']) == size, f'{side} manifest size')
        require(abs(meta['fps'] - ctx.fps(camera_names[side])) < 1e-6, f'{side} fps changed')
        camera = f'observation.images.{camera_names[side]}_rectified'
        require(ctx.info['features'][camera]['shape'] == [size[1], size[0], 3], f'{side} feature shape')
        original = pq.read_table(episode / f'observation.images.{camera_names[side]}.parquet')
        copied = pq.read_table(ctx.root / timestamp_relative[side])
        require(original.equals(copied), f'{side} timestamps changed')
        raw = cv2.VideoCapture(str(ctx.video(episode, camera_names[side])))
        try:
            require(raw.isOpened(), f'{side} source unreadable')
            require(int(raw.get(cv2.CAP_PROP_FRAME_COUNT)) == meta['frame_count'], f'{side} source frame count changed')
        finally:
            raw.release()
        cap = cv2.VideoCapture(str(video))
        try:
            require(cap.isOpened(), f'{side} output unreadable')
            require((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == size, f'{side} video size')
            require(abs(cap.get(cv2.CAP_PROP_FPS) - meta['fps']) < .001, f'{side} video fps')
            if full_decode_validation:
                count = 0
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    count += 1
            else:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            require(count > 0 and count == meta['frame_count'] == original.num_rows, f'{side} decoded frame count')
            counts[side] = count
        finally:
            cap.release()
        intrinsics = manifest['intrinsics'][side]
        projection = params['P1' if side == 'left' else 'P2']
        require(np.allclose([intrinsics[k] for k in ('fx', 'fy', 'cx', 'cy')],
                            [projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2]]), f'{side} intrinsics')
    require(np.isclose(manifest['extrinsics']['baseline_m'], -params['P2'][0, 3] / params['P2'][0, 0]), 'baseline changed')
    length = max(counts.values())
    if mapping is not None:
        require(mapping['frame_index'] == list(range(length)), 'mapping frame indices')
        for side in ('left', 'right'):
            require(mapping[f'{side}_source_index'] == [i if i < counts[side] else None for i in range(length)], f'{side} mapping')
    # Paths are derived from this dataset layout, never followed from the old manifest.
    manifest['source_episode'] = str(episode)
    if mapping_path is not None:
        manifest['frame_mapping'] = str(mapping_path)
    else:
        manifest.pop('frame_mapping', None)
    manifest['dataset_name'] = ctx.root.name
    for side in ('left', 'right'):
        manifest['videos'][side]['path'] = str(ctx.root / video_relative[side])
    return manifest, counts
