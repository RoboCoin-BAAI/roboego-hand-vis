"""Optional chronological grids; individual images remain the default."""

import base64

import cv2
import numpy as np

from roboego_hand_vis.qwen_annotation.frame_annotation import build_qwen_frame_annotation_messages


def _grid(paths, times, cell_width):
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read sampled frame: {path}')
        scale = min(1., cell_width / image.shape[1])
        if scale < 1:
            image = cv2.resize(image, (cell_width, max(1, round(image.shape[0] * scale))))
        images.append(image)
    gap, header = 4, 36
    height = max(image.shape[0] for image in images)
    canvas = np.full((height + header, len(images) * (cell_width + gap) - gap, 3), 18, np.uint8)
    for index, (image, timestamp) in enumerate(zip(images, times)):
        x = index * (cell_width + gap)
        label = f't={float(timestamp):.6f}s'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(label, font, .6, 1)[0][0]
        size = .6 * min(1., max(1, cell_width - 8) / text_width)
        cv2.putText(canvas, label, (x + 4, 24), font, size, (0, 230, 255), 1, cv2.LINE_AA)
        offset = x + (cell_width - image.shape[1]) // 2
        canvas[header:header + image.shape[0], offset:offset + image.shape[1]] = image
    success, jpeg = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError('Failed to encode chronological grid')
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()


def build_messages(*, system_prompt, user_prompt, frame_paths, timestamps_s,
                   image_mode, grid_columns, grid_cell_width):
    if len(frame_paths) != len(timestamps_s) or not frame_paths:
        raise ValueError('Each image must have one timestamp')
    if image_mode == 'individual':
        return build_qwen_frame_annotation_messages(system_prompt=system_prompt, user_prompt=user_prompt,
                                                   frame_paths=frame_paths, timestamps_s=timestamps_s)
    if image_mode != 'grid' or grid_columns < 1 or grid_cell_width < 1:
        raise ValueError('Invalid grid image configuration')
    content = [{'type': 'text', 'text': user_prompt + '\nImages are chronological grids: '
                'read tiles left to right, then continue with the next grid. '
                'Each tile is a different video time, not a different camera. '
                'Use its printed timestamp to assign frame labels.'}]
    for start in range(0, len(frame_paths), grid_columns):
        times = timestamps_s[start:start + grid_columns]
        content.append({'type': 'text', 'text': 'Grid timestamps, left to right: ' +
                        ', '.join(f'{float(t):.6f}' for t in times)})
        content.append({'type': 'image_url', 'image_url': {'url': _grid(
            frame_paths[start:start + grid_columns], times, grid_cell_width)}})
    return [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}]
