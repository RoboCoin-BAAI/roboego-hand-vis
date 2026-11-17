"""Register published depth videos in LeRobot session metadata."""
import json


def video_feature_definition(storage):
    return {
        'dtype': 'video',
        'shape': [storage['height'], storage['width'], 1],
        'names': ['h', 'w', 'c'],
        'fps': storage['fps'],
        'info': {
            'video.width': storage['width'],
            'video.height': storage['height'],
            'video.codec': storage['codec'],
            'video.pix_fmt': storage['pixel_format'],
            'video.fps': storage['fps'],
            'video.channels': 1,
            'video.is_depth_map': True,
            'has_audio': False,
        },
        'depth_storage': {
            key: storage[key] for key in (
                'container', 'codec', 'pixel_format', 'unit', 'invalid_value',
                'max_depth_m', 'lossless', 'requires_valid_mask',
            ) if key in storage
        },
    }


def stage_video_info(session, stage, report):
    """Stage info alongside video assets for transactional publication."""
    path = session / 'meta/info.json'
    if not path.is_file():
        return  # Standalone video exports may have no LeRobot metadata.
    info = json.loads(path.read_text(encoding='utf-8-sig'))
    features = info.setdefault('features', {})
    feature = features.setdefault(report['depth_feature'], {})
    feature.update(video_feature_definition(report['depth_storage']))
    for key in ('depth_metadata_path', 'mask_path', 'timestamp_index', 'frame_mapping'):
        feature.pop(key, None)
    videos = set()
    for root in (session, stage):
        for video in (root / 'videos').glob('chunk-*/*/episode_*.mp4'):
            if features.get(video.parent.name, {}).get('dtype') == 'video':
                videos.add(video.relative_to(root).as_posix())
    info['total_videos'] = len(videos)
    target = stage / 'meta/info.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
