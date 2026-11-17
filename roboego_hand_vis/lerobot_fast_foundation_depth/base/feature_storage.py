"""Publish a single depth episode in a chunk/feature directory."""
import json
from pathlib import Path
import shutil
import tempfile


def depth_feature(camera_names):
    left = (camera_names or {}).get('left', '')
    if left == 'headband_left_rgb_rectified':
        return 'observation.images.depth_headband_left_rectified'
    return None


def publish_feature(work, output, episode, feature):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import publish_files

    work, output, episode = map(Path, (work, output, episode))
    if output.is_symlink():
        raise ValueError('Depth feature output cannot be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    index = int(episode.name.removeprefix('episode_'))
    path = output / 'lerobot_depth.json'
    previous_report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if previous_report.get('schema_version') == 2:
        if set(previous_report.get('episodes', {})) != {episode.name}:
            raise ValueError('Single-episode output cannot replace a multiple/different episode manifest')
    for sibling in output.glob('episode_*.mkv'):
        if sibling.name != episode.name + '.mkv':
            raise ValueError('Single-episode output contains another episode video')
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=output) as tmp:
        stage = Path(tmp)
        report = json.loads((work / 'lerobot_depth.json').read_text(encoding='utf-8'))
        storage = report.get('depth_storage', {})
        if storage.get('file'):
            filename = episode.name + '.mkv'
            shutil.move(str(work / storage['file']), stage / filename)
            storage['file'] = filename
        else:
            # Per-frame output stays isolated by episode within the feature.
            for folder in ('depth', 'depth_uint16_png'):
                if (work / folder).exists():
                    shutil.copytree(work / folder, stage / episode.name / folder)
            for path in work.glob('frame_*.npy'):
                (stage / episode.name).mkdir(exist_ok=True)
                shutil.copy2(path, stage / episode.name / path.name)
        for folder in ('vis',):
            if (work / folder).exists():
                shutil.copytree(work / folder, stage / episode.name / folder)
        metadata = work / 'fast_foundation_stereo_video_meta.json'
        if metadata.exists():
            shutil.copy2(metadata, stage / (episode.name + '_meta.json'))
        timestamps = pq.read_table(work / 'source_timestamps.parquet')
        if 'episode_index' not in timestamps.column_names:
            timestamps = timestamps.append_column('episode_index', pa.array([index]*timestamps.num_rows, type=pa.int64()))
        if not pc.all(pc.equal(timestamps['episode_index'], index)).as_py():
            raise ValueError('Depth timestamp episode_index does not match output episode')
        timestamp_name = feature + '.parquet'
        if (output / timestamp_name).exists():
            previous = pq.read_table(output / timestamp_name)
            if not pc.all(pc.equal(previous['episode_index'], index)).as_py():
                raise ValueError('Single-episode output contains timestamps from another episode')
        pq.write_table(timestamps, stage / timestamp_name)
        report['timestamp_file'] = timestamp_name
        report['timestamp_lookup'] = 'filter episode_index, then index rows by original source frame number'
        if not storage.get('file'):
            report['frame_files_directory'] = episode.name
        report['depth_feature'] = feature
        (stage / 'lerobot_depth.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        publish_files(stage, output)
    shutil.rmtree(work)
    return output
