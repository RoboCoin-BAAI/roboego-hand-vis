"""Publish compressed depth and its exact validity mask as one transaction."""
import json
import os
from pathlib import Path
import shutil
import tempfile

from .metadata import stage_video_info


def publish_video(work, output, session, episode, camera_feature, *, offset=0):
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import publish_files

    work, output, session, episode = map(Path, (work, output, session, episode))
    session = session.resolve()
    feature = camera_feature.replace('observation.images.', 'observation.images.depth_', 1)
    feature = feature.replace('_rgb_rectified', '_rectified')
    if offset:
        feature += f'_right_offset_{"m" if offset < 0 else "p"}{abs(offset)}'
    video = Path('videos') / episode.parent.name / feature / (episode.name + '.mp4')
    mask = output.resolve().relative_to(session) / 'valid_mask_ffv1.mkv'
    report = json.loads((work / 'lerobot_depth.json').read_text(encoding='utf-8'))
    report['depth_storage'].update(file=video.as_posix(), mask_file=mask.as_posix(),
                                   path_base='session')
    report['depth_feature'] = feature
    timestamp = episode.resolve().relative_to(session) / (feature + '.parquet')
    report['timestamp_file'] = timestamp.as_posix()
    report['timestamp_lookup'] = 'session-relative timestamp_file, rows indexed by original source frame number'
    if report.get('timestamp_index') == 'video_frame':
        report['timestamp_lookup'] = 'session-relative timestamp_file, rows indexed by depth video frame; processed_frames maps to original source frames'
    report['session_root_relative'] = os.path.relpath(session, output).replace('\\', '/')
    relative_output = output.resolve().relative_to(session)
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_root = work.parent.resolve()
    if publish_root.is_relative_to(session) or session.is_relative_to(publish_root):
        publish_root = Path(tempfile.gettempdir())/'roboego-depth-publish'
        publish_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.depth-publish-', dir=publish_root) as tmp:
        stage = Path(tmp)
        for source, target in ((work / 'depth_hevc12.mp4', stage / video),
                               (work / 'valid_mask_ffv1.mkv', stage / mask)):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (stage / timestamp).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / 'source_timestamps.parquet', stage / timestamp)
        metadata = stage / relative_output
        metadata.mkdir(parents=True, exist_ok=True)
        for source in work.iterdir():
            if source.name.endswith('.log'):
                continue
            if source.name in ('depth_hevc12.mp4', 'valid_mask_ffv1.mkv', 'lerobot_depth.json', 'source_timestamps.parquet'):
                continue
            if source.is_dir():
                shutil.copytree(source, metadata / source.name)
            else:
                shutil.copy2(source, metadata / source.name)
        (metadata / 'lerobot_depth.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        stage_video_info(session, stage, report)
        publish_files(stage, session, staging_base=publish_root)
    shutil.rmtree(work)
