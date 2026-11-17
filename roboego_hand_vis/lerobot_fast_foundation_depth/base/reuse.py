"""Conservative completeness check for full-session depth reuse."""
import json
from pathlib import Path
import subprocess


def existing_valid(plan, ffprobe='ffprobe', *, verify_video=True):
    """Check metadata; optionally decode the video to verify actual frame count."""
    import pyarrow.parquet as pq
    try:
        episode=Path(plan['episode'])
        output=Path(plan['output'])
        report=json.loads((output/'lerobot_depth.json').read_text(encoding='utf-8'))
        legacy_manifest = report.get('schema_version') == 2
        session_layout = report.get('depth_storage', {}).get('path_base') == 'session'
        feature_layout = legacy_manifest or (bool(report.get('depth_feature')) and not session_layout)
        if legacy_manifest:
            report = report['episodes'][episode.name]
        if report.get('right_frame_offset',0)!=0:
            return False
        timestamp_root = episode.resolve().parents[2] if session_layout else output
        timestamp_path = timestamp_root/report['timestamp_file'] if (session_layout or feature_layout) else output/'source_timestamps.parquet'
        saved=pq.read_table(timestamp_path)
        if feature_layout:
            import pyarrow.compute as pc
            saved = saved.filter(pc.equal(saved['episode_index'], int(episode.name.removeprefix('episode_'))))
        candidates=[p for p in episode.glob('observation.images.*left*rectified.parquet')
                    if not p.name.startswith('observation.images.depth_')]
        if len(candidates)!=1 or not saved.equals(pq.read_table(candidates[0])):
            return False
        count=saved.num_rows
        if count<1 or report.get('processed_frames')!=list(range(count)):
            return False
        storage=report.get('depth_storage',{})
        if storage.get('file'):
            root = episode.resolve().parents[2] if session_layout else output.resolve()
            video=root/storage['file']
            if not video.resolve().is_relative_to(root) or not video.is_file():
                return False
            mask = None
            if storage.get('requires_valid_mask'):
                mask = root/storage['mask_file']
                if not mask.resolve().is_relative_to(root) or not mask.is_file() or mask.stat().st_size == 0:
                    return False
            if verify_video and mask is not None:
                checked = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                    '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(mask)],
                    capture_output=True, text=True, check=True)
                stream, = json.loads(checked.stdout)['streams']
                if (checked.stderr.strip() or int(stream['nb_read_frames']) != count
                        or stream['width'] != storage['width'] or stream['height'] != storage['height']):
                    return False
            if not verify_video:
                return (video.stat().st_size > 0 and storage.get('frames') == count
                        and storage.get('width', 0) > 0 and storage.get('height', 0) > 0)
            run=subprocess.run([ffprobe,'-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(video)],
                capture_output=True,text=True,check=True)
            stream,=json.loads(run.stdout)['streams']
            return (not run.stderr.strip() and int(stream['nb_read_frames'])==count
                    and stream['width']==storage['width'] and stream['height']==storage['height'])
        from roboego_hand_vis.common.depth_io import depth_path_for_frame,depth_uint16_png_path_for_frame
        if feature_layout:
            output = output/report['frame_files_directory']
        return all(any(p.is_file() and p.stat().st_size>0 for p in
            (depth_path_for_frame(output,i),depth_uint16_png_path_for_frame(output,i))) for i in range(count))
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError):
        return False
