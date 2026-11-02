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
        if report.get('right_frame_offset',0)!=0:
            return False
        saved=pq.read_table(output/'source_timestamps.parquet')
        candidates=list(episode.glob('observation.images.*left*rectified.parquet'))
        if len(candidates)!=1 or not saved.equals(pq.read_table(candidates[0])):
            return False
        count=saved.num_rows
        if count<1 or report.get('processed_frames')!=list(range(count)):
            return False
        storage=report.get('depth_storage',{})
        if storage.get('file'):
            video=output/storage['file']
            if not video.resolve().is_relative_to(output.resolve()) or not video.is_file():
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
        return all(any(p.is_file() and p.stat().st_size>0 for p in
            (depth_path_for_frame(output,i),depth_uint16_png_path_for_frame(output,i))) for i in range(count))
    except (OSError,ValueError,KeyError,TypeError,subprocess.CalledProcessError):
        return False
