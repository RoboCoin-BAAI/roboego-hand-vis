"""Decode-count MP4s with ffprobe, without extracting image files."""
from fractions import Fraction
import json
import subprocess
from ..common import CAMERAS, guarded, result
from .video_cache import identity, read_receipt, write_receipt


def check_camera(ctx, episode, camera):
    path = ctx.video(episode, camera)
    timestamps = episode / f'observation.images.{camera}.parquet'
    before = identity(path, timestamps)
    stream, producer = (None, None) if ctx.full_video_check else read_receipt(ctx, episode, camera, before)
    errors = ''
    if stream is None:
        run = subprocess.run([ctx.ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(path)],
            capture_output=True, text=True, check=True)
        stream, = json.loads(run.stdout)["streams"]
        errors = run.stderr.strip()
        if before != identity(path, timestamps):
            raise ValueError('Video or timestamps changed during validation')
        if not errors and int(stream['nb_read_frames']) > 0:
            write_receipt(ctx, episode, camera, before, stream)
    frames = int(stream["nb_read_frames"])
    shape = ctx.info["features"][f"observation.images.{camera}"]["shape"]
    fps = float(Fraction(stream["avg_frame_rate"]))
    checks = [result("decodable", frames > 0 and not errors,
                     decoded_frames=frames, decoder_errors=errors,
                     validation_source=producer or 'ffprobe', reused_validation=producer is not None),
        result("dimensions", [stream["height"], stream["width"], 3] == list(shape)),
        result("fps", abs(fps - ctx.fps(camera)) <= 1, level="warning", actual_fps=fps)]
    checks.append(result("timestamp_count", frames == len(ctx.times(episode, camera))))
    checks.append(result("master_count", frames == len(ctx.table(episode, "master_index")["timestamp"])))
    return checks


def check(ctx):
    checks = []
    for episode in ctx.episodes:
        for camera in CAMERAS:
            prefix = f"{episode.parent.name}/{episode.name}/{camera}"
            items = guarded("readable", lambda: check_camera(ctx, episode, camera))
            for item in items:
                item["name"] = f"{prefix}/{item['name']}"
            checks.extend(items)
    return checks
