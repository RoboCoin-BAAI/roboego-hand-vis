"""File-bound ffprobe receipts shared with the L1 converter (schema v1)."""
import hashlib
import json
from uuid import uuid4


def identity(video, timestamps):
    stat = video.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                timestamps_sha256=hashlib.sha256(timestamps.read_bytes()).hexdigest())


def cache_path(ctx, episode, camera):
    return ctx.root / 'annotation/video_validation' / f'{episode.parent.name}_{episode.name}_{camera}.json'


def read_receipt(ctx, episode, camera, fingerprint):
    relative = ctx.video(episode, camera).relative_to(ctx.root).as_posix()
    for path in (cache_path(ctx, episode, camera), ctx.root / 'meta/l1_video_validation.json'):
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            entry = receipt['videos'][relative]
            stream = entry['stream']
            if (receipt['version'] == 1 and entry['identity'] == fingerprint
                    and entry['decoder_errors'] == '' and int(stream['nb_read_frames']) > 0
                    and all(k in stream for k in ('width', 'height', 'avg_frame_rate'))):
                return stream, receipt.get('producer', 'unknown')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, None


def write_receipt(ctx, episode, camera, fingerprint, stream):
    path = cache_path(ctx, episode, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = ctx.video(episode, camera).relative_to(ctx.root).as_posix()
    receipt = dict(version=1, producer='lerobot_qc', videos={relative: dict(
        identity=fingerprint, stream=stream, decoder_errors='')})
    temporary = path.with_name('.' + path.name + '-' + uuid4().hex)
    temporary.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    temporary.replace(path)
