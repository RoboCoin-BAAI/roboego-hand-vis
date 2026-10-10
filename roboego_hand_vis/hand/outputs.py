"""Final delivery publication and artifact checks; no algorithm dependencies at import."""

import argparse
import json
import os
from pathlib import Path
import shutil


DELIVERY_NAMES = ('hand_tracking.parquet', 'render.mp4')


def video_frames(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot read video: {path}')
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError(f'Video contains no frames: {path}')
        return count
    finally:
        capture.release()


def validate_mint(path, video):
    import numpy as np

    count = video_frames(video)
    with np.load(path, allow_pickle=False) as archive:
        for key, width in (('pose_enc', 9), ('hand', 218), ('hand_presence_logits', 2)):
            if archive[key].shape != (count, width):
                raise ValueError(f'MINT {key} shape {archive[key].shape}; expected {(count, width)}')
    return {'frame_count': count}


def validate_final(final, video):
    import pyarrow.parquet as pq

    expected = video_frames(video)
    # Iterate row groups so damaged data pages are detected without loading all rows at once.
    count = sum(batch.num_rows for batch in pq.ParquetFile(final / DELIVERY_NAMES[0]).iter_batches())
    rendered = video_frames(final / DELIVERY_NAMES[1])
    if count != expected or rendered != expected:
        raise ValueError(f'Frame mismatch: input={expected}, parquet={count}, render={rendered}')
    summary = final / 'summary.json'
    return {'input_frames': expected, 'parquet_rows': count, 'render_frames': rendered,
            'summary': json.loads(summary.read_text()) if summary.is_file() else {},
            'quality_status': 'not_validated'}


def publish(final, output):
    from roboego_hand_vis.hand.state import checked_path, digest_file, sync_directory

    hashes = {}
    # Keep the source copies until BOTH public files have been installed and checkpointed.
    for name in DELIVERY_NAMES:
        source = final / name
        target = checked_path(output, name)
        temporary = checked_path(output, f'.{name}.publishing')
        if target.exists() and digest_file(target) != digest_file(source):
            raise FileExistsError(f'Refusing to overwrite different delivery file: {target}')
        with source.open('rb') as reader, temporary.open('wb') as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        hashes[name] = digest_file(temporary)
        if hashes[name] != digest_file(source):
            raise RuntimeError(f'Published copy verification failed: {name}')
        os.replace(temporary, target)
    sync_directory(output)
    return hashes


def verify_delivery(output, hashes):
    from roboego_hand_vis.hand.state import checked_path, digest_file

    if set(hashes) != set(DELIVERY_NAMES):
        raise ValueError('Incomplete delivery checkpoint')
    for name, expected in hashes.items():
        path = checked_path(output, name)
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f'Delivery changed or missing: {path}; intermediates retained')


def cleanup_work(output):
    from roboego_hand_vis.hand.state import checked_path, sync_directory

    work = checked_path(output, '.work')
    # Leave state.json until last so a failed/interrupted cleanup remains resumable.
    for child in work.iterdir():
        if child.name == 'state.json':
            continue
        checked_path(output, child.relative_to(output))
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    checked_path(output, '.work/state.json').unlink()
    work.rmdir()
    sync_directory(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('mint', 'final'))
    parser.add_argument('artifact', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('result', type=Path)
    args = parser.parse_args()
    result = (validate_mint if args.kind == 'mint' else validate_final)(args.artifact, args.video)
    args.result.write_text(json.dumps(result))
    print('Artifact validation: ' + json.dumps(result), flush=True)
