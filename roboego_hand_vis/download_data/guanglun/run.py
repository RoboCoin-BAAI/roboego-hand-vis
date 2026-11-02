"""Freeze a remote inventory, then process batches using node-local storage."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .inventory import SOURCE, DESTINATION, plan, write_plan
from roboego_hand_vis.upload_data.ks3 import KS3, atomic_json


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, value, status, **extra):
    if status not in ('failed', 'rejected', 'conversion_failed'):
        value.pop('error', None)
    value.update(status=status, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    atomic_json(path, value)


def cleanup(work, local_root):
    work, local_root = work.resolve(), local_root.resolve()
    if work == local_root or not work.is_relative_to(local_root) or len(work.relative_to(local_root).parts) != 2:
        raise ValueError("Unsafe cleanup path: " + str(work))
    if work.exists():
        shutil.rmtree(work)


def process(item, batch, options):
    state = Path(options['state_root'])/batch/'records'/(item['id']+'.json')
    work = Path(options['local_root'])/batch/item['id']
    record = read(state, dict(item))
    try:
        # Migrate the known legacy data error without downloading it again.
        if record.get('status') == 'failed' and 'ValueError: Invalid semantic interval:' in record.get('error', ''):
            save(state, record, 'conversion_failed', failure_reason='invalid_semantic_interval', local_cleaned=False)
        if record.get('status') in ('uploaded', 'rejected', 'conversion_failed'):
            status = 'uploaded' if record['status'] == 'uploaded' else 'conversion_failed'
            if record['status'] == 'rejected':
                record['failure_reason'] = 'pose_quality_rejected'
            cleanup(work, Path(options['local_root']))
            save(state, record, status, local_cleaned=True)
            return
        client = KS3(options['ks3_tool'], options['ks3_config'])
        source, output = work/'source.mcap', work/'converted'
        client.download(item, source)
        save(state, record, 'downloaded')
        if not (output/'meta/info.json').is_file():
            work.mkdir(parents=True, exist_ok=True)
            with (work/'conversion.log').open('w', encoding='utf-8') as log:
                result = subprocess.run([options['python'], str(Path(options['converter_repo'])/'format_convert/guanglun/convert_guanglun_mcap.py'),
                    '--source', str(source), '--out', str(output), '--ffmpeg', options['ffmpeg'],
                    '--head-jump-m', str(options.get('head_jump_m', .4)),
                    '--hand-jump-m', str(options.get('hand_jump_m', .6)),
                    '--head-jump-deg', str(options.get('head_jump_deg', 120.)),
                    '--hand-jump-deg', str(options.get('hand_jump_deg', 160.))],
                    cwd=options['converter_repo'], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                # Keep diagnostics on the shared disk, including pose-QC rejection.
                details = (work/'conversion.log').read_text(encoding='utf-8', errors='replace')
                reason = ('pose_quality_rejected' if result.returncode == 2 else
                          'invalid_semantic_interval' if 'ValueError: Invalid semantic interval:' in details else
                          'converter_error')
                save(state, record, 'conversion_failed', failure_reason=reason,
                     converter_returncode=result.returncode, error=details[-16000:], local_cleaned=False)
                cleanup(work, Path(options['local_root']))
                save(state, record, 'conversion_failed', local_cleaned=True)
                return
        try:
            metadata = read(output/'meta/guanglun_source.json')
            if not isinstance(metadata, dict) or metadata.get('rectification_status') != 'native_rectified':
                raise ValueError('Missing native rectification confirmation')
        except (ValueError, TypeError, OSError) as exc:
            save(state, record, 'conversion_failed', failure_reason='invalid_conversion_output',
                 error=str(exc), local_cleaned=False)
            cleanup(work, Path(options['local_root']))
            save(state, record, 'conversion_failed', local_cleaned=True)
            return
        save(state, record, 'converted', rectification_status='native_rectified')
        receipt = client.upload_directory(output, item['destination_uri'])
        # Durable receipt precedes deletion. A restart retries cleanup only.
        save(state, record, 'uploaded', upload=receipt, local_cleaned=False)
        cleanup(work, Path(options['local_root']))
        save(state, record, 'uploaded', local_cleaned=True)
    except Exception as exc:
        if record.get('status') in ('uploaded', 'conversion_failed', 'rejected'):
            save(state, record, record['status'], local_cleaned=False, cleanup_error=str(exc))
        else:
            save(state, record, 'failed', error=str(exc))


def summarize(root, batch, items):
    records = [read(root/batch/'records'/(item['id']+'.json'), dict(item, status='pending')) for item in items]
    atomic_json(root/batch/'progress.json', records)
    # Same string-array shape as L1; these strings are downloadable KS3 prefixes.
    atomic_json(root/batch/'rectification_done.json', sorted(r['destination_uri'] for r in records if r['status'] == 'uploaded'))
    return records


def batch_complete(records):
    return all(r['status'] in ('uploaded', 'conversion_failed') and r.get('local_cleaned') for r in records)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inventory', 'run'])
    p.add_argument('--state-root', type=Path, default=Path('/share/guanglun_batches'))
    p.add_argument('--local-root', type=Path, default=Path('/home/guanglun_work'))
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--destination', default=DESTINATION)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--ks3-tool', required=True)
    p.add_argument('--ks3-config', required=True)
    p.add_argument('--converter-repo', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--head-jump-m', type=float, default=.4)
    p.add_argument('--hand-jump-m', type=float, default=.6)
    p.add_argument('--head-jump-deg', type=float, default=120.)
    p.add_argument('--hand-jump-deg', type=float, default=160.)
    p.add_argument('--batches', nargs='+')
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error('--workers must be positive')
    a.state_root = a.state_root.resolve()
    a.state_root.mkdir(parents=True, exist_ok=True)
    # Single coordinator owns the shared summaries. Stale lock requires inspection.
    lock = a.state_root/'runner.lock'
    try:
        fd = lock.open('x')
    except FileExistsError:
        p.error('runner.lock exists; another coordinator may be running')
    try:
        fd.close()
        if a.action == 'inventory':
            objects = KS3(a.ks3_tool, a.ks3_config).objects(a.source)
            write_plan(a.state_root, plan(objects, a.source, a.destination, a.batch_size))
            return 0
        if not a.converter_repo or not (a.converter_repo/'format_convert/guanglun/convert_guanglun_mcap.py').is_file():
            p.error('--converter-repo must point to Multi-Frequency-LeRobot')
        a.converter_repo = a.converter_repo.resolve()
        a.local_root = a.local_root.resolve()
        if a.local_root.is_relative_to(a.state_root) or a.state_root.is_relative_to(a.local_root):
            p.error('local-root and state-root must be separate')
        inventory = read(a.state_root/'inventory.json')
        if not inventory:
            p.error('Run inventory first')
        batches = a.batches or list(inventory['batches'])
        if any(b not in inventory['batches'] for b in batches):
            p.error('Unknown batch')
        failures = False
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for batch in batches:
                items = inventory['batches'][batch]
                summarize(a.state_root, batch, items)
                futures = [pool.submit(process, item, batch, vars(a)) for item in items]
                for future in as_completed(futures):
                    future.result()
                    summarize(a.state_root, batch, items)
                records = summarize(a.state_root, batch, items)
                if not batch_complete(records):
                    failures = True
                    # Bound node-local disk use when conversion or upload fails.
                    break
        return int(failures)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
