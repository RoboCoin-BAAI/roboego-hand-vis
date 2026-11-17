"""逐批下载简智深度输入，整批计算完成后进入下一批，GPU 模型跨批常驻。"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue, Empty
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.jianzhi import download_depth
from scripts.L1.gpu_supplement import Worker
from roboego_hand_vis.download_data.base.run import read_batch
from roboego_hand_vis.upload_data.ks3 import atomic_json
from roboego_hand_vis.upload_data.ks3 import prefix
from roboego_hand_vis.lerobot_fast_foundation_depth.base.compute import plan_session
from roboego_hand_vis.lerobot_fast_foundation_depth.base.reuse import existing_valid


def remote_children(client, parent):
    """读取 ks3util -d 输出中的直接子目录，不递归扫描所有文件。"""
    parent = prefix(parent)
    children = set()
    for line in client._run(['ls', '-d', parent]).splitlines():
        uri = line.strip()
        if not uri.startswith(parent) or not uri.endswith('/'):
            continue
        relative = uri[len(parent):-1]
        if relative and '/' not in relative and '\\' not in relative and relative not in ('.', '..'):
            children.add(uri)
    return sorted(children)


def discover_batches(a):
    """按编号逐批查询 KS3，不依赖根目录的一次性枚举结果。"""
    runtime = a.runtime_dir/'inventory'
    runtime.mkdir(parents=True, exist_ok=True)
    client = download_depth.Client(a.ks3_tool.resolve(), a.ks3_config.resolve(), runtime)
    count = 0
    for number in range(a.first_batch, a.last_batch + 1):
        name = f'batch_{number:07d}'
        uri = prefix(a.remote_root) + name + '/'
        try:
            sessions = remote_children(client, uri)
            if not sessions:
                raise ValueError('远端批次没有数据目录: '+uri)
        except Exception as exc:
            atomic_json(a.runtime_dir/'failed'/name/'discovery.json', dict(stage='discovery', error=str(exc)))
            print('BATCH SKIPPED', name, str(exc), flush=True)
            continue
        manifest = runtime/(name+'.csv')
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['目标路径'])
            writer.writerows([[session] for session in sessions])
        count += 1
        (a.runtime_dir/'failed'/name/'discovery.json').unlink(missing_ok=True)
        print('REMOTE BATCH', name, 'datasets', len(sessions), flush=True)
        yield manifest
    print(f'批次范围检查结束: {a.first_batch}–{a.last_batch}，发现非空批次 {count}', flush=True)


def run_batches(a, batches):
    workers = [None] * len(a.devices)
    logs = a.runtime_dir / 'depth_logs'
    logs.mkdir(parents=True, exist_ok=True)
    settings = dict(model_root=str(a.model_root.resolve()), ckpt=str(a.ckpt.resolve()),
                    batch_size=a.batch_size, valid_iters=a.valid_iters, depth_format='hevc12')
    total_failures = 0
    try:
        for batch in batches:
            print('DOWNLOAD BATCH', batch.stem, flush=True)
            result_file = a.runtime_dir/'download_results'/(batch.stem+'.json')
            result_file.unlink(missing_ok=True)
            try:
                result = download_depth.main([
                    '--batch-csv', str(batch), '--output-dir', str(a.source),
                    '--runtime-dir', str(a.runtime_dir/'download'),
                    '--ks3-tool', str(a.ks3_tool), '--ks3-config', str(a.ks3_config),
                    '--remote-root', a.remote_root, '--jobs', str(a.jobs),
                    '--result-file', str(result_file),
                    *(['--dry-run'] if a.dry_run else [])])
            except Exception as exc:
                result = 1
                print('???????????????:', batch.stem, str(exc), flush=True)
                atomic_json(a.runtime_dir/'failed'/batch.stem/'download_batch.json', dict(stage='download', error=str(exc)))
            if result:
                total_failures += 1
                print('本批部分下载失败，跳过失败项，继续处理成功项。', flush=True)
            downloads = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {}
            if a.dry_run:
                continue
            tasks = Queue()
            for uri in read_batch(batch):
                session = a.source / batch.stem / uri.rstrip('/').rsplit('/', 1)[-1]
                receipt = a.runtime_dir/'done'/batch.stem/(session.name+'.json')
                download = downloads.get(session.name)
                if (download and download['status'] != 'downloaded') or (result and download is None):
                    atomic_json(a.runtime_dir/'failed'/batch.stem/(session.name+'.json'),
                                dict(stage='download', session=str(session), error=(download or {}).get('error', '下载未完成')))
                    continue
                try:
                    if receipt.exists():
                        old = json.loads(receipt.read_text(encoding='utf-8'))
                        plans = plan_session(session)
                        if old.get('settings') == settings and plans and all(existing_valid(p, verify_video=False) for p in plans):
                            print('SKIP', session, flush=True)
                            continue
                except Exception as exc:
                    print('已有记录无法复用，继续尝试该条:', session, str(exc), flush=True)
                tasks.put((session, receipt))
            print('DEPTH BATCH', batch.stem, 'pending', tasks.qsize(), flush=True)

            def lane(index):
                failures = 0
                while True:
                    try:
                        session, receipt = tasks.get_nowait()
                    except Empty:
                        return failures
                    try:
                        if workers[index] is None:
                            workers[index] = Worker('depth', a.devices[index], logs)
                        log = logs / batch.stem / (session.name+'.log')
                        log.parent.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, 'depth', '--source', str(session),
                                   '--model-root', str(a.model_root), '--ckpt', str(a.ckpt),
                                   '--engine', 'throughput', '--batch-size', str(a.batch_size),
                                   '--num-frames', 'auto', '--valid-iters', str(a.valid_iters),
                                   '--depth-format', 'hevc12', '--hevc12-preset', a.hevc12_preset,
                                   '--work-root', str(a.runtime_dir/'work'),
                                   '--ffmpeg', a.ffmpeg]
                        if workers[index].execute(command, log):
                            raise RuntimeError('GPU 计算失败，日志: '+str(log))
                        plans = plan_session(session)
                        if not plans or not all(existing_valid(p, verify_video=False) for p in plans):
                            raise RuntimeError('深度发布结果校验失败')
                        atomic_json(receipt, dict(session=str(session), settings=settings))
                        (a.runtime_dir/'failed'/batch.stem/(session.name+'.json')).unlink(missing_ok=True)
                        print('DONE', session, flush=True)
                    except Exception as exc:
                        failures += 1
                        atomic_json(a.runtime_dir/'failed'/batch.stem/(session.name+'.json'),
                                    dict(stage='depth', session=str(session), error=str(exc)))
                        print('FAILED', session, str(exc), flush=True)
                        worker = workers[index]
                        if worker is not None and getattr(worker, 'process', None) is not None and worker.process.poll() is not None:
                            worker.close()
                            workers[index] = None
                return failures

            with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
                failures = sum(pool.map(lane, range(len(a.devices))))
            if failures:
                total_failures += failures
                print('本批失败项已记录，继续下一批。失败条数:', failures, flush=True)
            print('BATCH DONE', batch.stem, flush=True)
        print('全部批次处理结束；失败汇总:', total_failures, flush=True)
        return int(bool(total_failures))
    finally:
        for worker in workers:
            if worker is not None:
                worker.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, help='可选旧 CSV 模式；默认直接枚举 KS3 batch/ID/')
    p.add_argument('--source', type=Path, required=True, help='下载根目录，存放 batch/ID/')
    p.add_argument('--runtime-dir', type=Path, required=True, help='数据目录外的日志和完成记录')
    p.add_argument('--ks3-tool', type=Path, required=True)
    p.add_argument('--ks3-config', type=Path, required=True)
    p.add_argument('--remote-root', default='ks3://baai-eai-pwm/jianzhi/wait_for_publish/')
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--valid-iters', type=int, default=4)
    p.add_argument('--hevc12-preset', default='fast',
                   choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
                   help='首次深度编码预设；已完成结果继续复用')
    p.add_argument('--jobs', type=int, default=64)
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--first-batch', type=int, default=1)
    p.add_argument('--last-batch', type=int, default=200, help='按编号检查至此批次，默认 200（含）')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    a.devices = [d.strip() for d in a.gpus.split(',')]
    if (any(not d.isdigit() for d in a.devices) or len(set(a.devices)) != len(a.devices)
            or min(a.jobs, a.batch_size, a.valid_iters) < 1
            or not 1 <= a.first_batch <= a.last_batch):
        p.error('GPU 或数值参数无效')
    a.source, a.runtime_dir = a.source.resolve(), a.runtime_dir.resolve()
    if a.runtime_dir.is_relative_to(a.source):
        p.error('runtime-dir 必须在 source 之外')
    if a.batch_dir:
        batches = [b for b in sorted(a.batch_dir.glob('batch_*.csv'))
                   if b.stem[6:].isdigit() and a.first_batch <= int(b.stem[6:]) <= a.last_batch]
        if not batches:
            p.error('没有选中批次 CSV')
    else:
        batches = discover_batches(a)
    if not a.dry_run:
        for path in (a.model_root, a.ckpt, a.ckpt.with_name('cfg.yaml')):
            if not path.exists():
                p.error('模型路径不存在: '+str(path))
    a.runtime_dir.mkdir(parents=True, exist_ok=True)
    lock = a.runtime_dir/'depth_batches.lock'
    try:
        with lock.open('x'):
            pass
    except FileExistsError:
        p.error('已有运行锁；确认旧进程退出后再移除 '+str(lock))
    try:
        return run_batches(a, batches)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
