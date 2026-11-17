"""按简智批次 CSV 下载深度所需的矫正双目视频及全部非视频目录。"""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from roboego_hand_vis.download_data.base.run import read_batch
from roboego_hand_vis.upload_data.ks3 import KS3, parse_listing, prefix

CAMERAS = {'observation.images.head_left_rectified', 'observation.images.head_right_rectified'}


def select_objects(objects, source):
    source = prefix(source)
    selected = []
    pairs = {}
    for item in objects:
        if not item['uri'].startswith(source):
            raise ValueError('远端对象不在数据目录内: ' + item['uri'])
        name = item['uri'][len(source):]
        if not name or name.endswith('/'):
            continue
        parts = PurePosixPath(name).parts
        if '\\' in name or any(p in ('.', '..', '') for p in name.split('/')):
            raise ValueError('不安全的远端路径: ' + name)
        if parts[0] in ('logs', 'ks3util_output', '.ks3util_checkpoint') or any(p.startswith('.') for p in parts):
            continue
        if parts[0] == 'videos':
            if len(parts) != 4 or parts[2] not in CAMERAS or not parts[3].endswith('.mp4'):
                continue
            pairs.setdefault((parts[1], parts[3]), set()).add(parts[2])
        selected.append(dict(item, relative=name, source_uri=item['uri']))
    if not pairs or any(cameras != CAMERAS for cameras in pairs.values()):
        raise ValueError('远端缺少完整的矫正左右目视频: ' + source)
    if not any(item['relative'] == 'meta/info.json' for item in selected):
        raise ValueError('远端缺少 meta/info.json: ' + source)
    return selected


class Client(KS3):
    def __init__(self, tool, config, runtime):
        super().__init__(tool, config)
        self.runtime = runtime

    def _run(self, arguments):
        # 工具日志、断点和清单均留在独立运行目录。
        command = [self.tool, *map(str, arguments), '-c', self.config, '--loglevel=off']
        if arguments[0] == 'cp':
            command += ['--checkpoint-dir=' + str(self.runtime/'checkpoints'),
                        '--output-dir=' + str(self.runtime/'output')]
        result = subprocess.run(command,
                                cwd=self.runtime, capture_output=True, text=True,
                                encoding='utf-8', errors='replace')
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-4000:])
        return result.stdout


def copy_selected(client, source, selected, target, jobs):
    """按目录调用 ks3util cp，传输并发交给 ks3util 自身。"""
    groups = set()
    for item in selected:
        parts = PurePosixPath(item['relative']).parts
        if parts[0] == 'videos':
            groups.add(('/'.join(parts[:3]), True))
        elif len(parts) > 1:
            groups.add((parts[0], True))
        else:
            groups.add((parts[0], False))
    for relative, directory in sorted(groups):
        destination = target.joinpath(*PurePosixPath(relative).parts)
        (destination if directory else destination.parent).mkdir(parents=True, exist_ok=True)
        client._run(['cp', prefix(source)+relative+('/' if directory else ''), str(destination),
                     *(['-r'] if directory else []), '-u', '--disable-temp-file', '--read-timeout', '6', f'--jobs={jobs}'])
    for item in selected:
        path = target.joinpath(*PurePosixPath(item['relative']).parts)
        if not path.is_file() or path.stat().st_size != item['size']:
            raise RuntimeError('下载后文件缺失或大小不符: '+str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--batch-dir', type=Path)
    inputs.add_argument('--batch-csv', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--ks3-tool', type=Path, required=True)
    p.add_argument('--ks3-config', type=Path, required=True)
    p.add_argument('--remote-root', default='ks3://baai-eai-pwm/jianzhi/wait_for_publish/',
                   help='已处理数据前缀，按 batch/ID/ 下载；CSV 仅提供批次和 ID')
    p.add_argument('--jobs', type=int, default=64)
    p.add_argument('--dry-run', action='store_true', help='只列远端清单并统计，不下载数据')
    p.add_argument('--result-file', type=Path, help='在外部运行目录记录各条下载状态')
    a = p.parse_args(argv)
    if a.jobs < 1:
        p.error('jobs 必须为正数')
    output, runtime = a.output_dir.resolve(), a.runtime_dir.resolve()
    if runtime.is_relative_to(output):
        p.error('runtime-dir 必须在 output-dir 之外')
    batches = [a.batch_csv] if a.batch_csv else sorted(a.batch_dir.glob('batch_*.csv'))
    if not batches:
        p.error('没有找到批次 CSV')
    runtime.mkdir(parents=True, exist_ok=True)
    client = Client(a.ks3_tool.resolve(), a.ks3_config.resolve(), runtime)
    failed = 0
    results = {}
    for batch in batches:
        for original in read_batch(batch):
            name = original.rstrip('/').rsplit('/', 1)[-1]
            source = prefix(a.remote_root) + batch.stem + '/' + name + '/' if a.remote_root else original
            try:
                objects = parse_listing(client._run(['ls', source]))
                selected = select_objects(objects, source)
                target = output / batch.stem / name
                print(json.dumps(dict(source=source, target=str(target), files=len(selected),
                                      selected_bytes=sum(i['size'] for i in selected),
                                      skipped_bytes=sum(i['size'] for i in objects)-sum(i['size'] for i in selected)),
                                 ensure_ascii=False), flush=True)
                if a.dry_run:
                    continue
                copy_selected(client, source, selected, target, a.jobs)
                results[name] = dict(status='downloaded')
            except Exception as exc:
                failed += 1
                results[name] = dict(status='failed', error=str(exc))
                print(f'下载失败 {source}: {exc}', file=sys.stderr, flush=True)
    if a.result_file:
        from roboego_hand_vis.upload_data.ks3 import atomic_json
        if a.result_file.resolve().is_relative_to(output):
            raise ValueError('result-file 必须在数据目录外')
        atomic_json(a.result_file, results)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
