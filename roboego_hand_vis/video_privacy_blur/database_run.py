"""Independent L1 queue entry reusing the resumable face-blur batch."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import signal
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

from roboego_hand_vis.database.validators import resolve_session
from . import batch
from .stage_cli import DEFAULT_CONFIG, build_batch_command, config_from_yaml, load_yaml


def _config(data, session):
    args=argparse.Namespace(input_dir=str(session/'videos'), output_dir=str(session/'privacy/video_privacy_blur'),
                            video=None, dry_run=False)
    return config_from_yaml(data,args)


def _validate(shared_root, data, limit):
    root=Path(shared_root).expanduser().resolve()
    if limit < 0 or not root.is_dir():
        raise ValueError('Need a valid shared root and nonnegative limit')
    data=copy.deepcopy(data)
    config=_config(data,root)
    if config.dry_run:
        raise ValueError('Database mode does not allow dry_run')
    weights=config.face_weights.expanduser().resolve()
    if not weights.is_file():
        raise ValueError(f'Face weights not found: {weights}')
    data.setdefault('paths',{})['face_weights']=str(weights)
    python=shutil.which(config.python)
    if not python or Path(python).resolve()!=Path(sys.executable).resolve():
        raise ValueError('Database mode requires the current Python; run this command in the desired environment')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('ffmpeg and ffprobe are required')
    if config.device not in ('auto','cpu','cuda','mps') or config.method not in ('gaussian','pixelate','egoblur'):
        raise ValueError('Unsupported device or blur method')
    if (config.threads < 1 or config.retries < 0 or config.imgsz <= 0 or config.scale < 1
            or not .25 <= config.conf <= 1 or config.blur_strength <= 0
            or (config.jobs != 'auto' and (not config.jobs.isdigit() or int(config.jobs) < 1))):
        raise ValueError('Invalid privacy blur parameters')
    videos=data.get('parameters',{}).get('videos',[])
    if videos is not None and not isinstance(videos,list):
        raise ValueError('parameters.videos must be a list of relative filenames')
    return root,data


def _inside(path, session):
    if not path.resolve().is_relative_to(session):
        raise ValueError('Privacy paths must remain inside their session')
    return path


def _task(data, session):
    config=_config(data,session)
    _inside(config.input_dir,session)
    _inside(config.output_dir,session)
    names=(list(config.selected_videos) if config.selected_videos else
           [str(p.relative_to(config.input_dir)) for p in sorted(config.input_dir.rglob('*'))
            if p.is_file() and p.suffix.lower() in batch.VIDEO_SUFFIXES])
    sources=batch.select_videos(config.input_dir,names)
    outputs=[config.output_dir/p.relative_to(config.input_dir).with_name(p.stem+'_blurred.mp4') for p in sources]
    if len(set(outputs))!=len(outputs):
        raise ValueError('Multiple input videos map to the same blurred output')
    for path in outputs:
        _inside(path,session)
    state=_inside(config.output_dir/'.batch',session)
    if state.is_symlink() or any(p.is_symlink() for p in state.rglob('*')):
        raise ValueError('Batch state directory must not contain symbolic links')
    # Freeze this session's selection so newly added files cannot change its workload.
    return replace(config,selected_videos=tuple(names)),sources


def _run_session(config):
    config.output_dir.mkdir(parents=True,exist_ok=True)
    log=config.output_dir/'database_run.log'
    _inside(log,config.output_dir)
    previous=signal.getsignal(signal.SIGTERM)
    try:
        with log.open('w',encoding='utf-8') as handle, redirect_stdout(handle), redirect_stderr(handle):
            code=batch.main(build_batch_command(config)[3:])
    finally:
        signal.signal(signal.SIGTERM,previous)
    if code == 130:
        raise KeyboardInterrupt
    if code:
        raise RuntimeError(f'Privacy batch exited with code {code}; see {log}')


def _check_outputs(config, sources):
    state=config.output_dir/'.batch'
    summary=json.loads((state/'summary.json').read_text())
    counts=summary.get('counts',{})
    if (counts.get('failed')!=0 or summary.get('failures')
            or counts.get('completed',0)+counts.get('skipped',0)!=len(sources)):
        raise RuntimeError('Privacy batch did not complete every selected video')
    for source in sources:
        relative=source.relative_to(config.input_dir)
        key=hashlib.sha256(str(relative).encode()).hexdigest()
        record=json.loads((state/'jobs'/f'{key}.json').read_text())
        output=config.output_dir/relative.with_name(relative.stem+'_blurred.mp4')
        details=record.get('details',{})
        if (record.get('status')!='complete' or record.get('source')!=str(source)
                or record.get('output')!=str(output) or not output.is_file() or output.stat().st_size==0
                or details.get('status')!='complete' or details.get('frames',0)<=0
                or record.get('input_sha256')!=batch.sha256(source)
                or record.get('output_sha256')!=batch.sha256(output)):
            raise RuntimeError(f'Incomplete or changed privacy output: {relative}')


def run_pending(service, shared_root, data, *, limit=0, runner=None):
    root,data=_validate(shared_root,data,limit)
    runner=runner or _run_session
    results=[]
    while not limit or len(results)<limit:
        row=service.claim_video_privacy_blur()
        if row is None:
            break
        result=dict(id=row['id'],status='failed')
        succeeded=False
        try:
            config,sources=_task(data,resolve_session(root,row['local_path']))
            runner(config)
            _check_outputs(config,sources)
            result.update(status='completed',video_count=len(sources),output_dir=str(config.output_dir))
            succeeded=True
        except Exception as exc:
            result['error']=str(exc)
        except BaseException:
            if not service.finish_video_privacy_blur(row['id'],succeeded=False):
                raise RuntimeError(f"Failed to write interrupted privacy status: {row['id']}")
            raise
        if not service.finish_video_privacy_blur(row['id'],succeeded=succeeded):
            raise RuntimeError(f"Failed to write privacy status: {row['id']}")
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description='从所选数据表领取立体矫正已完成的数据，执行视频人脸打码。')
    parser.add_argument('--shared-root',required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args=parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url=database_url_or_error(parser)
    try:
        data=load_yaml(Path(args.config).expanduser())
        _validate(args.shared_root,data,args.limit)
        for name in ('ultralytics','torch','torchvision','numpy','opencv-python'):
            importlib.metadata.version(name)
    except (ValueError,OSError,importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    service=stage_service_or_error(parser, url, args.table, 'video_privacy_blur', args=args)
    def interrupted(_signum,_frame):
        raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        results=run_pending(service,args.shared_root,data,limit=args.limit)
        print(json.dumps({'stage':'video_privacy_blur','records':results},indent=2,ensure_ascii=False))
        if any(r['status']!='completed' for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM,previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__=='__main__':
    main()
