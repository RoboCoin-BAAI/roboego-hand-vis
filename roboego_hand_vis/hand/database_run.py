"""L1 hand queue using the native MINT/EgoHandKit launcher unchanged."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import signal
import tempfile

import yaml

from roboego_hand_vis.database.validators import resolve_session, validate_relative_path
from . import run as native
from . import outputs
from .state import atomic_json, checked_path, file_identity, tree_identity, load_state

DEFAULT_CONFIG=Path(__file__).with_name('recommended.yaml')


def _validate(shared_root, data, limit):
    root=Path(shared_root).expanduser().resolve()
    if limit<0 or not root.is_dir():
        raise ValueError('Need a valid shared root and nonnegative limit')
    for name,default in (('video','videos/left_rectified.mp4'),('depth','depth')):
        validate_relative_path(data.get('input',{}).get(name,default))
    plan=native.build_plan(data,root,root/'hand')
    mint=Path(plan['environment']['MINT_ROOT'])
    ego=Path(plan['environment']['EGOHANDKIT_ROOT'])
    required=[(Path(plan['checkpoint']),'MINT checkpoint'),(ego/'run.py','EgoHandKit run.py'),
              (ego/'scripts/run_without_front.sh','EgoHandKit launcher'),
              (ego/'scripts/export_egohandkit_cache_mint.py','EgoHandKit exporter')]
    for path,name in required:
        if not path.is_file():
            raise ValueError(f'{name} not found: {path}')
    for path in (mint/'mint',mint/'scripts'):
        if not path.is_dir():
            raise ValueError(f'MINT directory not found: {path}')
    for value in (plan['stages'][0]['command'][0],plan['environment']['EGOHANDKIT_PYTHON']):
        if not Path(value).is_file() or not os.access(value,os.X_OK):
            raise ValueError(f'Python executable unavailable: {value}')
    if 'EGOHANDKIT_PYTHON' not in (ego/'scripts/run_without_front.sh').read_text():
        raise ValueError('EgoHandKit launcher needs the native path/Python configuration patch')
    return root


def _task(data, session, full_output):
    output=checked_path(session,'hand')
    annotation=checked_path(session,'annotation')
    receipt=checked_path(session,'annotation/hand_database_receipt.json')
    log=checked_path(session,'annotation/hand_database.log')
    plan=native.build_plan(data,session,output,full_output=True)
    for name in ('video','depth'):
        if not Path(plan[name]).is_relative_to(session):
            raise ValueError('Hand inputs must remain inside their session')
    if not Path(plan['video']).is_file() or not Path(plan['depth']).is_dir():
        raise ValueError('Rectified video or per-frame depth directory is missing')
    checkpoint=checked_path(output,'.work/state.json')
    if output.exists() and not receipt.is_file() and not checkpoint.is_file():
        raise ValueError('Existing hand output has no owned database receipt or resumable checkpoint')
    annotation.mkdir(parents=True,exist_ok=True)
    # A completion receipt survives compact-mode removal of the native checkpoint.
    return dict(plan=plan, output=output, annotation=annotation, receipt=receipt, log=log,
                resume=checkpoint.is_file(), checkpoint=checkpoint, full_output=full_output)


def _check_delivery(task):
    plan=task['plan']
    hashes=native.artifact_hashes(task['output'],task['output'],'egohandkit')
    fd,name=tempfile.mkstemp(prefix='.hand-validation-',suffix='.json',dir=task['annotation'])
    os.close(fd)
    result=Path(name)
    stage={'name':'validate_database_delivery','cwd':plan['input_dir'],'command':[
        plan['environment']['EGOHANDKIT_PYTHON'],str(Path(outputs.__file__).resolve()),'final',
        str(task['output']),plan['video'],str(result)]}
    try:
        code=native.run_stage(stage,{**os.environ,**plan['environment']},task['log'])
        if code:
            raise RuntimeError('Hand delivery validation failed; see annotation/hand_database.log')
        validation=json.loads(result.read_text())
        if (validation.get('input_frames',0)<=0
                or validation.get('parquet_rows')!=validation['input_frames']
                or validation.get('render_frames')!=validation['input_frames']):
            raise RuntimeError('Hand delivery has inconsistent frame counts')
        outputs.verify_delivery(task['output'],hashes)
        if not checked_path(task['output'],'run.log').is_file():
            raise RuntimeError('Native hand run.log is missing')
        return hashes,validation
    finally:
        result.unlink(missing_ok=True)


def run_pending(service, shared_root, data, *, limit=0, full_output=False, runner=None):
    root=_validate(shared_root,data,limit)
    runner=runner or native.execute
    results=[]
    while not limit or len(results)<limit:
        row=service.claim_hand()
        if row is None:
            break
        result=dict(id=row['id'],status='failed')
        succeeded=False
        try:
            task=_task(data,resolve_session(root,row['local_path']),full_output)
            plan=task['plan']
            inputs=(file_identity(plan['video']),tree_identity(plan['depth']))
            fingerprint=native.fingerprint(plan)
            if task['receipt'].is_file():
                receipt=json.loads(task['receipt'].read_text())
                if (receipt.get('plan')!=plan or receipt.get('fingerprint')!=fingerprint
                        or receipt.get('status')!='completed' or receipt.get('full_output')!=full_output):
                    raise ValueError('Completed hand inputs/configuration changed; existing output is retained')
                outputs.verify_delivery(task['output'],receipt['hashes'])
            else:
                if task['resume']:
                    state=load_state(task['checkpoint'])
                    if state.get('plan')!=plan or state.get('fingerprint')!=fingerprint:
                        raise ValueError('Hand checkpoint inputs/configuration changed; existing output is retained')
                with task['log'].open('a') as handle,redirect_stdout(handle),redirect_stderr(handle):
                    code=runner(plan,resume=task['resume'])
                if code==130:
                    raise KeyboardInterrupt
                if code!=0:
                    raise RuntimeError(f'Native hand launcher exited with code {code}; see hand/run.log')
            hashes,validation=_check_delivery(task)
            if inputs!=(file_identity(plan['video']),tree_identity(plan['depth'])):
                raise ValueError('Hand inputs changed during processing or validation; delivery retained')
            atomic_json(task['receipt'],dict(status='completed',plan=plan,
                        fingerprint=native.fingerprint(plan),hashes=hashes,validation=validation,full_output=full_output))
            if not full_output:
                work=checked_path(task['output'],'.work')
                if work.is_dir():
                    native.log_event(checked_path(task['output'],'run.log'),'database_cleanup_started')
                    if checked_path(task['output'],'.work/state.json').is_file():
                        outputs.cleanup_work(task['output'])
                    elif not any(work.iterdir()):
                        work.rmdir()
                    else:
                        raise ValueError('Hand cleanup found nonempty work directory without checkpoint')
                    native.log_event(checked_path(task['output'],'run.log'),'database_cleanup_completed')
            result.update(status='completed',output_dir=str(task['output']),
                          frame_count=validation['input_frames'],quality_status='not_validated')
            succeeded=True
        except Exception as exc:
            result['error']=str(exc)
        except BaseException:
            if not service.finish_hand(row['id'],succeeded=False):
                raise RuntimeError(f"Failed to write interrupted hand status: {row['id']}")
            raise
        if not service.finish_hand(row['id'],succeeded=succeeded):
            raise RuntimeError(f"Failed to write hand status: {row['id']}")
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description='从 L1 数据库领取深度估计已完成的数据，运行 MINT 与 EgoHandKit。')
    parser.add_argument('--shared-root',required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--full-output',action='store_true')
    args=parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    from roboego_hand_vis.database import DataService
    url=database_url_or_error(parser)
    try:
        with Path(args.config).expanduser().open() as stream:
            data=yaml.safe_load(stream)
        if not isinstance(data,dict):
            raise ValueError('Hand config must be a YAML mapping')
        _validate(args.shared_root,data,args.limit)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    service=DataService.from_url(url)
    def interrupted(_signum,_frame):
        raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        results=run_pending(service,args.shared_root,data,limit=args.limit,full_output=args.full_output)
        print(json.dumps({'stage':'hand','records':results},indent=2,ensure_ascii=False))
        if any(r['status']!='completed' for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM,previous)
        service.database.close()


if __name__=='__main__':
    main()
