"""L1 rectified VIO queue, reusing the original OpenVINS prepare/run command."""

import argparse
import json
import math
import os
from pathlib import Path
import signal

from roboego_hand_vis.common.processes import run_logged_command
from roboego_hand_vis.database.validators import resolve_session
from .cli import DEFAULT_CONFIG, config_from_yaml_and_args, load_yaml, parse_args
from .runner import build_openvins_command

PROJECT_ROOT=Path(__file__).resolve().parents[2]


def _config(data,session):
    return config_from_yaml_and_args(data,parse_args([
        '--session',str(session),'--output_dir',str(session/'vio'),'--input_mode','rectified']))


def _validate(shared_root,data,limit):
    root=Path(shared_root).expanduser().resolve()
    if limit<0 or not root.is_dir():
        raise ValueError('Need a valid shared root and nonnegative limit')
    config=_config(data,root)
    for path,name in ((config.openvins_binary,'OpenVINS binary'),(config.system_python,'System Python')):
        if not path.is_file() or not os.access(path,os.X_OK):
            raise ValueError(f'{name} is missing or not executable: {path}')
    if (not math.isfinite(config.start_s) or config.start_s<0
            or (config.duration_s is not None and (not math.isfinite(config.duration_s) or config.duration_s<=0))
            or not math.isfinite(config.stereo_tolerance_s) or config.stereo_tolerance_s<0
            or not math.isfinite(config.max_imu_gap_s) or config.max_imu_gap_s<=0
            or not math.isfinite(config.max_imu_boundary_gap_s) or config.max_imu_boundary_gap_s<0):
        raise ValueError('Invalid VIO timing parameters')
    return root


def _inside(path,session):
    if not path.resolve().is_relative_to(session):
        raise ValueError('VIO input/output paths must remain inside their session')
    return path


def _identity(path):
    if not path.is_file():
        return None
    stat=path.stat()
    return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def _task(data,session):
    config=_config(data,session)
    _inside(config.output_dir,session)
    if any(path.is_symlink() for path in config.output_dir.rglob('*')):
        raise ValueError('VIO output must not contain symbolic links')
    manifest=_inside(session/'dataset_manifest/dataset_manifest.json',session)
    required=[manifest,session/'timestamps/left_timestamps.txt',session/'timestamps/right_timestamps.txt',
              session/'imu/imu0.csv',session/'calibrations/camera_imu-camchain-imucam.yaml',
              session/'calibrations/camera_imu-imu.yaml',session/'calibrations/rectify_params.yaml']
    info=json.loads(manifest.read_text())
    required += [Path(info['videos'][eye]['path']).expanduser().resolve() for eye in ('left','right')]
    optional=session/'calibrations/camera-camchain.yaml'
    if optional.exists() or optional.is_symlink():
        required.append(optional)
    for path in required:
        if not _inside(path,session).is_file():
            raise ValueError(f'VIO input file is missing: {path.name}')
    annotation=_inside(session/'annotation',session).resolve()
    annotation.mkdir(parents=True,exist_ok=True)
    log=_inside(annotation/'vio_database.log',session)
    paths=[config.output_dir/'run_manifest.json',config.output_dir/'trajectory/run_report.json',
           config.output_dir/'trajectory/trajectory.jsonl']
    for path in paths:
        _inside(path,session)
    return config,log,{path:_identity(path) for path in paths}


def _run_session(config,log):
    command=build_openvins_command(config,stage_dir=Path(__file__).resolve().parent)
    run_logged_command(['bash','-lc',command],log,cwd=PROJECT_ROOT)


def _check_outputs(config,before):
    for path,identity in before.items():
        current=_identity(path)
        if current is None or current[0]<=0 or current==identity:
            raise RuntimeError(f'VIO did not produce a fresh complete output: {path.name}')
    manifest=json.loads((config.output_dir/'run_manifest.json').read_text())
    if (manifest.get('session_dir')!=str(config.session_dir) or manifest.get('output_dir')!=str(config.output_dir)
            or manifest.get('input_mode')!='rectified'):
        raise RuntimeError('VIO manifest does not match the current rectified session')
    trajectory=config.output_dir/'trajectory/trajectory.jsonl'
    report=json.loads((config.output_dir/'trajectory/run_report.json').read_text())
    count=report.get('pose_count')
    if report.get('initialized') is not True or type(count) is not int or count<=0:
        raise RuntimeError('OpenVINS was not initialized or produced no poses')
    if report.get('artifacts',{}).get('trajectory_jsonl')!=str(trajectory):
        raise RuntimeError('VIO report references a different trajectory')
    total=0
    previous=None
    with trajectory.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            pose=json.loads(line)
            timestamp=pose['timestamp_camera']
            if (not math.isfinite(timestamp) or (previous is not None and timestamp<=previous)
                    or not math.isfinite(pose['timestamp_imu'])):
                raise RuntimeError('VIO trajectory contains invalid or unordered timestamps')
            for key in ('frame_index','right_frame_index'):
                if type(pose.get(key)) is not int or pose[key]<0:
                    raise RuntimeError('VIO trajectory contains invalid frame indices')
            for key,width in (('orientation_world_imu_xyzw',4),('position_world_imu',3),('velocity_world_imu',3)):
                value=pose[key]
                if not isinstance(value,list) or len(value)!=width or not all(math.isfinite(x) for x in value):
                    raise RuntimeError(f'VIO trajectory contains invalid {key}')
            if sum(x*x for x in pose['orientation_world_imu_xyzw'])<=0:
                raise RuntimeError('VIO trajectory contains a zero quaternion')
            previous=timestamp
            total+=1
    if total!=count:
        raise RuntimeError('VIO trajectory record count does not match pose_count')
    return count


def run_pending(service,shared_root,data,*,limit=0,runner=None):
    root=_validate(shared_root,data,limit)
    results=[]
    while not limit or len(results)<limit:
        row=service.claim_vio()
        if row is None:
            break
        result=dict(id=row['id'],status='failed')
        succeeded=False
        try:
            config,log,before=_task(data,resolve_session(root,row['local_path']))
            if runner is None:
                _run_session(config,log)
            else:
                runner(config)
            count=_check_outputs(config,before)
            result.update(status='completed',pose_count=count,output_dir=str(config.output_dir))
            succeeded=True
        except Exception as exc:
            result['error']=str(exc)
        except BaseException:
            if not service.finish_vio(row['id'],succeeded=False):
                raise RuntimeError(f"Failed to write interrupted VIO status: {row['id']}")
            raise
        if not service.finish_vio(row['id'],succeeded=succeeded):
            raise RuntimeError(f"Failed to write VIO status: {row['id']}")
        results.append(result)
    return results


def main():
    parser=argparse.ArgumentParser(description='从所选数据表领取立体矫正已完成的数据，运行 rectified VIO。')
    parser.add_argument('--shared-root',required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--input_mode',choices=('rectified',),default='rectified')
    from roboego_hand_vis.database.cli import add_table_argument, stage_service_or_error
    add_table_argument(parser)
    args=parser.parse_args()
    from roboego_hand_vis.database.connection_config import database_url_or_error
    url=database_url_or_error(parser)
    try:
        data=load_yaml(Path(args.config).expanduser())
        _validate(args.shared_root,data,args.limit)
    except (ValueError,OSError,TypeError) as exc:
        parser.error(str(exc))
    service=stage_service_or_error(parser, url, args.table, 'vio', args=args)
    def interrupted(_signum,_frame):
        raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        results=run_pending(service,args.shared_root,data,limit=args.limit)
        print(json.dumps({'stage':'vio','records':results},indent=2,ensure_ascii=False))
        if any(r['status']!='completed' for r in results):
            raise SystemExit(1)
    finally:
        signal.signal(signal.SIGTERM,previous)
        if hasattr(service, "workspace"):
            service.workspace.release()
        service.database.close()


if __name__=='__main__':
    main()
