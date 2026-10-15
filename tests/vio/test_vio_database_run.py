import json
import sys
from pathlib import Path

import pytest

from roboego_hand_vis.vio.database_run import run_pending


class Service:
    def __init__(self,paths):
        self.rows=[dict(id=i,local_path=p) for i,p in enumerate(paths)]
        self.results=[]
    def claim_vio(self):
        return self.rows.pop(0) if self.rows else None
    def finish_vio(self,data_id,*,succeeded):
        self.results.append((data_id,succeeded))
        return True


def setup(root):
    binary=root/'openvins/bin'
    binary.parent.mkdir()
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    session=root/'session'
    paths=['timestamps/left_timestamps.txt','timestamps/right_timestamps.txt','imu/imu0.csv',
           'calibrations/camera_imu-camchain-imucam.yaml','calibrations/camera_imu-imu.yaml',
           'calibrations/rectify_params.yaml','videos/left_rectified.mp4','videos/right_rectified.mp4']
    for name in paths:
        p=session/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('input')
    manifest=session/'dataset_manifest/dataset_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'videos':{eye:{'path':str(session/f'videos/{eye}_rectified.mp4')} for eye in ('left','right')}}))
    return {'openvins':{'root':str(binary.parent),'binary':str(binary)},'runtime':{'system_python':sys.executable}}


def publish(config):
    assert config.input_mode=='rectified' and not config.prepare_only
    out=config.output_dir
    trajectory=out/'trajectory'
    trajectory.mkdir(parents=True,exist_ok=True)
    (out/'run_manifest.json').write_text(json.dumps({'session_dir':str(config.session_dir),'output_dir':str(out),'input_mode':'rectified'}))
    (trajectory/'run_report.json').write_text(json.dumps({'initialized':True,'pose_count':2,'published_stereo_pairs':3,
        'artifacts':{'trajectory_jsonl':str(trajectory/'trajectory.jsonl')}}))
    rows=[{'timestamp_camera':1000+i*.03,'timestamp_imu':1000+i*.03,'frame_index':i+1,'right_frame_index':i+1,
           'orientation_world_imu_xyzw':[0,0,0,1],'position_world_imu':[0,0,0],'velocity_world_imu':[0,0,0]} for i in range(2)]
    (trajectory/'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_success_does_not_require_pose_for_every_video_frame(tmp_path):
    data=setup(tmp_path)
    service=Service(['session'])
    result=run_pending(service,tmp_path,data,runner=publish)
    assert result[0]['status']=='completed',result
    assert result[0]['pose_count']==2
    assert service.results==[(0,True)]


@pytest.mark.parametrize('failure',['uninitialized','truncated','nan'])
def test_invalid_trajectory_fails(tmp_path,failure):
    data=setup(tmp_path)
    service=Service(['session'])
    def broken(config):
        publish(config)
        trajectory=config.output_dir/'trajectory'
        if failure=='uninitialized':
            (trajectory/'run_report.json').write_text('{"initialized":false,"pose_count":0}')
        else:
            p=trajectory/'trajectory.jsonl'
            lines=p.read_text().splitlines()
            if failure=='truncated':
                p.write_text(lines[0]+'\n')
            else:
                row=json.loads(lines[0]);row['position_world_imu'][0]=float('nan')
                p.write_text(json.dumps(row)+'\n'+lines[1]+'\n')
    assert run_pending(service,tmp_path,data,runner=broken)[0]['status']=='failed'
    assert service.results==[(0,False)]


def test_old_success_cannot_mask_new_run_without_outputs(tmp_path):
    data=setup(tmp_path)
    run_pending(Service(['session']),tmp_path,data,runner=publish)
    result=run_pending(Service(['session']),tmp_path,data,runner=lambda cfg:None)
    assert result[0]['status']=='failed'


def test_bad_paths_preflight_and_interrupt(tmp_path):
    data=setup(tmp_path)
    service=Service(['../outside','missing','session'])
    def interrupted(config):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_pending(service,tmp_path,data,runner=interrupted)
    assert service.results==[(0,False),(1,False),(2,False)]
    service=Service(['session'])
    (tmp_path/'openvins/bin').unlink()
    with pytest.raises(ValueError,match='binary'):
        run_pending(service,tmp_path,data,runner=publish)
    assert len(service.rows)==1


def test_default_runner_uses_logged_subprocess(tmp_path,monkeypatch):
    import inspect
    import shlex
    from roboego_hand_vis.vio import database_run
    data=setup(tmp_path)
    source=inspect.getsource(publish)
    script=("import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"+source+
            "\npublish(SimpleNamespace(session_dir=Path("+repr(str(tmp_path/'session'))+
            "),output_dir=Path("+repr(str(tmp_path/'session/vio'))+
            "),input_mode='rectified',prepare_only=False))\n")
    monkeypatch.setattr(database_run,'build_openvins_command',lambda cfg,**kw:shlex.join([sys.executable,'-c',script]))
    result=run_pending(Service(['session']),tmp_path,data)
    assert result[0]['status']=='completed',result
    assert (tmp_path/'session/annotation/vio_database.log').exists()


def make_chroot_runner(root, *, failure=None):
    import inspect
    import shlex
    # Exercise an actual shell subprocess with the same interface as vio-run.sh.
    script = root / 'vio runner.sh'
    source = inspect.getsource(publish)
    code = ("import json, sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n" + source + '''
session, output = map(Path, sys.argv[1:3])
assert not list(output.iterdir()), 'wrapper requires empty output'
assert sys.argv[3:6] == ['run', '--input_mode', 'rectified']
publish(SimpleNamespace(session_dir=session, output_dir=output, input_mode='rectified', prepare_only=False))
for path in [output/'run_manifest.json', output/'trajectory/run_report.json']:
    path.write_text(path.read_text().replace(str(output), '/data/output'))
''')
    if failure == 'exit':
        code += '\nsys.exit(7)\n'
    elif failure == 'wrong_session':
        code += "\np=output/'run_manifest.json'; d=json.loads(p.read_text()); d['session_dir']='/other'; p.write_text(json.dumps(d))\n"
    elif failure == 'uninitialized':
        code += "\n(output/'trajectory/run_report.json').write_text('{\"initialized\":false,\"pose_count\":0}')\n"
    elif failure == 'symlink':
        code += "\n(output/'external').symlink_to(session/'imu/imu0.csv')\n"
    script.write_text('exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(code) + ' "$@"\n')
    return script


def test_chroot_runner_without_host_dependencies_and_retry(tmp_path):
    data = setup(tmp_path)
    (tmp_path/'openvins/bin').unlink()
    data['runtime']['system_python'] = '/missing/python'
    script = make_chroot_runner(tmp_path)
    for _ in range(2):
        service = Service(['session'])
        result = run_pending(service, tmp_path, data, chroot_runner=script)
        assert result[0]['status'] == 'completed', result
        assert service.results == [(0, True)]
    out = tmp_path/'session/vio'
    assert json.loads((out/'run_manifest.json').read_text())['output_dir'] == str(out)
    assert json.loads((out/'trajectory/run_report.json').read_text())['artifacts']['trajectory_jsonl'] == str(out/'trajectory/trajectory.jsonl')
    assert list((tmp_path/'session').glob('.vio-previous-*'))


@pytest.mark.parametrize('failure', ['exit', 'wrong_session', 'uninitialized', 'symlink'])
def test_chroot_failure_preserves_previous_output(tmp_path, failure):
    data = setup(tmp_path)
    run_pending(Service(['session']), tmp_path, data, runner=publish)
    old = (tmp_path/'session/vio/run_manifest.json').read_bytes()
    script = make_chroot_runner(tmp_path, failure=failure)
    service = Service(['session'])
    result = run_pending(service, tmp_path, data, chroot_runner=script)
    assert result[0]['status'] == 'failed', result
    assert service.results == [(0, False)]
    assert (tmp_path/'session/vio/run_manifest.json').read_bytes() == old
    assert list((tmp_path/'session').glob('.vio-chroot-*'))


def test_missing_chroot_runner_fails_before_claim(tmp_path):
    data = setup(tmp_path)
    service = Service(['session'])
    with pytest.raises(ValueError, match='runner'):
        run_pending(service, tmp_path, data, chroot_runner=tmp_path/'missing.sh')
    assert len(service.rows) == 1


def test_chroot_forwards_vio_settings(tmp_path):
    data = setup(tmp_path)
    data['run'] = {'start_s': 2.5, 'duration_s': 8}
    data['prepare'] = {'stereo_tolerance_s': 0.002, 'max_imu_gap_s': 0.07,
                       'max_imu_boundary_gap_s': 0.2, 'rectified_downsample_cameras': False}
    script = make_chroot_runner(tmp_path)
    import shlex
    capture = tmp_path/'args.txt'
    script.write_text('printf "%s\\n" "$@" > ' + shlex.quote(str(capture)) + '\n' + script.read_text())
    result = run_pending(Service(['session']), tmp_path, data, chroot_runner=script)
    assert result[0]['status'] == 'completed', result
    args = capture.read_text().splitlines()
    for key, value in [('start_s', '2.5'), ('duration_s', '8.0'),
                       ('stereo_tolerance_s', '0.002'), ('max_imu_gap_s', '0.07'),
                       ('max_imu_boundary_gap_s', '0.2')]:
        assert float(args[args.index('--' + key) + 1]) == float(value)
    assert '--no-downsample_cameras' in args


def test_chroot_serializes_sessions_sharing_script(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import shlex
    import shutil
    data = setup(tmp_path)
    shutil.copytree(tmp_path/'session', tmp_path/'session2')
    manifest = tmp_path/'session2/dataset_manifest/dataset_manifest.json'
    manifest.write_text(manifest.read_text().replace('/session/', '/session2/'))
    script = make_chroot_runner(tmp_path)
    # mkdir fails if two wrappers enter the shared-rootfs critical section together.
    marker = shlex.quote(str(tmp_path/'rootfs-busy'))
    script.write_text('set -e\nmkdir ' + marker + '\nsleep 0.15\nrmdir ' + marker + '\n' + script.read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_pending, Service([name]), tmp_path, data, chroot_runner=script)
                   for name in ('session', 'session2')]
        for future in futures:
            result = future.result(timeout=10)
            assert result[0]['status'] == 'completed', result
