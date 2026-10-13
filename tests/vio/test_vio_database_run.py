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
