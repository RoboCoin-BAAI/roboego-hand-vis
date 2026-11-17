import numpy as np
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from roboego_hand_vis.lerobot_hand_pose.L3.world_points import (
    world_points, T_TRACKER_WRIST, R_TRACKER_SKELETON, R_TRACKER_SKELETON_RIGHT)


def test_mount_offset_rotates_and_finger_points_use_axis_alignment():
    assert np.allclose(T_TRACKER_WRIST, [0., -.085, -.058])
    points=np.array([[[0.,0,0],[.1,0,0]],[[0.,0,0],[.1,0,0]]])
    angles=np.array([[0.,0,0],[0,0,90.]])
    positions=np.array([[1.,2,3],[1.,2,3]])
    out,valid=world_points([0.,1.],points,[0.,1.],positions,angles,max_gap_s=2)
    assert valid.all()
    rotated=np.array([T_TRACKER_WRIST, [.085,0,-.058]])
    assert np.allclose(out[:,0],positions+rotated)
    assert np.allclose(out[0,1]-out[0,0],[0,0,.1])
    assert np.allclose(R_TRACKER_SKELETON.T@R_TRACKER_SKELETON,np.eye(3))
    assert np.isclose(np.linalg.det(R_TRACKER_SKELETON),1.)


def test_update_existing_offset_preserves_sources_and_skips_current(tmp_path):
    from scripts.L3.update_points_offset import update_session
    from roboego_hand_vis.lerobot_hand_pose.L3.world_points import convert
    ep = make_session(tmp_path)
    original_offset = T_TRACKER_WRIST.copy()
    try:
        T_TRACKER_WRIST[:] = [0., -.1085, -.0372]
        convert(tmp_path)
    finally:
        T_TRACKER_WRIST[:] = original_offset
    original_source = (ep / 'observation.hand_skeleton.parquet').read_bytes()
    old_points = (ep / 'observation.hand_left_points.parquet').read_bytes()
    assert update_session(tmp_path, True)['status'] == 'would_update'
    assert (ep / 'observation.hand_left_points.parquet').read_bytes() == old_points
    assert update_session(tmp_path)['status'] == 'done'
    for hand in ('left', 'right'):
        output = pq.read_table(ep / f'observation.hand_{hand}_points.parquet')
        assert np.allclose(output['wrist_y'].to_numpy(), -.085)
        assert np.allclose(output['wrist_z'].to_numpy(), -.058)
        assert not (ep / f'hand_{hand}_points_provenance.json').exists()
    assert update_session(tmp_path)['status'] == 'done'
    assert (ep / 'observation.hand_skeleton.parquet').read_bytes() == original_source


def test_both_hands_use_proper_rotation_without_extra_reflection():
    assert np.array_equal(R_TRACKER_SKELETON_RIGHT,R_TRACKER_SKELETON)
    assert np.isclose(np.linalg.det(R_TRACKER_SKELETON_RIGHT),1.)
    assert np.allclose(R_TRACKER_SKELETON@[1.,0,0],[0,0,1.])
    assert np.allclose(R_TRACKER_SKELETON@[0.,1,0],[1.,0,0])
    assert np.allclose(R_TRACKER_SKELETON@[0.,0,1],[0,1.,0])


def test_relaxed_timestamps_preserve_queries_and_keep_last_tracker_duplicate():
    from roboego_hand_vis.lerobot_hand_pose.L3.world_points import normalize_tracker_times
    queries=np.array([.01,0.,.01,np.nan,np.inf,.02])
    tracker=np.array([.02,0.,.02,np.nan])
    xyz=np.array([[99.,0,0],[0.,0,0],[2.,0,0],[100.,0,0]])
    rpy=np.zeros((4,3))
    out,valid=world_points(queries,np.zeros((6,21,3)),tracker,xyz,rpy,max_gap_s=.02)
    assert valid.tolist()==[True,True,True,False,False,True]
    assert np.allclose(out[valid,0,0],[1.,0.,1.,2.])
    assert np.isnan(out[~valid]).all()
    _,_,_,report=normalize_tracker_times(tracker,xyz,rpy)
    assert report['duplicate_timestamp_rows']==1
    assert report['nonfinite_timestamp_rows']==1
    assert report['backward_steps']==1


def test_single_or_no_finite_tracker_timestamp():
    points=np.zeros((3,21,3))
    out,valid=world_points([0.,.01,.02],points,[.01],np.zeros((1,3)),np.zeros((1,3)))
    assert valid.tolist()==[False,True,False]
    assert np.isfinite(out[1]).all()
    out,valid=world_points([0.,.01,.02],points,[np.nan],np.zeros((1,3)),np.zeros((1,3)))
    assert not valid.any() and np.isnan(out).all()


def test_default_gap_accepts_30ms_and_rejects_larger_bracket():
    points=np.zeros((1,21,3))
    xyz=np.zeros((2,3))
    _,valid=world_points([.015],points,[0.,.03],xyz,xyz)
    assert valid.tolist()==[True]
    out,valid=world_points([.015],points,[0.,.031],xyz,xyz)
    assert valid.tolist()==[False] and np.isnan(out).all()


def test_slerp_translation_and_no_extrapolation_or_gap_bridging():
    times=np.array([-.1,0,.5,1.,1.1])
    points=np.zeros((5,21,3))
    xyz=np.array([[0.,0,0],[2.,0,0]])
    rpy=np.array([[0.,0,0],[0.,0,90.]])
    out,valid=world_points(times,points,[0.,1.],xyz,rpy,max_gap_s=2.)
    assert valid.tolist()==[False,True,True,True,False]
    expected=np.array([1.+.085/np.sqrt(2),-.085/np.sqrt(2),-.058])
    assert np.allclose(out[2,0],expected)
    _,valid=world_points(times,points,[0.,1.],xyz,rpy,max_gap_s=.02)
    assert valid.tolist()==[False,True,False,True,False]


def make_session(root):
    ep=root/'data/chunk-000/episode_000000'
    ep.mkdir(parents=True)
    (root/'meta').mkdir()
    (root/'meta/info.json').write_text(json.dumps(dict(features={
        'observation.hand_skeleton':dict(fps=100.)},fps=30)))
    (root/'meta/episodes_stats.jsonl').write_text(json.dumps(dict(episode_index=0,stats={'original':{'count':[3]}}))+'\n')
    skeleton=dict(timestamp=[0.,.01,.02],episode_index=[0,0,0])
    for node in range(50):
        for a,value in zip('xyz',[.01*(node%25),.002*(node%25)*(1 if node<25 else -1),0.]):
            skeleton[f'node{node:02d}_{a}']=[value]*3
    pq.write_table(pa.table(skeleton),ep/'observation.hand_skeleton.parquet')
    for side,x in [('left',1.),('right',-1.)]:
        pq.write_table(pa.table(dict(timestamp=[0.,.02],episode_index=[0,0],
            x=[x,x],y=[0.,0.],z=[0.,0.],roll=[0.,0.],pitch=[0.,0.],yaw=[0.,0.])),
            ep/f'observation.{side}_wrist_pose.parquet')
    return ep


def test_both_hand_publish_stats_verified_reuse_and_source_change(tmp_path):
    from roboego_hand_vis.lerobot_hand_pose.L3.world_points import convert
    from roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points import worker
    ep=make_session(tmp_path)
    reports=convert(tmp_path)
    assert len(reports)==2
    for side,x in [('left',1.),('right',-1.)]:
        t=pq.read_table(ep/f'observation.hand_{side}_points.parquet').to_pydict()
        assert np.allclose(t['wrist_x'],x)
        assert np.allclose(t['wrist_y'],-.085)
        assert np.allclose(t['wrist_z'],-.058)
        assert np.allclose(np.array(t['thumbCMC_z'])-np.array(t['wrist_z']),.01)
    stats=json.loads((tmp_path/'meta/episodes_stats.jsonl').read_text())['stats']
    assert stats['original']=={'count':[3]}
    assert stats['observation.hand_left_points']['count']==[3]
    assert stats['observation.hand_right_points']['count']==[3]
    assert worker(tmp_path,'both',.03,False)['status']=='reused'
    p=ep/'observation.left_wrist_pose.parquet'
    table=pq.read_table(p)
    table=table.set_column(table.schema.get_field_index('x'),'x',pa.array([2.,2.]))
    pq.write_table(table,p)
    assert worker(tmp_path,'both',.03,False)['status']=='reused'
    assert worker(tmp_path,'both',.03,True)['status']=='done'
    assert worker(tmp_path,'both',.03,False)['status']=='reused'


def test_missing_right_input_never_publishes_left(tmp_path):
    from roboego_hand_vis.lerobot_hand_pose.L3.world_points import convert
    ep=make_session(tmp_path)
    (ep/'observation.right_wrist_pose.parquet').unlink()
    before=(tmp_path/'meta/info.json').read_bytes()
    with pytest.raises(FileNotFoundError):
        convert(tmp_path)
    assert not (ep/'observation.hand_left_points.parquet').exists()
    assert (tmp_path/'meta/info.json').read_bytes()==before
    assert not list(tmp_path.glob('.hand-world-*'))


def test_dry_run_batch_and_direct_script(tmp_path,capsys):
    from roboego_hand_vis.lerobot_hand_pose.L3.convert_world_points import main
    session=tmp_path/'L3-test'
    ep=make_session(session)
    assert main(['--source',str(tmp_path),'--dry-run'])==0
    assert 'observation.hand_right_points' in capsys.readouterr().out
    assert not (ep/'observation.hand_left_points.parquet').exists()
