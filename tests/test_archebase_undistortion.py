from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from roboego_hand_vis.lerobot_stereo_rectification.archebase.compute import geometry, project_points, transform_hands


def test_projection_matches_fisheye_undistort_points():
    k=np.array([[680.,0,946.],[0,681.,586.],[0,0,1.]])
    d=np.array([.04,-.02,.002,-.002])
    g=geometry(dict(k=k.ravel().tolist(),d=d.tolist(),width=1920,height=1200,distortion_model='equidistant'))
    xyz=np.array([[[-.05,.18,.57],[.2,.3,.8]]])
    raw,_=cv2.fisheye.projectPoints(xyz.reshape(-1,1,3),np.zeros(3),np.zeros(3),k,d)
    expected=cv2.fisheye.undistortPoints(raw,k,d,R=g['R'],P=g['K_new']).reshape(1,2,2)
    transformed,actual=project_points(xyz,g['R'],g['K_new'])
    np.testing.assert_allclose(actual,expected,atol=1e-8)
    np.testing.assert_array_equal(transformed,xyz)


def test_rotation_and_invalid_depth():
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    xyz=np.array([[1.,2.,4.],[1.,2.,-1.]])
    transformed,uv=project_points(xyz,rotation,np.eye(3))
    np.testing.assert_allclose(transformed[0],[-2,1,4])
    np.testing.assert_allclose(uv[0],[-.5,.25])
    assert np.isnan(uv[1]).all()


def test_hand_files_share_left_geometry_and_preserve_timestamps(tmp_path):
    root=tmp_path/'dataset'
    episode=root/'data/chunk-000/episode_000000'
    episode.mkdir(parents=True)
    info=dict(features={})
    names=[f'p{i}_{a}' for i in range(21) for a in ('x','y','z')]
    for side in ('left','right'):
        key=f'observation.hand_{side}_points'
        cols=dict(timestamp=pa.array([-.001,0.033],type=pa.float64()),episode_index=[0,0])
        cols.update({n:pa.array([1. if n.endswith('_z') else .1]*2,type=pa.float32()) for n in names})
        pq.write_table(pa.table(cols),episode/f'{key}.parquet')
        info['features'][key]=dict(dtype='float32',shape=[63],names=names)
    staging=root/'stage'
    ctx=SimpleNamespace(root=root,info=info)
    result=transform_hands(ctx,episode,staging,dict(R=np.eye(3),K_new=np.array([[100.,0,20],[0,100,30],[0,0,1.]])),info)
    assert len(result)==4
    for side in ('left','right'):
        source=pq.read_table(episode/f'observation.hand_{side}_points.parquet')
        derived=pq.read_table(staging/episode.relative_to(root)/f'observation.hand_{side}_points_undistorted.parquet')
        assert source.equals(derived)
        uv=pq.read_table(staging/episode.relative_to(root)/f'observation.hand_{side}_points_2d_undistorted.parquet')
        np.testing.assert_allclose(uv['p0_u'].to_numpy(),30)
        np.testing.assert_allclose(uv['p0_v'].to_numpy(),40)
        assert uv['timestamp'].equals(source['timestamp'])
