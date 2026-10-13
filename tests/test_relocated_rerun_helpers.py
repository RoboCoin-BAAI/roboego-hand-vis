from pathlib import Path

from roboego_hand_vis.hand_depth_alignment import rerun_utils
from roboego_hand_vis.hand_rerun_visualization import export_depth_pointcloud


def test_relocated_artifact_resolver_uses_repository_root(tmp_path,monkeypatch):
    assert rerun_utils.PROJECT_ROOT==Path(__file__).resolve().parents[1]
    artifact=tmp_path/'data/example.npy'
    artifact.parent.mkdir()
    artifact.write_bytes(b'depth')
    monkeypatch.setattr(rerun_utils,'PROJECT_ROOT',tmp_path)
    assert rerun_utils.resolve_existing_artifact_path('../RoboEgoPipe/data/example.npy')==artifact
    assert rerun_utils.resolve_existing_artifact_path(artifact)==artifact


def test_pointcloud_cli_remains_available_after_move(monkeypatch):
    monkeypatch.setattr('sys.argv',['export_depth_pointcloud','--dataset','ropedia','--frame','7'])
    args=export_depth_pointcloud.parse_args()
    path,video,intrinsics,scale=export_depth_pointcloud.dataset_defaults(args)
    assert path.name=='frame_000007_foundation_depth_m.npy'
    assert intrinsics==export_depth_pointcloud.ROPEDIA_K
    assert scale==1
