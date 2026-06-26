import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hand.mesh_depth_alignment_impl import parse_args
import scripts.hand.mesh_depth_alignment_impl as alignment_impl


def test_parse_new_method_only_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["view", "--new_method_only"])

    args = parse_args()

    assert args.new_method_only is True


def test_alignment_impl_keeps_runtime_mesh_conversion_dependency():
    assert callable(alignment_impl.hamer_rendered_obj_to_camera_opencv)
    assert callable(alignment_impl.rotate_points_axis_angle)
