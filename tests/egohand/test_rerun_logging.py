import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboego_hand_vis.egohand.rerun_logging import rgb_overlay_entity, side_mesh_entity


def test_side_mesh_entity_uses_camera_opencv_hands_namespace():
    assert side_mesh_entity("ego_left", "current") == "camera_opencv/hands/ego_left/current"


def test_rgb_overlay_entity_uses_camera_left_rgb_namespace():
    assert (
        rgb_overlay_entity("normal", "ego_right", "mesh_projected_points")
        == "camera_opencv/camera/left/rgb/overlay/normal/ego_right/mesh_projected_points"
    )
