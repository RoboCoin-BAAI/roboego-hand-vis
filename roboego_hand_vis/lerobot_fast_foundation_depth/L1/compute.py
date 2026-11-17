"""L1 camera and configuration adapter for shared depth inference."""
from pathlib import Path
from ..base import compute as shared

CAMERAS = {'left': 'head_left_rectified', 'right': 'head_right_rectified'}
DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")


def episode_inputs(session, episode, camera_names=None):
    return shared.episode_inputs(session, episode, camera_names=camera_names or CAMERAS)


def plan_session(session, camera_names=None):
    return shared.plan_session(session, camera_names=camera_names or CAMERAS)


def run_session(session, **options):
    options.setdefault("config", DEFAULT_CONFIG)
    options.setdefault("camera_names", CAMERAS)
    return shared.run_session(session, **options)
