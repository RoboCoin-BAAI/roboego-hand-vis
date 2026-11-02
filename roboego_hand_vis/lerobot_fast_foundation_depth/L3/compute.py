"""L3 camera-name adapter; reuse the GPU inference and storage implementation."""
from pathlib import Path

from roboego_hand_vis.lerobot_fast_foundation_depth.L1 import compute as shared

CAMERAS = {'left': 'headband_left_rgb_rectified', 'right': 'headband_right_rgb_rectified'}
DEFAULT_CONFIG = Path(__file__).with_name("recommended.yaml")

def episode_inputs(session, episode):
    return shared.episode_inputs(session, episode, camera_names=CAMERAS)

def plan_session(session):
    return shared.plan_session(session, camera_names=CAMERAS)

def run_session(session, **options):
    options.setdefault("config", DEFAULT_CONFIG)
    return shared.run_session(session, camera_names=CAMERAS, **options)
