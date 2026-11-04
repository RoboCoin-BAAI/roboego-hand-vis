"""L1 front stereo alignment using rectified first frames."""
import hashlib

from ..common import guarded
from ..stereo_alignment import check_stereo_alignment


def check(ctx):
    from roboego_hand_vis.lerobot_stereo_rectification.L1.compute import load_geometry

    checks = []
    for episode in ctx.episodes:
        def assess():
            path = ctx.root / 'calibrations/camera-camchain.yaml'
            geometry = load_geometry(path)
            provenance = dict(path='calibrations/camera-camchain.yaml',
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            return check_stereo_alignment(ctx, episode, geometry=geometry,
                provenance=provenance, camera_names=('head_left', 'head_right'))

        items = guarded('readable', assess)
        for item in items:
            item['name'] = f'{episode.parent.name}/{episode.name}/{item["name"]}'
        checks.extend(items)
    return checks
