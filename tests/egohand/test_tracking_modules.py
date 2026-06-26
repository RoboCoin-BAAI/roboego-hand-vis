import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_tracking_report_module_exports_json_and_csv_writers():
    from roboego_hand_vis.egohand.track_report import jsonable, write_tracks_csv, write_tracks_json

    assert callable(jsonable)
    assert callable(write_tracks_csv)
    assert callable(write_tracks_json)


def test_tracking_debug_module_exports_contact_sheet_helpers():
    from roboego_hand_vis.egohand.track_debug import draw_debug_frame, export_debug_contact_sheet, make_contact_sheet

    assert callable(draw_debug_frame)
    assert callable(export_debug_contact_sheet)
    assert callable(make_contact_sheet)


def test_tracking_rerun_module_exports_layer_helpers():
    from roboego_hand_vis.egohand.track_rerun import (
        hand_mesh_entity_paths,
        hand_rgb_overlay_layers,
        iter_rerun_offsets,
        log_rerun,
    )

    args = SimpleNamespace(start=10, num_frames=5, rrd_start=12, rrd_num_frames=2)

    assert list(iter_rerun_offsets(args)) == [(2, 12), (3, 13)]
    assert hand_mesh_entity_paths("ego_left", {"status": "recovered"}, True) == [
        "camera_opencv/hands/ego_left/current",
        "camera_opencv/hands/ego_left/recovered",
    ]
    assert hand_rgb_overlay_layers("ego_left", {"status": "detected"}, True) == ["current", "normal"]
    assert callable(log_rerun)
