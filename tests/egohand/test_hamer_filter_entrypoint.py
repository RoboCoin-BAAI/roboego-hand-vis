import json
from pathlib import Path

from scripts.hand.run_hamer_ego_filter import (
    _load_vlm_frame_priors,
    _load_wearer_regions,
    _stamp_ego_mode,
    _wearer_region_for_frame,
    _vlm_prior_for_frame,
)


def test_stamp_ego_mode_records_tolerant_mode_in_candidates_json(tmp_path: Path):
    output_dir = tmp_path / "hamer_out"
    output_dir.mkdir()
    candidates = output_dir / "ego_hand_candidates.json"
    candidates.write_text(json.dumps({"frames": []}), encoding="utf-8")

    _stamp_ego_mode(str(output_dir), "tolerant")

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert payload["ego_mode"] == "tolerant"
    assert payload["metadata"]["ego_mode"] == "tolerant"


def test_load_wearer_regions_reads_visible_vlm_bboxes(tmp_path: Path):
    regions_json = tmp_path / "stage1.json"
    regions_json.write_text(
        json.dumps(
            {
                "frames": [
                    {"t": 8, "wearer_region": {"visible": True, "bbox": [0.2, 0.5, 0.8, 1.0]}},
                    {"t": 9, "wearer_region": {"visible": False, "bbox": [0.0, 0.0, 1.0, 1.0]}},
                ]
            }
        ),
        encoding="utf-8",
    )

    regions = _load_wearer_regions(str(regions_json))

    assert regions == {8.0: [0.2, 0.5, 0.8, 1.0]}


def test_wearer_region_for_frame_uses_time_offset_and_nearest_second():
    regions = {8.0: [0.2, 0.5, 0.8, 1.0], 10.0: [0.3, 0.4, 0.9, 1.0]}

    bbox = _wearer_region_for_frame(regions, frame_idx=61, fps=30.0, stage1_time_offset_s=6.0)

    assert bbox == [0.2, 0.5, 0.8, 1.0]


def test_load_vlm_frame_priors_reads_hand_visibility_and_bbox(tmp_path: Path):
    regions_json = tmp_path / "stage1.json"
    regions_json.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "t": 8,
                        "hand_visible": True,
                        "hands_visible": {"left": False, "right": True},
                        "wearer_region": {"visible": True, "bbox": [0.2, 0.5, 0.8, 1.0]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    priors = _load_vlm_frame_priors(str(regions_json))
    prior = _vlm_prior_for_frame(priors, frame_idx=1, fps=30.0, stage1_time_offset_s=8.0)

    assert prior == {
        "wearer_bbox_xyxy": [0.2, 0.5, 0.8, 1.0],
        "hand_visible": True,
        "hands_visible": {"left": False, "right": True},
        "t": 8.0,
    }
