import json
from pathlib import Path

from scripts.hand.run_hamer_ego_filter import _stamp_ego_mode


def test_stamp_ego_mode_records_tolerant_mode_in_candidates_json(tmp_path: Path):
    output_dir = tmp_path / "hamer_out"
    output_dir.mkdir()
    candidates = output_dir / "ego_hand_candidates.json"
    candidates.write_text(json.dumps({"frames": []}), encoding="utf-8")

    _stamp_ego_mode(str(output_dir), "tolerant")

    payload = json.loads(candidates.read_text(encoding="utf-8"))
    assert payload["ego_mode"] == "tolerant"
    assert payload["metadata"]["ego_mode"] == "tolerant"
