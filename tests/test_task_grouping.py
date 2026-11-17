import json
from pathlib import Path

from roboego_hand_vis.annotation_statistics.group_tasks import suggest

RULES = json.loads((Path(__file__).parents[1] / "roboego_hand_vis" /
                    "annotation_statistics/task_coarse_rules.json").read_text())


def test_document_destinations_are_ignored():
    for title in ["Organize documents into a folder", "Organize papers into a stack",
                  "Organize files on the desk"]:
        assert suggest(title, RULES)[0] == "organize document"


def test_broad_objects_and_goals():
    assert suggest("Clean bedside table", RULES)[0] == "clean furniture"
    assert suggest("Clean and organize desk", RULES)[0] == "clean furniture"
    assert suggest("Fold and store clothing", RULES)[0] == "organize clothing"
    assert suggest("Store data cable", RULES)[0] == "organize cable"
    assert suggest("Repair data cable", RULES)[0] == "repair cable"


def test_unknown_titles_are_not_forced_into_existing_tasks():
    assert suggest("Unspecified activity", RULES)[0] == ""
