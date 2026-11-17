import json
from pathlib import Path

import pytest

from roboego_hand_vis.annotation_statistics.clean_objects import clean_document, clean_name

RULES = json.loads((Path(__file__).parents[1] / "roboego_hand_vis" /
                    "annotation_statistics/object_rules.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("original,expected", [
    (" Small WHITE pill bottles, ", "pill bottle"),
    ("Flower pot 2", "flower pot"), ("open book", "book"),
    ("crumpled paper", "paper"), ("candy wrappers", "candy wrapper"),
    ("scissors", "scissors"), ("glasses", "glasses"), ("pliers", "pliers"),
    ("clothes", "clothes"), ("plastic bags", "plastic bag"),
    ("washing machine", "washing machine"), ("cutting board", "cutting board"),
    ("charging cable", "charging cable"), ("orange juice bottle", "orange juice bottle"),
    ("green tea bag", "green tea bag"), ("A4 paper", "a4 paper"),
    ("2.5kg weight plate", "2.5kg weight plate"),
    ("box with screws", "box with screws")])
def test_name_rules(original, expected):
    assert clean_name(original, RULES) == expected
    assert clean_name(expected, RULES) == expected


def test_only_object_names_change_and_instances_are_preserved():
    source = {"objects": [{"name": "blue cups", "attributes": {"color": "blue"}},
                          {"name": "white cups"}],
              "subtask": [{"object": "blue cups"}], "actions": {"vocabulary": ["hold"]}}
    result = clean_document(source, RULES)
    assert [obj["name"] for obj in result["objects"]] == ["cup", "cup"]
    assert result["objects"][0]["attributes"] == {"color": "blue"}
    assert result["subtask"] == source["subtask"]
    assert result["actions"] == source["actions"]
    assert source["objects"][0]["name"] == "blue cups"


@pytest.mark.parametrize("original,expected", [
    ("pliers (red handle)", "pliers"), ("flower pot (left)", "flower pot"),
    ("手机 (phone)", "phone"), ("水杯架 (cup holder)", "cup holder"),
    ("牛奶 card", "card"), ("card: grape juice (葡萄汁)", "card"),
    ("workpiece (hex nut)", "hex nut"),
    ("black power tool (electric screwdriver)", "electric screwdriver")])
def test_reviewed_bilingual_and_parenthetical_names(original, expected):
    assert clean_name(original, RULES) == expected


def test_unmapped_chinese_is_not_silently_stripped():
    with pytest.raises(ValueError, match="Unmapped Chinese"):
        clean_name("未知物品", RULES)
