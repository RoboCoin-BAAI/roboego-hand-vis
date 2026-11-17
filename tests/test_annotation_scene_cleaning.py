import json
from pathlib import Path

import pytest

from roboego_hand_vis.annotation_statistics.clean_scenes import clean_document

RULES = json.loads((Path(__file__).parents[1] / "roboego_hand_vis" /
                    "annotation_statistics/scene_rules.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mapping", RULES["pair_mapping"])
def test_confirmed_pairs_preserve_descriptions(mapping):
    source = {"scene": "original scene", "scene_classification": {
        **mapping["from"], "original_description": "preserve"}, "objects": [{"name": "box"}]}
    result = clean_document(source, RULES)
    assert result["scene_classification"] == {**mapping["to"], "original_description": "preserve"}
    assert result["scene"] == source["scene"]
    assert result["objects"] == source["objects"]
    assert source["scene_classification"]["category"] == mapping["from"]["category"]
    assert clean_document(result, RULES) == result


@pytest.mark.parametrize("classification", [None, {}, {"category": None, "area": None}])
def test_missing_scene_becomes_unknown(classification):
    assert clean_document({"scene_classification": classification}, RULES)["scene_classification"] == {
        "category": "unknown", "area": "unknown"}


@pytest.mark.parametrize("category,area", [("household", "household_cleaning"),
                                          ("commercial_service", "food_preparation_area")])
def test_extensions_preserved(category, area):
    source = {"scene_classification": {"category": category, "area": area}}
    assert clean_document(source, RULES) == source


@pytest.mark.parametrize("category,area", [("household", "office"), ("other", "kitchen"),
                                          ("invalid", "office"), ("household", None)])
def test_unmapped_invalid_pairs_rejected(category, area):
    with pytest.raises(ValueError):
        clean_document({"scene_classification": {"category": category, "area": area}}, RULES)
