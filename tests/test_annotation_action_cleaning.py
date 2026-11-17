from roboego_hand_vis.annotation_statistics.clean_actions import clean_document


def test_current_rules_delete_confirmed_phase_labels():
    import json
    from pathlib import Path
    rules = json.loads((Path(__file__).parents[1] / "roboego_hand_vis" /
                        "annotation_statistics" / "nonverb_rules.json").read_text(encoding="utf-8"))
    words = ["continue", "repeat", "keep", "start", "complete", "finish", "hang"]
    result = clean_document({"subtask": [{"action": words}],
                             "actions": {"vocabulary": words}}, rules)
    assert result["subtask"][0]["action"] == words
    assert result["actions"]["vocabulary"] == ["hang"]
    assert "finish" not in rules["replace"]
    assert not (set(rules["replace"].values()) & set(rules["delete"]))


def test_delete_only_confirmed_labels():
    source = {"scene": "retain", "subtask": [{"action": ["carefully", "wipe", "hang"]}, {"action": ["neatly"]}, {"action": "carefully"}], "actions": {"vocabulary": ["carefully", "neatly", "wipe", "continue", "upright"]}}
    rules = {"delete": {"carefully": "", "neatly": ""}}
    result = clean_document(source, rules)
    assert result["subtask"] == source["subtask"]
    assert result["actions"]["vocabulary"] == ["wipe", "continue", "upright"]
    assert result["scene"] == "retain"
    assert source["subtask"][1]["action"] == ["neatly"]
    assert clean_document(result, rules) == result


def test_normalize_before_deletion_and_synonyms():
    rules = {"delete": {"carefully": ""}, "replace": {"put": "place"},
             "normalize": {"lowercase": True, "trim_whitespace_and_edge_punctuation": True}}
    words = [" PUT, ", "Place", "Carefully!", "lift,", "‘SORT，’", "pick_up", "fine-tune", ",,,"]
    source = {"subtask": [{"action": words}], "actions": {"vocabulary": words}}
    result = clean_document(source, rules)
    expected = ["place", "lift", "sort", "pick_up", "fine-tune"]
    assert result["subtask"] == source["subtask"]
    assert result["actions"]["vocabulary"] == expected
    assert clean_document(result, rules) == result


def test_confirmed_synonyms_and_stable_deduplication():
    rules = {"delete": {}, "replace": {"inspect": "check", "verify": "check",
             "grab": "grasp", "join": "connect", "fasten": "secure", "finish": "complete"}}
    words = ["inspect", "check", "verify", "grab", "grasp", "join", "fasten", "finish", "hang"]
    document = {"subtask": [{"action": words}, {"action": "verify"}],
                "actions": {"vocabulary": words}}
    result = clean_document(document, rules)
    expected = ["check", "grasp", "connect", "secure", "complete", "hang"]
    assert result["subtask"] == document["subtask"]
    assert result["actions"]["vocabulary"] == expected
    assert clean_document(result, rules) == result


def test_missing_vocabulary_does_not_modify_or_create_fields():
    document = {"subtask": [{"action": ["PUT,", "continue"]}], "scene": "retain"}
    assert clean_document(document, {"delete": {"continue": ""},
                                   "replace": {"put": "place"}}) == document
