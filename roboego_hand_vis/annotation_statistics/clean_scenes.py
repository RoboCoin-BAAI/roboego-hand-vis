"""Migrate scene classification using confirmed HumanCOIN scene pairs."""
import argparse
import copy
import json
import re
from pathlib import Path

from .run import find_annotations
from .readme_sync import prepare_readme, write_readme


def validate_pair(category, area, rules):
    if category not in rules["categories"]:
        raise ValueError(f"Unmapped scene category: {category!r}")
    if not isinstance(area, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", area):
        raise ValueError(f"Invalid scene area: {area!r}")
    if category in ("unknown", "other") and area != category:
        raise ValueError(f"Fallback category and area must match: {category}/{area}")
    for owner, areas in rules["categories"].items():
        if area in areas and category != owner:
            raise ValueError(f"Area {area!r} belongs to {owner!r}, not {category!r}")


def clean_document(document, rules):
    result = copy.deepcopy(document)
    classification = result.get("scene_classification")
    if classification is None:
        classification = {}
    if not isinstance(classification, dict):
        raise ValueError("scene_classification must be an object or null")
    classification = copy.deepcopy(classification)
    category, area = classification.get("category"), classification.get("area")
    missing = lambda value: value is None or isinstance(value, str) and not value.strip()
    if missing(category) and missing(area):
        classification.update(rules["missing_classification"])
    else:
        if missing(category) or missing(area):
            raise ValueError("Partial scene classification; no confirmed mapping")
        for mapping in rules["pair_mapping"]:
            if mapping["from"] == {"category": category, "area": area}:
                classification.update(mapping["to"])
                break
    validate_pair(classification.get("category"), classification.get("area"), rules)
    result["scene_classification"] = classification
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("scene_rules.json"))
    parser.add_argument("--pattern", default="episode_000000_structured.json")
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8-sig"))
    paths = find_annotations(args.input_root.resolve(), args.pattern)
    changed = failed = readmes = 0
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("JSON root must be an object")
            cleaned = clean_document(document, rules)
            prepared = prepare_readme(path, clean_document, rules, 'scene')
            if cleaned != document:
                path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                changed += 1
            readmes += write_readme(prepared)
        except (OSError, ValueError) as error:
            failed += 1
            print(f"Error: {path}: {error}")
    print(f"Scanned: {len(paths)}, modified: {changed}, readmes modified: {readmes}, failed: {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
