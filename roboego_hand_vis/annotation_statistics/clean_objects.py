"""Clean objects[].name using manually reviewed object naming rules."""
import argparse
import copy
import json
import re
from pathlib import Path

from .clean_actions import normalize_label
from .run import find_annotations
from .readme_sync import prepare_readme, write_readme


def clean_name(name, rules):
    name = rules.get("name_mapping", {}).get(name.lower().strip(), name)
    if rules.get("reject_unmapped_cjk") and re.search(r"[\u3400-\u9fff\uf900-\ufaff]", name):
        raise ValueError(f"Unmapped Chinese object name: {name!r}; add an English name_mapping first")
    name = normalize_label(name, rules)
    if rules.get("remove_trailing_instance_number"):
        name = re.sub(r"\s+\d+$", "", name).strip()
    words = name.split()
    removable = set(rules["remove_prefix_words"])
    while len(words) > 1 and words[0] in removable:
        phrase = " ".join(words)
        if any(phrase == prefix or phrase.startswith(prefix + " ")
               for prefix in rules.get("protected_prefixes", [])):
            break
        words.pop(0)
    # Only unambiguous terminal plurals from the explicit mapping are changed.
    # Leave prepositional descriptions for a later, separately confirmed round.
    if words and not set(words).intersection({"of", "with", "in", "on", "for", "from", "to"}):
        if words[-1] not in rules.get("preserve_plurals", []):
            words[-1] = rules["plural_to_singular"].get(words[-1], words[-1])
    return " ".join(words) or name


def clean_document(document, rules):
    result = copy.deepcopy(document)
    objects = result.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict) and isinstance(obj.get("name"), str):
                name = clean_name(obj["name"], rules)
                if name:
                    obj["name"] = name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("object_rules.json"))
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
            prepared = prepare_readme(path, clean_document, rules, 'objects')
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
