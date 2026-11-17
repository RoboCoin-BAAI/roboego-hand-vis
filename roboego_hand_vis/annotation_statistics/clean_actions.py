"""Clean only actions.vocabulary; preserve original subtask annotations."""
import argparse
import copy
import json
import unicodedata
from pathlib import Path
from .run import find_annotations
from .readme_sync import prepare_readme, write_readme


def normalize_label(value, rules):
    options = rules.get("normalize", {})
    if options.get("lowercase"):
        value = value.lower()
    if options.get("trim_whitespace_and_edge_punctuation"):
        def edge(char):
            return char.isspace() or unicodedata.category(char).startswith("P")
        start, end = 0, len(value)
        while start < end and edge(value[start]):
            start += 1
        while end > start and edge(value[end - 1]):
            end -= 1
        value = value[start:end]
    return value


def clean_document(document, rules):
    result = copy.deepcopy(document)
    deleted = set(rules["delete"])
    replacements = rules.get("replace", {})
    def clean(value):
        if isinstance(value, list):
            cleaned = []
            seen = set()
            for item in value:
                if isinstance(item, str):
                    item = normalize_label(item, rules)
                    if not item or item in deleted:
                        continue
                    item = replacements.get(item, item)
                    if item in seen:
                        continue
                    seen.add(item)
                cleaned.append(item)
            return cleaned
        if isinstance(value, str):
            value = normalize_label(value, rules)
            return [] if not value or value in deleted else replacements.get(value, value)
        return value
    actions = result.get("actions")
    if isinstance(actions, dict) and "vocabulary" in actions:
        actions["vocabulary"] = clean(actions["vocabulary"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("nonverb_rules.json"))
    parser.add_argument("--pattern", default="episode_000000_structured.json")
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8-sig"))
    if not isinstance(rules.get("delete"), dict) or not all(isinstance(w, str) and w for w in rules["delete"]):
        parser.error("Rules must contain a delete object with nonempty word keys")
    replacements = rules.get("replace", {})
    if not isinstance(replacements, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v
        for k, v in replacements.items()
    ):
        parser.error("replace must map nonempty strings to nonempty strings")
    if any(k in rules["delete"] or v in rules["delete"] or v in replacements
           for k, v in replacements.items()):
        parser.error("Replacement rules must not overlap deletions or form chains")
    paths = find_annotations(args.input_root.resolve(), args.pattern)
    changed = failed = readmes = 0
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError("JSON root must be an object")
            cleaned = clean_document(document, rules)
            prepared = prepare_readme(path, clean_document, rules, 'actions')
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
