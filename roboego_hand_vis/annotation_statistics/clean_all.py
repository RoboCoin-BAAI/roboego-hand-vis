"""Clean annotations and existing readmes in one fixed-depth pass."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .assign_common_tasks import assign_document, load_mapping
from .clean_actions import clean_document as clean_actions
from .clean_objects import clean_document as clean_objects
from .clean_scenes import clean_document as clean_scenes
from .readme_sync import clean_readme_field
from .run import find_annotations


def load_object(path):
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def write_object(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clean_all(root, mapping, rules, pattern="episode_000000_structured.json"):
    # Discovery happens once. Only batch/session/annotation levels are enumerated.
    paths = find_annotations(root.resolve(), pattern)
    sessions = defaultdict(list)
    for path in paths:
        sessions[path.parent.parent].append(path)
    stats = dict(annotations_read=0, annotations_modified=0, readmes_read=0,
                 readmes_modified=0, unmatched_tasks=0, failed_sessions=0)
    cleaners = [("actions", clean_actions), ("objects", clean_objects), ("scene", clean_scenes)]
    for session, annotations in sessions.items():
        try:
            pending = []
            tasks = set()
            unmatched = 0
            for path in annotations:
                original = load_object(path)
                stats["annotations_read"] += 1
                cleaned = original
                for field, cleaner in cleaners:
                    cleaned = cleaner(cleaned, rules[field])
                cleaned = assign_document(cleaned, mapping)
                tasks.add(cleaned["common_task"])
                unmatched += cleaned["common_task"] is None
                if cleaned != original:
                    pending.append((path, cleaned, "annotations_modified"))
            readme_path = session / "meta/readme.json"
            if readme_path.is_file():
                original = load_object(readme_path)
                stats["readmes_read"] += 1
                if len(tasks) != 1:
                    raise ValueError("Multiple episode tasks cannot share a single readme.common_task")
                cleaned = original
                for field, cleaner in cleaners:
                    cleaned = clean_readme_field(cleaned, cleaner, rules[field], field)
                cleaned["common_task"] = next(iter(tasks))
                if cleaned != original:
                    pending.append((readme_path, cleaned, "readmes_modified"))
            # All transformations for this session succeed before writing any file.
            for path, document, counter in pending:
                write_object(path, document)
                stats[counter] += 1
            stats["unmatched_tasks"] += unmatched
        except (OSError, ValueError) as error:
            stats["failed_sessions"] += 1
            print(f"Error: {session}: {error}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--dataset", help="Dataset column filter for combined mappings")
    parser.add_argument("--rules-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--pattern", default="episode_000000_structured.json")
    args = parser.parse_args()
    mapping = load_mapping(args.mapping, args.dataset)
    rules = {field: load_object(args.rules_dir / filename) for field, filename in [
        ("actions", "nonverb_rules.json"), ("objects", "object_rules.json"), ("scene", "scene_rules.json")]}
    replacements = rules["actions"].get("replace", {})
    deleted = rules["actions"]["delete"]
    if any(key in deleted or value in deleted or value in replacements for key, value in replacements.items()):
        parser.error("Action mapping contains deletion conflicts or replacement chains")
    stats = clean_all(args.input_root, mapping, rules, args.pattern)
    print(json.dumps(stats, indent=2))
    return int(stats["failed_sessions"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
