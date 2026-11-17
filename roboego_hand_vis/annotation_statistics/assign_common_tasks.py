"""Add common_task from an explicitly supplied task mapping; preserve task_name."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .run import find_annotations
from .readme_sync import write_readme


def load_mapping(path, dataset=None):
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"original", "canonical"}.issubset(reader.fieldnames or []):
            raise ValueError("Mapping requires original and canonical columns")
        for row in reader:
            if dataset and row.get("dataset") and row["dataset"] != dataset:
                continue
            original, canonical = row["original"], row["canonical"].strip()
            if not canonical:
                continue
            if original in mapping and mapping[original] != canonical:
                raise ValueError(f"Conflicting mapping for {original!r}")
            mapping[original] = canonical
    if not mapping:
        raise ValueError("No nonempty task mappings found")
    return mapping


def assign_document(document, mapping):
    result = dict(document)
    # Unmatched titles are explicitly unassigned; never overwrite the original task_name.
    result["common_task"] = mapping.get(document.get("task_name"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--dataset", help="Filter the dataset column when using a combined CSV")
    args = parser.parse_args()
    mapping = load_mapping(args.mapping, args.dataset)
    paths = find_annotations(args.input_root.resolve(), "episode_*_structured.json")
    sessions = defaultdict(list)
    for path in paths:
        sessions[path.parent.parent].append(path)
    modified = readmes = unmatched = failed = 0
    for session, annotations in sessions.items():
        try:
            pending = []
            names = set()
            for path in annotations:
                document = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(document, dict):
                    raise ValueError(f"Annotation must be an object: {path}")
                cleaned = assign_document(document, mapping)
                names.add(cleaned["common_task"])
                pending.append((path, document, cleaned))
            readme_path = session / "meta/readme.json"
            prepared = None
            if readme_path.is_file():
                if len(names) > 1:
                    raise ValueError("Multiple episode tasks in one session; cannot assign a single readme.common_task")
                readme = json.loads(readme_path.read_text(encoding="utf-8-sig"))
                if not isinstance(readme, dict):
                    raise ValueError("readme must be an object")
                common_task = next(iter(names))
                if "common_task" not in readme or readme["common_task"] != common_task:
                    readme["common_task"] = common_task
                    prepared = (readme_path, readme)
            for path, original, cleaned in pending:
                unmatched += cleaned["common_task"] is None
                if original != cleaned:
                    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    modified += 1
            readmes += write_readme(prepared)
        except (OSError, ValueError) as error:
            failed += 1
            print(f"Error: {session}: {error}")
    print(f"Annotations modified: {modified}, readmes modified: {readmes}, unmatched: {unmatched}, failed sessions: {failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
