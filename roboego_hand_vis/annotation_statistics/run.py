"""Collect exact annotation values without changing source files (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = ("task_name", "common_task", "scene", "category", "area", "object_name", "action", "action_vocabulary")


def strings(value, location, issues):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                result.append(item)
            else:
                issues.append(f"{location}[{index}]: expected string")
        return result
    issues.append(f"{location}: expected string or list of strings")
    return []


def extract(document):
    issues = []
    values = {field: [] for field in FIELDS}
    for field in ("task_name", "common_task", "scene"):
        value = document.get(field)
        if isinstance(value, str):
            values[field] = [value]
        elif value is not None:
            issues.append(f"{field}: expected string")
    objects = document.get("objects")
    if isinstance(objects, list):
        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                issues.append(f"objects[{index}]: expected object")
                continue
            name = obj.get("name")
            if isinstance(name, str):
                values["object_name"].append(name)
            elif name is not None:
                issues.append(f"objects[{index}].name: expected string")
    elif objects is not None:
        issues.append("objects: expected list")
    classification = document.get("scene_classification")
    if isinstance(classification, dict):
        for field in ("category", "area"):
            value = classification.get(field)
            if isinstance(value, str):
                values[field] = [value]
            elif value is not None:
                issues.append(f"scene_classification.{field}: expected string")
    elif classification is not None:
        issues.append("scene_classification: expected object")
    subtasks = document.get("subtask")
    if subtasks is not None and not isinstance(subtasks, list):
        issues.append("subtask: expected list")
    elif isinstance(subtasks, list):
        for index, subtask in enumerate(subtasks):
            if not isinstance(subtask, dict):
                issues.append(f"subtask[{index}]: expected object")
                continue
            values["action"].extend(strings(subtask.get("action"), f"subtask[{index}].action", issues))
    actions = document.get("actions")
    if isinstance(actions, dict):
        values["action_vocabulary"] = strings(actions.get("vocabulary"), "actions.vocabulary", issues)
    elif actions is not None:
        issues.append("actions: expected object")
    return values, issues


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def find_annotations(root: Path, pattern: str):
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    if Path(pattern).name != pattern or "/" in pattern or "\\" in pattern:
        raise ValueError("--pattern must be a filename glob without directory separators")
    # Guanglun has a fixed layout. Never descend into videos/data/chunks.
    if root.name == "annotation":
        annotation_dirs = [root]
    elif (root / "annotation").is_dir():
        annotation_dirs = [root / "annotation"]
    else:
        batches = [root] if root.name.startswith("batch_") else sorted(
            path for path in root.glob("batch_*") if path.is_dir())
        annotation_dirs = [session / "annotation" for batch in batches
                           for session in batch.iterdir()
                           if session.is_dir() and (session / "annotation").is_dir()]
    paths = sorted(path for directory in annotation_dirs for path in directory.glob(pattern)
                   if path.is_file())
    if not paths:
        raise ValueError(f"No annotation/{pattern} files found under {root}")
    return paths


def collect(root: Path, output: Path, pattern: str):
    root = root.resolve()
    output = output.resolve()
    paths = find_annotations(root, pattern)
    if any(output == path.parent or output in path.parents for path in paths):
        raise ValueError("Output must be a separate report directory, not an annotation ancestor")
    output.mkdir(parents=True, exist_ok=True)
    occurrences = {field: Counter() for field in FIELDS}
    episodes = {field: Counter() for field in FIELDS}
    missing = Counter()
    errors = []
    valid = 0
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            try:
                document = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(document, dict):
                    raise ValueError("JSON root must be an object")
            except (OSError, ValueError) as error:
                errors.append((relative, "read_error", str(error)))
                continue
            values, issues = extract(document)
            valid += 1
            errors.extend((relative, "field_issue", issue) for issue in issues)
            for field, items in values.items():
                occurrences[field].update(items)
                episodes[field].update(set(items))
                if not items or all(not item.strip() for item in items):
                    missing[field] += 1
            handle.write(json.dumps({"file": relative, **values}, ensure_ascii=False) + "\n")
    for field in FIELDS:
        ordered = sorted(occurrences[field], key=lambda value: (-occurrences[field][value], value))
        write_csv(output / f"{field}_counts.csv", ("value", "occurrences", "episode_count"),
                  ((value, occurrences[field][value], episodes[field][value]) for value in ordered))
        # Never overwrite a mapping a user may already have edited.
        mapping = output / f"{field}_mapping.csv"
        if not mapping.exists():
            write_csv(mapping, ("original", "canonical", "occurrences", "episode_count"),
                      ((value, "", occurrences[field][value], episodes[field][value]) for value in ordered))
    write_csv(output / "errors.csv", ("file", "kind", "message"), errors)
    summary = {
        "input_root": str(root), "pattern": pattern, "files_found": len(paths),
        "files_read": valid, "files_failed": len(paths) - valid,
        "field_issues": sum(kind == "field_issue" for _, kind, _ in errors),
        "fields": {field: {"unique_values": len(occurrences[field]),
                           "occurrences": sum(occurrences[field].values()),
                           "missing_or_blank_files": missing[field]} for field in FIELDS},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="episode_000000_structured.json",
                        help="Filename glob; use episode_*_structured.json for all episode indices")
    args = parser.parse_args()
    try:
        summary = collect(args.input_root, args.output_dir, args.pattern)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["files_failed"] or summary["field_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
