"""Build reviewable task naming mappings from statistics; never modify annotations."""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .run import write_csv


def suggest(title, rules):
    if "coarse_object_groups" in rules:
        return suggest_coarse(title, rules)
    text = title.lower().strip()
    goals = []
    for name, pattern in rules["goals"].items():
        for match in re.finditer(r"\b(?:" + pattern + r")\b", text):
            goals.append((match.start(), -len(match.group()), name, match.end()))
    goals.sort()
    if not goals or goals[0][0] > 12:
        return "", "unmatched_goal", ""
    _, _, goal, end = goals[0]
    # Tool/source/destination details are excluded from the object search.
    target = re.split(r"\b(?:with|using|into|onto|from|for)\b", text[end:], maxsplit=1)[0]
    objects = []
    for name, pattern in rules["objects"].items():
        for match in re.finditer(r"\b(?:" + pattern + r")\b", target):
            objects.append((match.start(), -len(match.group()), name))
    objects.sort()
    if not objects:
        return "", "unmatched_object", goal
    _, _, obj = objects[0]
    # Prefer the longest term at a shared position (candy wrapper over candy).
    selected_end = objects[0][0] - objects[0][1]
    multiple_objects = any(start >= selected_end and name != obj for start, _, name in objects)
    other_goals = {name for _, _, name, _ in goals} - {goal}
    reasons = []
    if multiple_objects:
        reasons.append("multiple_objects")
    if other_goals:
        reasons.append("multiple_goals")
    return f"{goal} {obj}", ";".join(reasons) or "single_goal_object", f"{goal} | {obj}"


def suggest_coarse(title, rules):
    if title in rules.get("task_overrides", {}):
        return rules["task_overrides"][title], "explicit_mapping", title
    goals = dict(rules["goals"])
    for goal, pattern in rules["extra_goal_patterns"].items():
        goals[goal] = goals.get(goal, r"(?!)") + "|" + pattern
    # One shared pattern per broad object class, including known fine object names.
    objects = dict(rules["extra_object_patterns"])
    for group, names in rules["coarse_object_groups"].items():
        patterns = [rules["objects"][name] for name in names]
        objects[group] = "|".join([objects.get(group, r"(?!)"), *patterns])
    merged = {"goals": goals, "objects": objects}
    canonical, reason, evidence = suggest(title, merged)
    if not canonical and reason == "unmatched_object":
        # Noun-first titles such as "Desk cleanup and organization".
        text = title.lower()
        matches = []
        for group, pattern in objects.items():
            match = re.search(r"\b(?:" + pattern + r")\b", text)
            if match:
                matches.append((match.start(), -len(match.group()), group))
        if matches and evidence:
            matches.sort()
            canonical = evidence + " " + matches[0][2]
            reason = "whole_title_object_review"
    if canonical:
        # Position, folding and hanging are methods of organizing these objects.
        goal, obj = canonical.split(" ", 1)
        if obj in {"clothing", "household linen", "cable", "rope"} and goal in {
            "fold", "hang", "button", "tie", "wrap", "place", "transfer", "pack", "unfold"}:
            canonical = "organize " + obj
        elif goal in {"place", "transfer", "push", "lay"}:
            canonical = "organize " + obj
    return canonical, reason, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=Path(__file__).with_name("task_group_rules.json"))
    args = parser.parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    # Separate generated suggestions from any manually edited mapping files.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    groups = defaultdict(lambda: [0, 0, set(), []])
    summary = {}
    for source in sorted(args.statistics_root.glob("*/task_name_counts.csv")):
        dataset = source.parent.name
        total = mapped = episodes = mapped_episodes = 0
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                original = item["value"]
                count = int(item["episode_count"])
                canonical, reason, evidence = suggest(original, rules)
                total += 1
                episodes += count
                mapped += bool(canonical)
                mapped_episodes += count if canonical else 0
                row = [dataset, original, canonical, "candidate" if canonical else "unmatched", reason, count, evidence]
                rows.append(row)
                if canonical:
                    group = groups[canonical]
                    group[0] += count
                    group[1] += 1
                    group[2].add(dataset)
                    if len(group[3]) < 5:
                        group[3].append(original)
        summary[dataset] = dict(original_names=total, candidate_names=mapped, unmatched_names=total-mapped,
                                files=episodes, candidate_files=mapped_episodes)
    if not rows:
        raise ValueError("No task_name_counts.csv files found")
    headers = ["dataset", "original", "canonical", "status", "reason", "episode_count", "rule_match"]
    write_csv(args.output_dir / "task_mapping_candidates.csv", headers, rows)
    write_csv(args.output_dir / "task_unmatched.csv", headers, (row for row in rows if not row[2]))
    for dataset in summary:
        write_csv(args.output_dir / f"{dataset}_task_mapping.csv", headers, (row for row in rows if row[0] == dataset))
    write_csv(args.output_dir / "task_catalog_candidates.csv",
              ["canonical", "episode_count", "original_name_count", "datasets", "examples"],
              ([name, data[0], data[1], ";".join(sorted(data[2])), " | ".join(data[3])]
               for name, data in sorted(groups.items(), key=lambda pair: (-pair[1][0], pair[0]))))
    summary["candidate_groups"] = len(groups)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
