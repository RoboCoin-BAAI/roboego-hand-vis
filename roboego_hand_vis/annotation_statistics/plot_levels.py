"""Plot L1=Guanglun, L3=measured L3, L2=2x L3 estimated counts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import write_csv


def read_counts(folder, field, column):
    with (folder / f"{field}_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["value"]: int(row[column]) for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.statistics_root
    l1 = root / "guanglun_depth"
    l3 = root / "L3_lerobot"
    summaries = [json.loads((folder / "summary.json").read_text(encoding="utf-8")) for folder in (l1, l3)]
    colors = ["#83BFFA", "#6AC7C7", "#F2CC80"]
    labels = ["L1 (measured)", "L2 (estimated: 2 x L3)", "L3 (measured)"]
    datasets = {}
    for field, column in [("category", "episode_count"), ("area", "episode_count"), ("action_vocabulary", "occurrences"), ("object_name", "occurrences")]:
        first, third = read_counts(l1, field, column), read_counts(l3, field, column)
        values = {name: [first.get(name, 0), 2 * third.get(name, 0), third.get(name, 0)]
                  for name in first.keys() | third.keys()}
        datasets[field] = values
        write_csv(root / f"levels_{field}_distribution.csv",
                  ["name", "L1_measured", "L2_estimated", "L3_measured", "total_with_estimate", "L1_pct", "L2_pct", "L3_pct"],
                  ([name, *counts, sum(counts), *[f"{100*n/sum(counts):.4f}" for n in counts]]
                   for name, counts in sorted(values.items(), key=lambda item: (-sum(item[1]), item[0]))))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    # Read actual category/area assignments, including the approved extended areas.
    owners = {}
    for folder in (l1, l3):
        with (folder / "episodes.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if len(row.get("category", [])) == len(row.get("area", [])) == 1:
                    category, area = row["category"][0], row["area"][0]
                    if area in owners and owners[area] != category:
                        raise ValueError(f"Area belongs to multiple categories: {area}")
                    owners[area] = category
    if set(datasets["area"]) - owners.keys():
        raise ValueError("Missing category assignments for scene areas")
    fig = plt.figure(figsize=(17, 15), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.8, 1])
    scene_ax = fig.add_subplot(grid[0, :])
    scene_data = datasets["area"]
    categories = sorted(set(owners.values()), key=lambda category: (
        -sum(sum(counts) for area, counts in scene_data.items() if owners[area] == category), category))
    scene_names, positions, ticks, tick_labels, headers = [], [], [], [], []
    for category in categories:
        header = len(ticks)
        headers.append(header)
        ticks.append(header)
        tick_labels.append(category.replace("_", " ").upper())
        names = sorted((area for area in scene_data if owners[area] == category),
                       key=lambda name: (-sum(scene_data[name]), name))
        for name in names:
            positions.append(len(ticks))
            ticks.append(len(ticks))
            tick_labels.append("    " + name.replace("_", " "))
            scene_names.append(name)
    y = np.array(positions)
    left = np.zeros(len(y))
    for index, (color, label) in enumerate(zip(colors, labels)):
        shares = np.array([100*scene_data[name][index]/sum(scene_data[name]) for name in scene_names])
        scene_ax.barh(y, shares, left=left, color=color, label=label, height=.73)
        left += shares
    scene_ax.set_yticks(ticks, tick_labels)
    for header in headers:
        scene_ax.get_yticklabels()[header].set_fontweight("bold")
        scene_ax.axhspan(header - .45, header + .45, color="#F0F3F6", zorder=0)
    scene_ax.invert_yaxis()
    scene_ax.set_xlim(0, 119)
    scene_ax.set_xticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])
    for position, name in zip(positions, scene_names):
        scene_ax.text(101, position, f"n={sum(scene_data[name]):,}", va="center", fontsize=8, color="#555555")
    scene_ax.set_xlabel("Share of annotation files within each scene (%) - not hours")
    scene_ax.set_title(f"(a) Scenes: {len(categories)} categories / {len(scene_names)} areas | L1 / L2 / L3 composition", loc="left", fontweight="bold")
    scene_ax.legend(loc="upper right", bbox_to_anchor=(1, 1.055), frameon=False, ncol=3, fontsize=8)
    write_csv(root / "levels_area_distribution.csv",
              ["category", "area", "L1_measured", "L2_estimated", "L3_measured", "total_with_estimate", "L1_pct", "L2_pct", "L3_pct"],
              ([owners[name], name, *scene_data[name], sum(scene_data[name]),
                *[f"{100*n/sum(scene_data[name]):.4f}" for n in scene_data[name]]] for name in scene_names))
    for axis, field, panel, title in [
        (fig.add_subplot(grid[1, 0]), "action_vocabulary", "b", "Actions"),
        (fig.add_subplot(grid[1, 1]), "object_name", "c", "Objects")]:
        data = datasets[field]
        names = sorted(data, key=lambda name: (-sum(data[name]), name))[:15]
        bottom = np.zeros(len(names))
        x = np.arange(len(names))
        for index, color in enumerate(colors):
            values = np.array([data[name][index] for name in names])
            axis.bar(x, values, bottom=bottom, color=color, width=.73)
            bottom += values
        axis.set_xticks(x, [name.replace("_", " ") for name in names], rotation=55, ha="right")
        axis.set_ylabel("Annotation occurrences (includes estimated L2)")
        axis.set_title(f"({panel}) {title}: {len(data):,} unique names | top 15", loc="left", fontweight="bold")
        axis.ticklabel_format(axis="y", style="plain")
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    counts = [summaries[0]["files_read"], 2*summaries[1]["files_read"], summaries[1]["files_read"]]
    fig.suptitle("Dataset distributions: L1 = Guanglun; L2 = 2 x L3 (estimated)\n"
                 f"Annotation files: L1 {counts[0]:,} | L2 {counts[1]:,} (estimated) | L3 {counts[2]:,}",
                 fontsize=16, fontweight="bold")
    for suffix in ("png", "pdf", "svg"):
        path = root / f"levels_distribution.{suffix}"
        try:
            fig.savefig(path, dpi=220, bbox_inches="tight")
        except PermissionError:
            path = root / f"levels_distribution_category_area.{suffix}"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            print(f"Original export is locked; saved to {path}")
    plt.close(fig)
    note = {
        "L1": {"source": "guanglun_depth", "files": counts[0], "measured": True},
        "L2": {"source": "L3_lerobot multiplied by 2", "files": counts[1], "measured": False},
        "L3": {"source": "L3_lerobot", "files": counts[2], "measured": True},
        "scene_metric": "Within-area share of annotation files; not duration share. Scene labels are not forced into the reference image's 18 classes.",
        "scene_hierarchy": "category -> area; grouped by actual annotations, ordered by estimated combined file count",
        "action_metric": "actions.vocabulary occurrences (file-label incidences); not physical action repetitions.",
        "object_metric": "objects[].name occurrences (annotated entries); not verified unique physical instances.",
        "unique_counts": {key: len(values) for key, values in datasets.items()},
        "note": "L2 is an estimate only and adds no unique labels. Action/object charts show top 15; CSVs contain all labels."
    }
    (root / "levels_distribution_metadata.json").write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
