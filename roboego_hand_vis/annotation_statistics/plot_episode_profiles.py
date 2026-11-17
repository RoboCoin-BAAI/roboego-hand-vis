"""Actual and illustrative target episode-duration / subtask-count distributions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DURATION_EDGES = list(range(10, 301, 10))
DURATION_LABELS = [f"{end-10}-{end} s" for end in DURATION_EDGES] + [">300 s"]
SUBTASK_LABELS = ["0"] + [str(count) for count in range(1, 31)] + [">30"]
TARGET_HOURS = {"L1": 10000, "L2": 2000, "L3": 500}
COLORS = ["#85BFFC", "#6CC8C5", "#F3CC81"]


def duration_bin(seconds):
    return int(np.searchsorted(DURATION_EDGES, seconds, side="left"))


def subtask_bin(count):
    return min(count, len(SUBTASK_LABELS)-1)


def summarize(records, weights):
    durations = np.zeros(len(DURATION_LABELS))
    subtasks = np.zeros(len(SUBTASK_LABELS))
    exact = {}
    for (seconds, count), weight in zip(records, weights):
        durations[duration_bin(seconds)] += weight
        subtasks[subtask_bin(count)] += weight
        exact[count] = exact.get(count, 0) + float(weight)
    total = float(sum(weights))
    seconds = float(sum(record[0]*weight for record, weight in zip(records, weights)))
    return {"duration": durations, "subtask": subtasks, "exact": exact,
            "count": total, "hours": seconds/3600, "mean_s": seconds/total}


def build_profiles(first, third, ideal=False, guanglun_subtask_factor=1):
    if not isinstance(guanglun_subtask_factor, int) or guanglun_subtask_factor < 1:
        raise ValueError("Guanglun subtask factor must be a positive integer")
    def weights(records, level):
        if not ideal:
            return np.ones(len(records))
        # Modest, explicit hypothetical curation, rather than invented observations.
        duration_weights = [.3, .9, 1.1, 1.15, 1.2, 1.1, .85, .5]
        values = np.array([duration_weights[int(np.searchsorted([10, 20, 30, 50, 120, 300, 600], seconds, side="right"))] * (
            0 if count == 0 else .9 if count <= 2 else 1.1 if count <= 8 else 1 if count <= 15 else .8)
            for seconds, count in records])
        denominator = sum(record[0]*w for record, w in zip(records, values))
        if denominator <= 0:
            raise ValueError("No usable episodes for target projection")
        return values*TARGET_HOURS[level]*3600/denominator
    w1 = weights(first, "L1")
    w3 = weights(third, "L3")
    # Adjust only the displayed segment-count proxy, after weighting. Duration,
    # episode weights and target volume must not change with this convention.
    first = [[seconds, count*guanglun_subtask_factor] for seconds, count in first]
    s1, s3 = summarize(first, w1), summarize(third, w3)
    scale = s3["hours"]/s1["hours"]
    s2 = summarize(first+third, np.concatenate([w1*scale, w3*3]))
    return {"L1": s1, "L2": s2, "L3": s3}


def render(profiles, root, ideal, guanglun_subtask_factor=1):
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(2, 1, left=.07, right=.97, bottom=.12, top=.79, hspace=.55)
    for ax, metric, labels, title in [
        (fig.add_subplot(grid[0]), "duration", DURATION_LABELS, "(a) Episode duration"),
        (fig.add_subplot(grid[1]), "subtask", SUBTASK_LABELS,
         f"(b) Segments per episode / Guanglun count x{guanglun_subtask_factor} (estimated adjustment)")]:
        x = np.arange(len(labels))
        for i, (level, color) in enumerate(zip(["L1", "L2", "L3"], COLORS)):
            stats = profiles[level]
            shares = stats[metric]/stats["count"]*100
            bars = ax.bar(x+(i-1)*.24, shares, width=.22, color=color, label=level)
            for bar, share in zip(bars, shares):
                if share >= 1:
                    ax.text(bar.get_x()+bar.get_width()/2, share+.5, f"{share:.1f}",
                            ha="center", va="bottom", fontsize=8, color="#45596E")
        ax.set_xticks(x, labels, rotation=55 if metric == "duration" else 0, ha="right" if metric == "duration" else "center", fontsize=8)
        ax.set_ylabel("Share of episodes (%)")
        ax.set_title(title, loc="left", fontweight="bold", color="#20374F")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1]*1.12)
        ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.text(.07, .95, "Illustrative target episode profiles" if ideal else "Episode profiles / adjusted segmentation",
             fontsize=24, fontweight="bold", color="#20374F")
    fig.text(.07, .908, "Denominator: number of episodes, not duration. L2 is a duration-weighted synthetic mixture.",
             fontsize=11, color="#697B8F")
    for i, level in enumerate(["L1", "L2", "L3"]):
        stats = profiles[level]
        count_label = "expected episodes" if ideal or level == "L2" else "observed episodes"
        fig.text(.07+i*.30,.855,
                 f"{level}: {stats['count']:,.0f} {count_label}\n{stats['hours']:,.2f} h  /  mean {stats['mean_s']:.1f} s",
                 fontsize=11, color="#20374F", linespacing=1.5)
    note = ("ILLUSTRATIVE PLAN: reduced weight for very short/long episodes; no zero-subtask episodes. Target hours: L1 10,000 / L2 2,000 / L3 500."
            if ideal else "L1 = Guanglun; L3 = L3 data. L2 = 3 x L3 + Guanglun scaled to L3 total duration. No source annotations modified.")
    note += f"\nSegmentation proxy only: Guanglun raw count x{guanglun_subtask_factor}; L3 unchanged. This does not establish equivalent annotation granularity."
    fig.text(.07, .025, note, fontsize=8.5, color="#697B8F", wrap=True)
    stem = "episode_profiles_ideal" if ideal else "episode_profiles_actual"
    for suffix in ("pdf", "png"):
        try:
            fig.savefig(root/f"{stem}.{suffix}", dpi=180, facecolor="white")
        except OSError as error:
            if suffix != "png":
                raise
            print(f"Preview could not be overwritten: {stem}.png: {error}")
    plt.close(fig)


def export_chart_values(path, actual, ideal):
    """Export a standalone CSV for each figure."""
    for suffix, profiles in [("actual", actual), ("ideal", ideal)]:
        destination = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["panel", "bin", "level", "count", "percentage"])
            for metric, labels in [("duration", DURATION_LABELS), ("subtask", SUBTASK_LABELS)]:
                for label_index, label in enumerate(labels):
                    for level in ["L1", "L2", "L3"]:
                        stats = profiles[level]
                        count = float(stats[metric][label_index])
                        writer.writerow([metric, label, level, count, count/stats["count"]*100])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--guanglun-subtask-factor", type=int, default=2)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    for data in source.values():
        if data["errors"]:
            raise ValueError(f"Source contains errors: {data['errors'][:2]}")
    first, third = source["guanglun_depth"]["records"], source["L3_lerobot"]["records"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    factor = args.guanglun_subtask_factor
    actual = build_profiles(first, third, guanglun_subtask_factor=factor)
    ideal = build_profiles(first, third, ideal=True, guanglun_subtask_factor=factor)
    for ideal_flag, profiles in [(False, actual), (True, ideal)]:
        render(profiles, args.output_root, ideal_flag, factor)
    export_chart_values(args.output_root/"episode_profile_values.csv", actual, ideal)
    for scenario, profiles in [("actual", actual), ("illustrative_target", ideal)]:
        print(scenario)
        for level, stats in profiles.items():
            print(level, "episodes", round(stats["count"], 2), "hours", round(stats["hours"], 4),
                  "mean_s", round(stats["mean_s"], 2), "most_common_subtasks", sorted(stats["exact"].items(),key=lambda item:-item[1])[:5])
    print("Same-distribution extrapolation to target hours:")
    for level, stats in actual.items():
        print(level, round(TARGET_HOURS[level]*3600/stats["mean_s"]))


if __name__ == "__main__":
    main()
