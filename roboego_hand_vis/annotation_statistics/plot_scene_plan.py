"""Illustrative collection plan, explicitly separate from measured statistics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    totals = [10000, 2000, 500]
    # User-approved hypothetical planning; none of these allocations are measurements.
    allocation = {
        "household": [45, 30, 25],
        "commercial_service": [25, 25, 15],
        "professional_work": [12, 20, 30],
        "industrial_production": [18, 25, 30],
    }
    areas = {
        "household": {"household_cleaning": 14, "household_storage": 14, "kitchen": 18,
                      "bedroom": 13, "living_room": 15, "dining_area": 13, "bathroom": 13},
        "commercial_service": {"retail_store": 22, "packing_area": 20, "shelving_area": 20,
                               "sorting_area": 16, "loading_area": 10, "food_preparation_area": 12},
        "professional_work": {"office": 100},
        "industrial_production": {"workshop": 30, "assembly": 24, "maintenance": 20,
                                  "inspection": 16, "machining": 10},
    }
    rows = []
    for category, weights in areas.items():
        budgets = [round(total * pct / 100) for total, pct in zip(totals, allocation[category])]
        values = {area: [0, 0, 0] for area in weights}
        for level, budget in enumerate(budgets):
            raw = [budget * weight / 100 for weight in weights.values()]
            assigned = [int(value) for value in raw]
            order = sorted(range(len(raw)), key=lambda i: raw[i]-assigned[i], reverse=True)
            for i in order[:budget-sum(assigned)]:
                assigned[i] += 1
            for area, hours in zip(weights, assigned):
                values[area][level] = hours
        for area, hours in values.items():
            rows.append([category, area, *hours, sum(hours), *[100*h/sum(hours) for h in hours]])
    assert [sum(row[2+i] for row in rows) for i in range(3)] == totals
    write_csv(root/'ideal_scene_plan.csv',
              ['category', 'area', 'L1_planned_hours', 'L2_planned_hours', 'L3_planned_hours',
               'total_planned_hours', 'L1_share_pct', 'L2_share_pct', 'L3_share_pct'], rows)
    colors = ['#83BFFA', '#6AC7C7', '#F2CC80']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, (cat_ax, area_ax) = plt.subplots(1, 2, figsize=(17, 10),
                                         gridspec_kw={'width_ratios': [1, 1.8]}, layout='constrained')
    names = list(allocation)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for level, color in enumerate(colors):
        values = [totals[level]*allocation[category][level]/100 for category in names]
        cat_ax.bar(x, values, bottom=bottom, color=color, width=.65,
                   label=f'L{level+1}: {totals[level]:,} h (planned)')
        bottom += values
    for position, total in zip(x, bottom):
        cat_ax.text(position, total+80, f'{total:,.0f} h', ha='center', fontweight='bold')
    cat_ax.set_xticks(x, [name.replace('_', '\n') for name in names])
    cat_ax.set_ylim(0, max(bottom)*1.13)
    cat_ax.set_ylabel('Planned collection hours')
    cat_ax.set_title('(a) Category budgets', loc='left', fontweight='bold')
    cat_ax.legend(frameon=False, loc='upper right')
    cat_ax.grid(axis='y', alpha=.15)
    cat_ax.set_axisbelow(True)
    ticks, labels, headers, y, ordered = [], [], [], [], []
    for category in names:
        headers.append(len(ticks))
        ticks.append(len(ticks))
        labels.append(category.replace('_', ' ').upper())
        for row in rows:
            if row[0] != category:
                continue
            y.append(len(ticks))
            ticks.append(len(ticks))
            labels.append('    '+row[1].replace('_', ' '))
            ordered.append(row)
    left = np.zeros(len(ordered))
    for level, color in enumerate(colors):
        values = np.array([row[6+level] for row in ordered])
        area_ax.barh(y, values, left=left, color=color, height=.72)
        left += values
    area_ax.set_yticks(ticks, labels)
    area_ax.invert_yaxis()
    for header in headers:
        area_ax.get_yticklabels()[header].set_fontweight('bold')
        area_ax.axhspan(header-.45, header+.45, color='#F0F3F6', zorder=0)
    for position, row in zip(y, ordered):
        area_ax.text(101, position, f'{row[5]:,} h', fontsize=8, va='center', color='#555555')
    area_ax.set_xlim(0, 117)
    area_ax.set_xticks([0, 20, 40, 60, 80, 100])
    area_ax.set_xlabel('Planned L1 / L2 / L3 share within each area (%)')
    area_ax.set_title('(b) Category > area: illustrative allocation', loc='left', fontweight='bold')
    fig.suptitle('IDEAL COLLECTION PLAN - NOT MEASURED DATA\n'
                 'L1: 10,000 h | L2: 2,000 h | L3: 500 h', fontsize=17, fontweight='bold')
    fig.supxlabel('Illustrative targets only. No measured counts, source annotations or existing distribution plots were modified.',
                  fontsize=10, color='#555555')
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(root/f'ideal_scene_plan.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    metadata = {'status': 'hypothetical collection plan; not measured', 'planned_hours': dict(zip(['L1','L2','L3'],totals)),
                'category_percent_within_level': allocation, 'areas': len(rows),
                'notes': ['Office is currently the sole named professional_work area in the input statistics.',
                          'Other and unknown are excluded from planned targets; actual statistics are unchanged.',
                          'L2/L3 intentionally emphasize professional and industrial work; no unsupported action/object counts are invented.']}
    (root/'ideal_scene_plan_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
