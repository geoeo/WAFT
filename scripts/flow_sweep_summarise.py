#!/usr/bin/env python3
"""Tables from the results of scripts/flow_sweep.py.

    python scripts/flow_sweep_summarise.py [--output_dir outputs/flow_sweep] [--reference 640x480_t5]

Prints, per size, iteration count and precision, the mean over the pairs of each figure,
split by the gap between the two images of a pair, followed by the timing table. All
pixel figures are in 448-equivalent pixels (see flow_sweep.py).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def mean(rows, name):
    values = [r[name] for r in rows if name in r and r[name] == r[name]]
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "flow_sweep")
    parser.add_argument("--reference", default="640x480_t5")
    args = parser.parse_args()

    pairs = {p["name"]: p for p in json.loads((args.output_dir / "frames" / "pairs.json").read_text())}
    reference = json.loads((args.output_dir / f"reference_{args.reference}" / "summary.json").read_text())
    print(f"## Reference {args.reference} itself\n")
    print("| gap | pairs | flow px | alpha1 mean | pairs with alpha1 < 0.5 | epipolar median px | epipolar p90 px |")
    print("|---|---|---|---|---|---|---|")
    for gap in sorted({p["gap_frames"] for p in pairs.values()}):
        rows = [row for name, row in reference.items() if pairs[name]["gap_frames"] == gap]
        low = sum(row["alpha1_mean"] < 0.5 for row in rows)
        print(f"| {gap} | {len(rows)} | {mean(rows, 'flow_mean'):.1f} | {mean(rows, 'alpha1_mean'):.3f} | {low} "
              f"| {mean(rows, 'epi_median'):.3f} | {mean(rows, 'epi_p90'):.3f} |")

    sweep = json.loads((args.output_dir / f"sweep_ref-{args.reference}.json").read_text())
    groups = defaultdict(list)
    order = []
    for row in sweep:
        key = (row["size"], row["precision"], row["iters"])
        if key not in groups:
            order.append(key)
        groups[key].append(row)

    for gap in sorted({row["gap_frames"] for row in sweep}):
        print(f"\n## Sweep, gap {gap} frames, reference {args.reference}\n")
        print("| size | precision | T | non-finite | EPE mean | EPE median | EPE p90 | EPE > 1 px | "
              "alpha1 mean | alpha1 >= 0.5 | alpha1 >= 0.9 | alpha1 shift | pairs alpha1 < 0.5 | "
              "epipolar median | epipolar p90 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in order:
            rows = [r for r in groups[key] if r["gap_frames"] == gap]
            good = [r for r in rows if r["finite"]]
            low = sum(r["alpha1_mean"] < 0.5 for r in good)
            print(f"| {key[0]} | {key[1]} | {key[2]} | {len(rows) - len(good)} "
                  f"| {mean(good, 'epe_mean'):.3f} | {mean(good, 'epe_median'):.3f} | {mean(good, 'epe_p90'):.3f} "
                  f"| {mean(good, 'epe_gt_1px'):.1%} | {mean(good, 'alpha1_mean'):.3f} "
                  f"| {mean(good, 'alpha1_ge_0.5'):.1%} | {mean(good, 'alpha1_ge_0.9'):.1%} "
                  f"| {mean(good, 'alpha1_shift'):+.3f} | {low} "
                  f"| {mean(good, 'epi_median'):.3f} | {mean(good, 'epi_p90'):.3f} |")

    timing = json.loads((args.output_dir / "timing.json").read_text())
    print("\n## Timing (development GPU)\n")
    print("| size | precision | T | ms/pair batch 1 | ms/pair batch 2 | peak MiB batch 1 | peak MiB batch 2 | "
          "batch 2 vs 1 EPE mean | max |")
    print("|---|---|---|---|---|---|---|---|---|")
    table = defaultdict(dict)
    order = []
    for row in timing:
        key = (row["size"], row["precision"], row["iters"])
        if key not in table:
            order.append(key)
        table[key][row["batch"]] = row
    for key in order:
        one, two = table[key].get(1, {}), table[key].get(2, {})
        number = lambda row, name, fmt: format(row[name], fmt) if name in row else "-"
        print(f"| {key[0]} | {key[1]} | {key[2]} | {number(one, 'ms_per_pair_median', '.0f')} "
              f"| {number(two, 'ms_per_pair_median', '.0f')} | {number(one, 'peak_memory_mib', '.0f')} "
              f"| {number(two, 'peak_memory_mib', '.0f')} | {number(two, 'batch_vs_single_epe_mean', '.1e')} "
              f"| {number(two, 'batch_vs_single_epe_max', '.1e')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
