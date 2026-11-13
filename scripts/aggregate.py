"""Collect final_metrics.json over seeds into a paper-vs-ours table (mean +- std).

  python scripts/aggregate.py runs/
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# One-shot "Ours" numbers (ALL, PRIV) from the paper, for side-by-side comparison.
PAPER = {
    ("office_a2d", "1shot", "none"): (72.2, 72.6), ("office_a2w", "1shot", "none"): (62.1, 62.0),
    ("office_d2a", "1shot", "none"): (62.6, 61.8), ("office_d2w", "1shot", "none"): (78.5, 78.7),
    ("office_w2a", "1shot", "none"): (62.1, 62.4), ("office_w2d", "1shot", "none"): (87.8, 87.6),
    ("office_a2d", "few5", "none"): (78.8, 74.3), ("office_a2w", "few5", "none"): (70.1, 69.8),
    ("office_d2a", "few5", "none"): (66.9, 67.1), ("office_d2w", "few5", "none"): (85.0, 84.6),
    ("office_w2a", "few5", "none"): (67.2, 65.3), ("office_w2d", "few5", "none"): (90.4, 90.8),
    ("office_a2d", "1shot", "a"): (67.4, 64.1), ("office_a2d", "1shot", "b"): (68.4, 70.2),
    ("office_a2d", "1shot", "c"): (70.0, 69.3),
    ("office_a2d_nols1", "1shot", "none"): (68.6, 70.4), ("office_a2d_noneg", "1shot", "none"): (67.4, 62.3),
    ("office20_a2d", "1shot", "none"): (73.3, 73.1), ("office20_a2w", "1shot", "none"): (63.6, 62.6),
    ("office20_d2a", "1shot", "none"): (64.1, 64.3), ("office20_d2w", "1shot", "none"): (80.3, 79.4),
    ("office20_w2a", "1shot", "none"): (63.7, 62.8), ("office20_w2d", "1shot", "none"): (89.5, 88.4),
    ("digits_s2m", "1shot", "none"): (46.4, 45.7), ("digits_m2u", "1shot", "none"): (72.5, 71.6),
    ("digits_u2m", "1shot", "none"): (69.4, 68.6),
    ("visda_sy2re", "1shot", "none"): (56.4, 56.3), ("visda_re2sy", "1shot", "none"): (55.8, 55.7),
}


def fmt(xs):
    return f"{st.mean(xs):5.1f} ± {st.stdev(xs):4.1f}" if len(xs) > 1 else f"{xs[0]:5.1f}"


def main(root):
    groups = defaultdict(list)
    for f in Path(root).glob("*/adapt_*/final_metrics.json"):
        m = re.match(r"adapt_(\w+?)_abl-(\w+?)(_.+)?_seed\d+$", f.parent.name)
        task = f.parent.parent.name
        groups[(task, m.group(1), m.group(2) + (m.group(3) or ""))].append(json.load(open(f)))
    wt = max([4] + [len(k[0]) for k in groups])
    wa = max([3] + [len(k[2]) for k in groups])
    print(f"{'task':{wt}} {'shots':6} {'abl':{wa}} {'n':>2} | {'ALL':>13} {'PRIV':>13} | {'ALL(cls)':>13} | paper ALL/PRIV")
    for key in sorted(groups):
        rs = groups[key]
        paper = PAPER.get(key)
        print(f"{key[0]:{wt}} {key[1]:6} {key[2]:{wa}} {len(rs):>2} | {fmt([r['all'] for r in rs]):>13} "
              f"{fmt([r['priv'] for r in rs]):>13} | {fmt([r['all_cls'] for r in rs]):>13} | "
              f"{'%.1f / %.1f' % paper if paper else '-'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
