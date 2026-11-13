"""Per-epoch metrics averaged over seeds, one block per variant of a task.

  python scripts/trajectory.py runs/digits_m2u                 # all variants
  python scripts/trajectory.py runs/digits_m2u --epochs 0 1 2 3 5 10 15
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

COLS = ["all", "priv", "shared", "shared_as_priv", "priv_as_shared", "pl_acc_all", "B_precision_priv", "B_min_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--epochs", nargs="*", type=int, default=[0, 1, 2, 3, 5, 8, 10, 15, 20, 30])
    args = ap.parse_args()
    variants = defaultdict(lambda: defaultdict(list))  # variant -> epoch -> [log rows]
    for log in sorted(Path(args.task_dir).glob("adapt_*/log.jsonl")):
        variant = re.sub(r"_seed\d+$", "", log.parent.name)
        rows = [json.loads(l) for l in open(log)]
        start = max(i for i, r in enumerate(rows) if r.get("event") == "labelled")  # last (re)start
        for r in rows[start:]:
            if "epoch" in r and "all" in r:
                variants[variant][r["epoch"]].append(r)
    for variant, by_epoch in variants.items():
        print(f"\n== {variant}  (mean over seeds)")
        print("epoch  n " + "".join(f"{c:>17}" for c in COLS))
        for e in args.epochs:
            rows = by_epoch.get(e)
            if not rows:
                continue
            vals = [st.mean(r[c] for r in rows if c in r) if any(c in r for r in rows) else float("nan") for c in COLS]
            print(f"{e:5d} {len(rows):2d} " + "".join(f"{v:17.1f}" for v in vals))


if __name__ == "__main__":
    main()
