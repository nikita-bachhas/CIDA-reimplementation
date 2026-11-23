"""ALL / PRIV at the best epoch, at chosen epochs and at the last epoch, one row per task and variant.

  python scripts/epochs.py runs_vanilla/             # best, epoch 15, last
  python scripts/epochs.py runs/ --epochs 10 15 30

Values are means over seeds. "best" is the epoch with the highest mean ALL, picked with target
labels, so it is an upper bound that the setting does not allow, not a result.
"""
import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ORDER = ["digits_m2u", "digits_u2m", "digits_s2m",
         "office_a2d", "office_a2w", "office_d2a", "office_d2w", "office_w2a", "office_w2d",
         "office20_a2d", "office20_a2w", "office20_d2a", "office20_d2w", "office20_w2a", "office20_w2d",
         "visda_sy2re", "visda_re2sy"]


def load(task_dir):
    """variant -> epoch -> list of (all, priv), one entry per seed.

    A seed resumed from its checkpoint after a requeue logs "labelled" again but keeps its earlier
    epochs, so each epoch takes its last logged value over the whole log (a rerun overwrites it).
    """
    variants = defaultdict(lambda: defaultdict(list))
    for log in sorted(task_dir.glob("adapt_*/log.jsonl")):
        variant = re.sub(r"^adapt_|_seed\d+$", "", log.parent.name)
        per_epoch = {}
        for l in open(log):
            r = json.loads(l) if l.strip() else {}
            if "epoch" in r and "all" in r:
                per_epoch[r["epoch"]] = (r["all"], r["priv"])
        for e, v in per_epoch.items():
            variants[variant][e].append(v)
    return variants


def cell(by_epoch, e, n_max):
    rows = by_epoch.get(e)
    if not rows:
        return f"{'-':>5} {'-':>5}  "
    mark = " " if len(rows) == n_max else "*"  # * = fewer seeds than the task has
    return f"{st.mean(a for a, _ in rows):5.1f} {st.mean(p for _, p in rows):5.1f}{mark} "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--epochs", nargs="*", type=int, default=[15])
    args = ap.parse_args()
    tasks = [d for d in Path(args.root).iterdir() if d.is_dir() and any(d.glob("adapt_*/log.jsonl"))]
    tasks.sort(key=lambda d: (ORDER.index(d.name) if d.name in ORDER else len(ORDER), d.name))

    head = f"{'task':<22}{'variant':<24}{'n':>2}  {'best ep':>7} {'ALL':>5} {'PRIV':>5}   "
    head += "".join(f"{'ep ' + str(e):>5} {'ALL':>5} {'PRIV':>5}   " for e in args.epochs)
    head += f"{'last ep':>7} {'ALL':>5} {'PRIV':>5}"
    print(head)
    for d in tasks:
        for variant, by_epoch in load(d).items():
            n = max(len(v) for v in by_epoch.values())
            full = [e for e, v in by_epoch.items() if len(v) == n]  # epochs every seed has reached
            best = max(full, key=lambda e: st.mean(a for a, _ in by_epoch[e]))
            last = max(by_epoch)  # marked * if some seed has not reached it
            line = f"{d.name:<22}{variant:<24}{n:>2}  {best:>7} {cell(by_epoch, best, n)}  "
            line += "".join(f"{e:>5} {cell(by_epoch, e, n)}  " for e in args.epochs)
            line += f"{last:>7} {cell(by_epoch, last, n)}"
            print(line)
    print("\nMeans over seeds. best = epoch with the highest mean ALL (chosen with target labels).")
    print("* = fewer seeds than the task has at that epoch. - = epoch not reached.")


if __name__ == "__main__":
    main()
