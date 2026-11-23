"""Summarise best, epoch 15 and final scores from existing logs; no training or Torch needed.

  python scripts/best_epoch_summary.py runs/
  python scripts/best_epoch_summary.py runs/digits_u2m --csv best_epochs.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics as st
import sys
from collections import defaultdict

METRICS = ('all', 'shared', 'priv', 'all_cls')


def number(row, key):
    value = row.get(key)
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def read_epochs(path, min_epoch=1):
    """Keep the latest record per epoch; a fresh epoch 0 starts a new training attempt."""
    epochs = {}
    malformed = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += bool(line.strip())
            continue
        if not isinstance(row, dict) or not isinstance(row.get('epoch'), int):
            continue
        if not all(number(row, k) is not None for k in ('all', 'shared', 'priv')):
            continue
        if row['epoch'] == 0:
            epochs.clear()
        if row['epoch'] >= min_epoch:
            epochs[row['epoch']] = row
    if malformed:
        print(f'Note: ignored {malformed} malformed/incomplete log line(s): {path}', file=sys.stderr)
    return list(epochs.values())


def choose_epoch(rows, select_by='all'):
    candidates = [r for r in rows if number(r, select_by) is not None]
    # ALL/SHARED/PRIV come from ONE epoch. Ties select the earliest epoch.
    return max(candidates, key=lambda r: (r[select_by], -r['epoch'])) if candidates else None


def collect(root, select_by='all', min_epoch=1):
    groups = defaultdict(list)
    for file in sorted(Path(root).rglob('final_metrics.json')):
        match = re.fullmatch(r'(adapt_.+)_seed(\d+)', file.parent.name)
        log = file.parent / 'log.jsonl'
        if not match or not log.is_file():
            continue
        epochs = read_epochs(log, min(min_epoch, 15))
        best = choose_epoch([r for r in epochs if r['epoch'] >= min_epoch], select_by)
        if best is None:
            print(f'Skipped: no eligible epoch metrics: {log}', file=sys.stderr)
            continue
        final = json.loads(file.read_text(encoding='utf-8'))
        epoch15 = next((r for r in epochs if r['epoch'] == 15), {})
        row = {'task': file.parent.parent.name, 'variant': match[1],
               'seed': int(match[2]), 'selected_epoch': best['epoch'], 'select_by': select_by}
        for key in METRICS:
            row['best_' + key], row['final_' + key] = number(best, key), number(final, key)
            row['epoch15_' + key] = number(epoch15, key)
        groups[(row['task'], row['variant'])].append(row)
    return groups


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return 'n/a'
    return f'{st.mean(values):.2f} +/- {st.stdev(values):.2f}' if len(values) > 1 else f'{values[0]:.2f}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', nargs='?', default='runs', type=Path)
    ap.add_argument('--select-by', choices=('all', 'priv', 'all_cls'), default='all')
    ap.add_argument('--min-epoch', type=int, default=1, help='Exclude pre-adaptation epoch 0 by default')
    ap.add_argument('--csv', type=Path, help='Optional per-seed CSV output')
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        ap.error(f'Run directory does not exist: {args.root}')
    groups = collect(args.root, args.select_by, args.min_epoch)
    if not groups:
        ap.error('No completed adaptation runs with eligible epoch logs found.')
    print(f'TEST-SELECTED DIAGNOSTIC: maximise {args.select_by.upper()} per seed; earliest epoch wins ties.')
    print('All metrics use the SAME selected epoch. Completed runs only. Std = sample standard deviation.')
    output = []
    for (task, variant), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r['seed'])
        print(f'\n{task} / {variant}  (n={len(rows)})')
        print(f'{"seed":>5} {"epoch":>6}' + ''.join(f'{k.upper():>20}' for k in METRICS))
        for r in rows:
            print(f'{r["seed"]:5d} {r["selected_epoch"]:6d}' + ''.join(
                f'{fmt([r["best_" + k]]):>20}' for k in METRICS))
        for prefix, label in (('best_', 'BEST'), ('epoch15_', 'EPOCH 15'), ('final_', 'FINAL')):
            print(f'{label:>12}' + ''.join(f'{fmt([r[prefix + k] for r in rows]):>20}' for k in METRICS))
        missing = [str(r['seed']) for r in rows if r['epoch15_all'] is None]
        print(f'Epoch 15 coverage: {len(rows) - len(missing)}/{len(rows)} seeds.' +
              (f' Missing seeds: {", ".join(missing)}; no other epoch substituted.' if missing else ''))
        output.extend(rows)
    if args.csv:
        with args.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
        print(f'\nSaved {args.csv}')
    print('\nThis summarises logged scores; it does not recover unsaved best-epoch model weights.')


if __name__ == '__main__':
    main()
