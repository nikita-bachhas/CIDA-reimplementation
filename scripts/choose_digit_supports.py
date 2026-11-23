"""Preview one-shot candidates, then select by appearance without scoring a model.

Run from the repository root:
  python scripts/choose_digit_supports.py --data-root /path/to/data
  python scripts/choose_digit_supports.py --data-root /path/to/data --picks 0,1,2,3,4
The five picks are preview-column numbers for private digits 5,6,7,8,9.
"""
import argparse
import os
from pathlib import Path
import sys


def support_positions(ds, classes, draws, picks):
    """Translate selected global dataset indices to fixed_shots' per-class positions."""
    if len(picks) != len(classes) or any(s < 0 or s >= len(draws) for s in picks):
        raise ValueError(f'Provide {len(classes)} picks, each between 0 and {len(draws) - 1}.')
    members = {c: [i for i, label in enumerate(ds.labels) if label == c] for c in classes}
    return {ds.class_names[c]: members[c].index(draws[s][c][0]) for c, s in zip(classes, picks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/digits_m2u_improved.yaml')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--picks', help='One preview-column number per private digit, e.g. 0,1,2,3,4')
    parser.add_argument('--candidates', type=int, default=5, help='Candidate support seeds 0..N-1')
    args = parser.parse_args(argv)
    if args.candidates < 1:
        parser.error('--candidates must be positive.')
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import numpy as np
    from PIL import Image, ImageDraw
    import yaml
    from cida.common import load_config
    from cida.data import build_domain, sample_labelled, fixed_labelled

    config = Path(args.config).resolve()
    cfg = load_config(config, [f'data_root={Path(args.data_root).resolve().as_posix()}'])
    if cfg['dataset'] != 'digits':
        parser.error('Use a digits config.')
    ds = build_domain(cfg, cfg['tgt_domain'], cfg['tgt_train_split'], train_aug=False)
    classes = list(range(ds.n_shared, ds.n_shared + ds.n_private))
    draws = [sample_labelled(ds.labels, classes, shots=1, seed=s) for s in range(args.candidates)]
    columns = min(args.candidates, 10)
    blocks = (args.candidates + columns - 1) // columns
    tile, top = 140, 30
    height = top + len(classes) * (tile + 30)
    sheet = Image.new('RGB', (columns * tile, blocks * height), 'white')
    pen = ImageDraw.Draw(sheet)
    for seed in range(args.candidates):
        left, y0 = seed % columns * tile, seed // columns * height
        pen.text((left + 15, y0 + 8), f'Support seed {seed}', fill='black')
        for row, c in enumerate(classes):
            index = draws[seed][c][0]
            x = ds[index][0].detach().cpu().numpy()
            pixels = np.clip((x.transpose(1, 2, 0) * 0.5 + 0.5) * 255, 0, 255).astype('uint8')
            image = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.NEAREST)
            y = y0 + top + row * (tile + 30)
            sheet.paste(image, (left, y))
            pen.text((left + 5, y + tile + 4), f'Digit {ds.class_names[c]}, idx {index}', fill='black')
    review = repo / 'oneshot_review'
    review.mkdir(exist_ok=True)
    preview = review / f'{cfg["name"]}_support_preview.png'
    sheet.save(preview)
    print(f'Open {preview}. Choose one support seed for each digit (rows top to bottom).')
    if not args.picks:
        return
    try:
        picks = [int(v.strip()) for v in args.picks.split(',')]
        selected = support_positions(ds, classes, draws, picks)
    except ValueError as exc:
        parser.error(str(exc))
    resolved = fixed_labelled(ds, classes, selected)
    assert all(resolved[c] == draws[s][c] for c, s in zip(classes, picks))
    output = repo / 'configs' / f'{cfg["name"]}_clearshot.yaml'
    tag = 'clearshot_' + '-'.join(map(str, picks))
    output.write_text(yaml.safe_dump({
        'base': Path(os.path.relpath(config, output.parent)).as_posix(),
        'data_root': cfg['data_root'],
        'adapt': {'shots': 1, 'fewshot_frac': None, 'seed_offset': 0,
                  'tag': tag, 'fixed_shots': selected},
    }, sort_keys=False), encoding='utf-8')
    print('Fixed support indices:', {ds.class_names[c]: resolved[c][0] for c in classes})
    print('These five images stay fixed across actual training seeds 0-4.')
    print(f'Run: sbatch slurm/run.sbatch {output.relative_to(repo).as_posix()}')


if __name__ == '__main__':
    main()
