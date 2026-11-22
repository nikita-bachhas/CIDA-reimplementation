"""Preview Office one-shot images, then fix one image per private class.

  python scripts/choose_office_supports.py --data-root /path/to/data --candidates 20
  python scripts/choose_office_supports.py --data-root /path/to/data --candidates 20 --picks ...

Default: Amazon -> DSLR, 20 shared / 11 private classes. --config accepts either
Office split and any transfer. Picks are the support-seed numbers shown above
the preview columns, one per private class in the printed order.
"""
import argparse
import os
from pathlib import Path
import sys


def fixed_filenames(ds, classes, draws, picks):
    """Use ImageFolder filenames, exactly as cida.data.fixed_labelled expects."""
    if len(picks) != len(classes) or any(p < 0 or p >= len(draws) for p in picks):
        raise ValueError(f'Provide {len(classes)} picks, each between 0 and {len(draws) - 1}.')
    selected = {}
    for c, pick in zip(classes, picks):
        index = draws[pick][c][0]
        name = Path(ds.base.samples[ds.indices[index]][0]).name
        matches = [i for i, label in enumerate(ds.labels)
                   if label == c and Path(ds.base.samples[ds.indices[i]][0]).name == name]
        if matches != [index]:
            raise ValueError(f'Ambiguous filename for {ds.class_names[c]}: {name}')
        selected[ds.class_names[c]] = name
    return selected


def clearshot_config(config, output, cfg, selected, picks):
    """Isolate this run and use the same seed / confident-La1 choices as Digits."""
    name = cfg['name'] if cfg['name'].endswith('_siddharth') else cfg['name'] + '_siddharth'
    return {
        'base': Path(os.path.relpath(config, output.parent)).as_posix(),
        'name': name,
        'out_dir': f'runs/{name}',
        'data_root': cfg['data_root'],
        'model': {'proj_residual': True, 'proj_final_act': False},
        'adapt': {
            'shots': 1, 'fewshot_frac': None, 'seed_offset': 0,
            'tag': 'clearshot_' + '-'.join(map(str, picks)) + '_la1conf',
            'fixed_shots': selected, 'la1_samples': 'confident', 'la1_temp': 1.0,
            'adabn': True, 'pseudo_label': 'nearest', 'rectify': {'enabled': False},
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/office20_a2d_siddharth.yaml')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--picks', help='Comma-separated support-seed numbers, one per private class')
    args = parser.parse_args(argv)
    if args.candidates < 1:
        parser.error('--candidates must be positive.')
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import numpy as np
    from PIL import Image, ImageDraw
    import yaml
    from cida.common import load_config
    from cida.data import MEAN, STD, build_domain, fixed_labelled, sample_labelled

    config = Path(args.config).resolve()
    cfg = load_config(config, [f'data_root={Path(args.data_root).resolve().as_posix()}'])
    if cfg['dataset'] != 'office31':
        parser.error('Use an Office config.')
    ds = build_domain(cfg, cfg['tgt_domain'], cfg['tgt_train_split'], train_aug=False)
    classes = list(range(ds.n_shared, ds.n_shared + ds.n_private))
    draws = [sample_labelled(ds.labels, classes, shots=1, seed=s) for s in range(args.candidates)]
    name = cfg['name'] if cfg['name'].endswith('_siddharth') else cfg['name'] + '_siddharth'
    output = repo / 'configs' / f'{name}_clearshot.yaml'
    if output == config:
        parser.error('Pass the baseline config, rather than a generated clearshot config.')

    columns = min(args.candidates, 10)
    blocks = (args.candidates + columns - 1) // columns
    tile, top, caption = 180, 30, 36
    height = top + len(classes) * (tile + caption)
    sheet = Image.new('RGB', (columns * tile, blocks * height), 'white')
    pen = ImageDraw.Draw(sheet)
    for seed in range(args.candidates):
        left, y0 = seed % columns * tile, seed // columns * height
        pen.text((left + 8, y0 + 8), f'Support seed {seed}', fill='black')
        for row, c in enumerate(classes):
            index = draws[seed][c][0]
            # Show the exact deterministic guide view, undoing ImageNet normalization.
            x = ds[index][0].detach().cpu().numpy().transpose(1, 2, 0)
            pixels = np.clip((x * np.asarray(STD) + np.asarray(MEAN)) * 255, 0, 255).astype('uint8')
            im = Image.fromarray(pixels).resize((tile, tile), Image.Resampling.BILINEAR)
            y = y0 + top + row * (tile + caption)
            sheet.paste(im, (left, y))
            filename = Path(ds.base.samples[ds.indices[index]][0]).name
            pen.text((left + 3, y + tile + 2), f'{ds.class_names[c]}\n{filename}', fill='black')
    review = repo / 'oneshot_review'
    review.mkdir(exist_ok=True)
    preview = review / f'{name}_support_preview.png'
    sheet.save(preview)
    print(f'Open {preview}')
    print('Private-class order:', ', '.join(ds.class_names[c] for c in classes))
    print(f'Choose {len(classes)} support-seed numbers; repeated candidate images are possible.')
    if not args.picks:
        return
    try:
        picks = [int(v.strip()) for v in args.picks.split(',')]
        selected = fixed_filenames(ds, classes, draws, picks)
        resolved = fixed_labelled(ds, classes, selected)
    except ValueError as exc:
        parser.error(str(exc))
    assert all(resolved[c] == draws[p][c] for c, p in zip(classes, picks))
    output.write_text(yaml.safe_dump(clearshot_config(config, output, cfg, selected, picks),
                                     sort_keys=False), encoding='utf-8')
    print('Fixed support images:', selected)
    print(f'These {len(classes)} images stay fixed across actual training seeds 0-4.')
    print(f'Run: sbatch slurm/run.sbatch {output.relative_to(repo).as_posix()}')


if __name__ == '__main__':
    main()
