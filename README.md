# CIDA reimplementation

A from-scratch PyTorch reimplementation of *Class-Incremental Domain Adaptation* (Kundu et al., ECCV 2020). The paper has no official code. See [SPEC.md](SPEC.md) for what the paper specifies and what we chose ourselves.

## Layout
```
cida/prototypes.py   Gaussian prototypes: fitting, log-density, proxy and negative sampling
cida/data.py         Digits / Office-31 / VisDA loaders, class splits, one-shot selection
cida/models.py       LeNet / ResNet-50 feature extractors, gs, gt, fe, fd (supp. Table 2)
cida/metrics.py      ALL / PRIV / shared accuracy (per-sample and class-averaged)
train_source.py      Stage 1: foresighted source training  -> runs/<name>/source/final.pt
adapt.py             Stage 2: CIDA on the target            -> runs/<name>/adapt_*_seed<k>/
scripts/aggregate.py Mean ± std over seeds next to the paper's numbers
configs/             One YAML per transfer (base.yaml holds the defaults)
slurm/               A40 job scripts with checkpoint/requeue for the 6 h limit
tests/               Prototype unit tests + an end-to-end smoke test on synthetic data
```

## Setup
```bash
conda create -n cida python=3.11 -y && conda activate cida
pip install -r requirements.txt
pytest -q          # ~1-2 min on a laptop CPU
```

Data goes under `data/` (or set `data_root=`):
- **Digits:** downloaded automatically by torchvision.
- **Office-31:** `data/office31/{amazon,dslr,webcam}/images/<class>/*.jpg`
- **VisDA-2017 (classification):** `data/visda/train/<class>/` (synthetic) and `data/visda/validation/<class>/` (real)

## Running
Locally:
```bash
python train_source.py --config configs/digits_m2u.yaml
python adapt.py --config configs/digits_m2u.yaml --seed 0
```

Any config value can be overridden:
```bash
python adapt.py --config configs/office_a2d.yaml --seed 0 --set adapt.ablation=b adapt.epochs=40
```

On the cluster (edit the partition, account and conda lines in `slurm/*.sbatch` first):
```bash
bash slurm/pipeline.sh configs/digits_m2u.yaml                       # source, then seeds 0-4
sbatch --array=0-4 slurm/adapt.sbatch configs/office_a2d.yaml --set adapt.ablation=a
python scripts/aggregate.py runs/
```

Both stages save a checkpoint every epoch and every 20 minutes, and resume automatically from `ckpt.pt` in their run folder. The SLURM scripts requeue themselves 3 minutes before the time limit.

## Suggested order
1. `configs/smoke.yaml` (tests) → **Digits M→U** (paper: 72.5 / 71.6). Tune the epoch counts here.
2. **Office A→D one-shot** (paper: 72.2 / 72.6; source accuracy 96.7%).
3. The other Office transfers, then ablations (`adapt.ablation=a|b|c`, `src_train.use_ls1=false`, `src_train.use_negatives=false`), then few-shot (`adapt.fewshot_frac=0.05`).
4. Table 2 settings (`office20_*`, `digits_s2m`, `digits_u2m`), then VisDA last.

Runs that change source training (for example `use_negatives=false`) need their own `out_dir`, so they don't overwrite the default source model:
`--set out_dir=runs/office_a2d_noneg src_train.use_negatives=false`.
