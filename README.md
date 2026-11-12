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
slurm/               CCDS GPU TC2 job scripts: run.sbatch (one experiment), check.sbatch (environment check)
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

### On the CCDS GPU TC2 cluster
The scripts in `slurm/` follow the cluster user guide:
- no code runs on the head node: everything goes through `sbatch`, not `srun`;
- QoS `normal` (1 GPU, ≤ 10 CPUs, ≤ 30 GB, ≤ 6 h);
- one job at a time.

One job runs a whole experiment: source training, then the five adaptation seeds one after another. Anything already finished is skipped, so you can simply resubmit the same command. Overrides go after the config path as plain `key=value` pairs, without `--set`.

```bash
sbatch slurm/check.sbatch                                   # once: GPU + environment + tests
sbatch slurm/run.sbatch configs/digits_m2u.yaml             # source, then seeds 0-4
sbatch slurm/run.sbatch configs/digits_m2u_chi2.yaml        # chi2 negative threshold
sbatch --export=ALL,STAGES=adapt slurm/run.sbatch configs/office_a2d.yaml adapt.ablation=a   # reuses the source model
sbatch --export=ALL,SEEDS="0" slurm/run.sbatch configs/digits_m2u.yaml                      # just one seed
squeue -u $USER                                             # status; output in logs/cida_<jobid>.out
python scripts/aggregate.py runs/                           # results table (cheap; or run it after a job)
```

Submit the next experiment only after the previous one finishes. With one GPU per user, extra jobs only sit in the queue as `QOSMaxGRESPerUser`, and the guide asks you to cancel those.

To keep data elsewhere, symlink it into the repo (`ln -s ~/datasets data`) or add `data_root=/path/to/data`.

Both stages save a checkpoint every epoch and every 20 minutes, and resume automatically from `ckpt.pt` in their run folder. `run.sbatch` requeues itself 3 minutes before the 6 h limit. `--force` makes `train_source.py` or `adapt.py` rerun a finished stage.

## Suggested order
1. `configs/smoke.yaml` (tests) → **Digits M→U** (paper: 72.5 / 71.6). Tune the epoch counts here.
2. **Office A→D one-shot** (paper: 72.2 / 72.6; source accuracy 96.7%).
3. The other Office transfers, then ablations, then few-shot (`adapt.fewshot_frac=0.05`):
   - adaptation ablations reuse the A→D source model: `adapt.ablation=a|b|c`;
   - source ablations train their own source model, through separate configs: `configs/office_a2d_nols1.yaml`, `configs/office_a2d_noneg.yaml`.
4. Table 2 settings (`office20_*`, `digits_s2m`, `digits_u2m`), then VisDA last.

Every config `name` gets its own `runs/<name>/` folder. So a variant that changes source training must have its own config with a new `name` (as the `*_chi2`, `*_nols1` and `*_noneg` configs do), or it would overwrite the default source model.
