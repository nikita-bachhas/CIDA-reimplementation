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
scripts/trajectory.py Per-epoch metrics averaged over seeds, per variant (for diagnosing drift)
scripts/confusion.py  Per-class confusion for finished runs (GPU job: slurm/confusion.sbatch)
configs/             One YAML per transfer (base.yaml holds the defaults)
slurm/               CCDS GPU TC2 job scripts: run.sbatch (one experiment), check.sbatch (environment check)
tests/               Prototype unit tests + an end-to-end smoke test on synthetic data
```

## Setup (laptop)
```bash
conda create -n cida -c conda-forge --override-channels python=3.11 -y && conda activate cida
pip install -r requirements.txt
pytest -q          # ~1-2 min on a laptop CPU
```

## Setup (CCDS GPU TC2 cluster)
These are the commands that worked on TC2. The cluster's rules, from its user guide:
- **No code runs on the head node** (`CCDS-TC2`). Installing packages and downloading data there is fine; training, tests and `nvidia-smi` must go through `sbatch`.
- **Use `sbatch`, not `srun`.** `sbatch` refuses to submit from a compute node (`TC2N0x`), so `exit` back to the head node first.
- **Use only your assigned QoS** (`normal`: 1 GPU, ≤ 10 CPUs, ≤ 30 GB, ≤ 6 h, one GPU job at a time). Jobs with any other QoS are killed without notice.
- **Nothing is backed up.** Copy results off the cluster.

**1. Clone** (on campus or via the NTU VPN; head node 10.96.189.12):
```bash
ssh <ntu-username>@10.96.189.12
git clone https://github.com/nikita-bachhas/CIDA-reimplementation.git && cd CIDA-reimplementation
```

**2. Conda environment.** The shared Anaconda is read-only, and its default channels fail with `CondaToSNonInteractiveError`, so create the environment in your home directory from conda-forge:
```bash
module load anaconda/25.5.1        # skip if your prompt already shows (base)
conda config --add envs_dirs ~/.conda/envs
conda create -n cida -c conda-forge --override-channels python=3.11 -y
conda activate cida && which python        # should end in ~/.conda/envs/cida/bin/python
```

**3. Packages.** Install PyTorch built for CUDA 12.1 first; a plain `pip install torch` may pick a CUDA build the nodes don't support. Then:
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"   # 2.5.1+cu121 0.20.1+cu121
```

**4. Pre-download data and weights on the head node.** Compute nodes may have no internet access.
```bash
python -c "from torchvision import datasets as d; [f('data', train=t, download=True) for f in (d.MNIST, d.USPS) for t in (True, False)]; [d.SVHN('data', split=s, download=True) for s in ('train','test')]"
python -c "import torchvision.models as m; m.resnet50(weights=m.ResNet50_Weights.IMAGENET1K_V1); print('resnet ok')"
```

**5. Check the environment on a GPU node** (about 2 minutes):
```bash
sbatch slurm/check.sbatch
squeue -u $USER                      # wait until the job disappears
cat logs/cida-check_<jobid>.out      # expect "cuda available: True", "16 passed", "== CHECK PASSED"
```

The job scripts use partition `MGPU-TC2`, QoS `normal`, account `stud` and `module load anaconda/25.5.1`. If your account or QoS differs (check with `sacctmgr show assoc user=$USER format=account,qos -P`), edit the `#SBATCH` lines in `slurm/*.sbatch`.

**Useful commands:**
- `squeue -u $USER`: your jobs. `PD` with `QOSMaxGRESPerUser` means another of your jobs holds the GPU.
- `scancel <jobid>`: cancel a job.
- `seff <jobid>`: CPU and memory used by a finished job.
- `ncdu ~`: disk usage (100 GB quota).

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

Measured on an A40 (job 53729): about 5 s per Digits source epoch and 8 s per Digits adaptation epoch. A full `digits_m2u` experiment (source + 5 seeds) takes about 25 minutes.

## Suggested order
1. `configs/smoke.yaml` (tests) → **Digits M→U** (paper: 72.5 / 71.6). Tune the epoch counts here.
2. **Office A→D one-shot** (paper: 72.2 / 72.6; source accuracy 96.7%).
3. The other Office transfers, then ablations, then few-shot (`adapt.fewshot_frac=0.05`):
   - adaptation ablations reuse the A→D source model: `adapt.ablation=a|b|c`;
   - source ablations train their own source model, through separate configs: `configs/office_a2d_nols1.yaml`, `configs/office_a2d_noneg.yaml`.
4. Table 2 settings (`office20_*`, `digits_s2m`, `digits_u2m`), then VisDA last.

Every config `name` gets its own `runs/<name>/` folder. So a variant that changes source training must have its own config with a new `name` (as the `*_chi2`, `*_nols1` and `*_noneg` configs do), or it would overwrite the default source model.
