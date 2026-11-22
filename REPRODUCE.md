# Reproducing our CIDA runs

This lists every change we made on top of the paper's method, the config switch for each, and the exact commands that reproduce our numbers. The evidence behind each change is in [SPEC.md](SPEC.md); environment setup and cluster rules are in [README.md](README.md).

## 1. Code version

- Repository: `https://github.com/nikita-bachhas/CIDA-reimplementation.git`
- Branch **`office-rectify-sweep`**. It is `master` at `d79b5c3` plus our commits: rectification, guide diagnostics, post-hoc prediction rules, hand-picked one-shot images. `master` has since gained three SPEC-only commits (vanilla results); the code is the same apart from our additions.
- Everything in the main configuration below is the same on `master` and on this branch. Rectification, `adapt.fixed_shots`, `scripts/posthoc.py`'s guide rules, `guide_diag.py`, `confusion.py`'s class table, `contact_sheet.py` and `shot_check.py` exist only on this branch.

```bash
git clone https://github.com/nikita-bachhas/CIDA-reimplementation.git && cd CIDA-reimplementation
git checkout office-rectify-sweep
python -m pytest -q          # 39 tests, about 2 min on a laptop CPU
```

## 2. Environment and data

- Python 3.11, `torch==2.5.1` and `torchvision==0.20.1` built for CUDA 12.1, then `pip install -r requirements.txt` (README.md, "Cluster setup", steps 2–3).
- The cluster runs used NTU CCDS TC2, one **NVIDIA A40** per job, through `slurm/run.sbatch`.
- ImageNet ResNet-50 weights: torchvision `ResNet50_Weights.IMAGENET1K_V1` (README step 4 pre-downloads them).
- Data under `data/` (or `data_root=/path`):
  - Office-31: `data/office31/{amazon,dslr,webcam}/images/<class>/frame_XXXX.jpg`, the standard release (DSLR 498, Webcam 795, Amazon 2817 images). `adapt.fixed_shots` refers to these file names, so the folders must be the standard ones.
  - Digits: MNIST / USPS / SVHN downloaded by torchvision.
  - VisDA-2017: `data/visda/train/<class>/` (synthetic), `data/visda/validation/<class>/` (real).

## 3. Baseline: the paper as written

`preset=vanilla` (`configs/presets/vanilla.yaml`) switches off all five method changes in section 4, so the code runs the method exactly as the paper describes it. Runs go to `runs_vanilla/<task>/`.

```bash
sbatch slurm/run.sbatch configs/office_a2d.yaml preset=vanilla
```

Baseline results, 5 seeds, ALL / PRIV at the last epoch: Office A→D 45.2 / 16.9 (paper 72.2 / 72.6); Digits M→U 43.7 / 50.5 (paper 72.5 / 71.6). Other tasks are in SPEC.md, "Vanilla preset".

## 4. Improvements in our main configuration

These five are on by default in `configs/base.yaml` and the task configs. The "Vanilla" column is what `preset=vanilla` sets.

| # | Change | Config key: main / vanilla | Why | Evidence (SPEC.md) |
|---|---|---|---|---|
| 1 | **Freeze the ResNet-50 backbone** at its ImageNet weights in both stages (Office, VisDA). BatchNorm statistics still adapt; the FC head, classifiers and fe/fd train at lr 1e-4. | `src_train.backbone_lr_mult`, `adapt.backbone_lr_mult`: `0.0` / `1.0` (set in `configs/office31.yaml`, `configs/visda.yaml`) | Fine-tuning the whole ResNet on ~900 source images of 10 classes destroys the features that separate the unseen private classes. | Office A→D: 49.8 / 33.3 → **70.3 / 48.5** (the 49.8 run also had the one-FC256 head bug below, so this jump combines both fixes). One-shot nearest-guide probe on DSLR, which isolates the backbone: 72.4 PRIV with ImageNet features vs 27.0 after full fine-tuning. |
| 2 | **AdaBN before adaptation:** re-estimate ft's BatchNorm running statistics on the unlabelled target set right after ft ← fs. | `adapt.adabn`: `true` / `false` | With source BN statistics, target shared samples lie closer to the one-shot target guides than to the source class means, so most get private pseudo-labels. | Digits: shared samples pseudo-labelled private 74–78% → 14–21%; their pseudo-label accuracy 28–38% → 61–69%. |
| 3 | **Residual fe / fd:** u + MLP(u) with the last layer zero-initialised, so fe = fd = identity exactly at the start (the paper's near-identity pre-training still runs, at zero error). | `model.proj_residual`: `true` / `false` | A plain MLP pre-trained to near-identity keeps ~13% relative error (24% on VisDA), which displaces the shared guides fe(μ_s) before adaptation starts. | Digits M→U: ALL 44.8 ± 5.2 with a plain MLP vs 70.5 with residual fe/fd. |
| 4 | **Linear last layer in fe / fd.** | `model.proj_final_act`: `false` / `true` | The ELU/LeakyReLU last layer listed in supp Table 2 cannot output negative values, so it cannot be the identity on BatchNorm features. | Required for change 3 to hold exactly; no separate ablation. |
| 5 | **La1 on the confident set B_t:** La1 batches are drawn from B_t (top 20% per guide) instead of all target samples. The label is still the nearest guide; P_t is still fit on all pseudo-labelled samples. | `adapt.la1_samples`: `confident` / `all` | On all samples, La1 trains ft on ~65%-accurate pseudo-labels and reinforces its own errors, pushing shared samples into private clusters. | Digits M→U ALL / PRIV / shared: paper La1 55.1 / 58.4 / 52.8 vs B_t 71.9 / 55.3 / 83.4. |

**Bug fix (on in both main and vanilla, because the paper specifies it):** the ResNet head is ResNet-50 → FC1024-ELU → BN → FC256-ELU → FC256-ELU → BN, as in supp Table 2A (`model.resnet_fc_out: 2`). Our first Office run had one FC256 layer by mistake.

## 5. Choices the paper leaves open (identical in main and vanilla)

| Item | Our value | Key |
|---|---|---|
| Likelihood inside the Ls1 / La1 softmax | log N(u; μ_c, Σ_c) as the logit (raw densities underflow in 100–256 dimensions) | built in |
| Covariance regularisation | Σ ← 0.9 Σ + (0.1 + 1e-4) · (tr Σ / D) · I | `cov_shrink: 0.1`, `cov_eps: 1e-4` (src_train and adapt) |
| Batch size | 64 Digits, 32 Office / VisDA | `batch_size` |
| Epochs, source / adaptation | Digits 10 / 30, Office 50 / 60, VisDA 5 / 10 | `src_train.epochs`, `adapt.epochs` |
| Adaptation "epoch" | len(target loader) × 5 losses iterations | `adapt.iters_per_epoch: null` |
| Negatives | 3σ test of supp Algo 1 | `src_train.neg_mode: paper` |
| Confident set | top 20% nearest per guide | `adapt.n_percent: 20` |
| Private guides | the one-shot images go through ft in the same batch as target samples (no single-image BN batches) | built in |
| Office classes (Table 3) | shared = indices 0–9, private = 20–29 (alphabetical; trash_can dropped, see SPEC Inconsistencies 1) | `configs/office31.yaml` |
| Office data | transductive: adapt and evaluate on the whole target domain, excluding the labelled one-shot images (317 DSLR images on A→D) | `tgt_*_split: all` |
| Digits data | adapt on the target train split, evaluate on the target test split | `configs/digits.yaml` |
| Model selection | last epoch (no target labels) | built in |
| Metrics | ALL / PRIV / shared per image, plus class-averaged `*_cls` | `cida/metrics.py` |
| Seeds | source model `source_seed: 0`; adaptation `--seed k` → one-shot draw `sample_labelled(seed=k)` and RNG `set_seed(1000 + k)`, k = 0–4 | |
| Optimiser | Adam, lr 1e-4, one optimiser per loss (paper) | `lr` |

## 6. Runs and expected results

Run one job at a time (QoS `normal` allows one GPU job). `run.sbatch` trains the source model if it is missing, then seeds 0–4, skipping anything already finished. Collect results with `python scripts/aggregate.py runs/`.

| Run | Command | Expected (5-seed mean, last epoch) |
|---|---|---|
| **Main, Office A→D** | `sbatch slurm/run.sbatch configs/office_a2d.yaml` | ALL 70.3 ± 4.5 / PRIV 48.5 ± 8.8 / shared 93.4; class-averaged 72.4 / 51.9 |
| Main, Digits M→U | `sbatch slurm/run.sbatch configs/digits_m2u.yaml` | ALL 70.5 ± 10.5 / PRIV 53.5 ± 17.4 |
| Main, Digits U→M / S→M | `configs/digits_u2m.yaml`, `configs/digits_s2m.yaml` | 65.5 / 51.0 and 58.8 / 44.0 (older runs, not re-run since the pipeline became deterministic) |
| Rectification (experimental) | `sbatch slurm/run.sbatch configs/office_a2d_rect.yaml` | 70.3 / 50.8 / 91.0 (no net gain; not adopted) |
| **Hand-picked one-shot images** (not the paper's protocol) | `sbatch slurm/run.sbatch configs/office_a2d_fixedshot.yaml` | **ALL 81.9 ± 1.9 / PRIV 69.8 ± 2.7 / shared 94.7; class-averaged 83.3 / 72.3** |
| Vanilla baseline | append `preset=vanilla` to any command | section 3 |

The other Office tasks (`configs/office_{a2w,d2a,d2w,w2a,w2d}.yaml`) and VisDA (`configs/visda_sy2re.yaml`) run the same way; we have only run them with `preset=vanilla` so far.

### Table 2 setting: 20 shared + 11 private classes

`configs/office20_*.yaml`: |Cs| = 20 (indices 0–19), |Ct| = 31 (the 20 shared + private indices 20–30, i.e. Table 3's 10 private classes plus trash_can). All 31 Office classes are used. Images per task:

| Task | Source (20 classes) | Target shared / private |
|---|---|---|
| A→D, W→D | Amazon 1805 / Webcam 528 | DSLR 310 / 188 |
| A→W, D→W | Amazon 1805 / DSLR 310 | Webcam 528 / 267 |
| D→A, W→A | DSLR 310 / Webcam 528 | Amazon 1805 / 1012 |

The main configuration (section 4) applies unchanged. Each task trains its own 20-class source model the first time. `CONFIGS` runs several configs in one job, one after another, which keeps to one job in the queue:

```bash
# A->D: random one-shot draws (the paper's protocol), then the hand-picked images; the second reuses the source model
sbatch --export=ALL,CONFIGS="configs/office20_a2d.yaml configs/office20_a2d_fixedshot.yaml" slurm/run.sbatch
# W->D, the other DSLR-target task, with the same picks
sbatch --export=ALL,CONFIGS="configs/office20_w2d.yaml configs/office20_w2d_fixedshot.yaml" slurm/run.sbatch
# all six tasks, random draws
sbatch --export=ALL,CONFIGS="configs/office20_a2d.yaml configs/office20_a2w.yaml configs/office20_d2a.yaml configs/office20_d2w.yaml configs/office20_w2a.yaml configs/office20_w2d.yaml" slurm/run.sbatch
```

- Paper (Table 2, ALL / PRIV): A→D 73.3 / 73.1, A→W 63.6 / 62.6, D→A 64.1 / 64.3, D→W 80.3 / 79.4, W→A 63.7 / 62.8, W→D 89.5 / 88.4. Our vanilla baseline is in SPEC.md; the main configuration has not been run in this setting yet.
- Hand-picked configs exist only for the DSLR-target tasks (`office20_a2d_fixedshot.yaml`, `office20_w2d_fixedshot.yaml`): the same DSLR picks as Table 3 plus `trash_can: frame_0004.jpg`. Webcam and Amazon targets would need their own picks.
- Amazon-target tasks adapt on 2817 images and take much longer per seed; `run.sbatch` requeues itself at the 6 h limit and resumes from the checkpoint.
- Results: `python scripts/aggregate.py runs/` (it lists the paper's Table 2 numbers next to ours); class-wise: `sbatch slurm/confusion.sbatch runs/office20_a2d [--runs adapt_1shot_abl-none_fixedshot]`.
- Expected, A→D hand-picked (5 seeds, ALL / PRIV / shared): paper rule **74.0 / 43.1 / 91.6**; with the `lp` post-hoc step **82.1 / 68.0 / 90.1**:

```bash
sbatch slurm/posthoc.sbatch runs/office20_a2d --runs adapt_1shot_abl-none_fixedshot
```

**Post-hoc prediction rules (no retraining)** on finished runs, `scripts/posthoc.py`. Label propagation from the guides (`lp`) is the best label-free rule:

```bash
sbatch slurm/posthoc.sbatch runs/office_a2d                                   # main runs:  lp 75.2 / 58.0 / 93.4
sbatch slurm/posthoc.sbatch runs/office_a2d --runs adapt_1shot_abl-none_rect  # rect runs:  lp 75.8 / 62.3 / 90.1
```

`--runs` takes the run-name prefix before `_seed`, not a glob. Class-wise accuracy: `sbatch slurm/confusion.sbatch runs/office_a2d [--runs <prefix>]`. Guide diagnostics: `sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d [--runs <prefix>] [--stage start|final]`.

### Report runs: every paper result, in priority order

Main configuration, i.e. all five improvements from section 4 on (the configs below need no extra switch for that), under the paper's protocol (random one-shot draw per seed, paper prediction rule). One job at a time; each job requeues itself at the 6 h limit, skips finished runs and resumes unfinished ones, so a long job needs no babysitting. Times are rough A40 estimates.

**Run first: the backbone and lp pseudo-label tests**, seeds 0–2 only (SPEC "Under test: lp pseudo-labels in training"). Its outcome decides whether `adapt.pseudo_label: lp` joins the main configuration before the queue below, since every report run would then need it:

```bash
sbatch --export=ALL,SEEDS="0 1 2",CONFIGS="configs/office20_a2d_ft.yaml configs/office20_a2d.yaml configs/office20_a2d_lpref.yaml" slurm/run.sbatch
```

- `SEEDS="0 1 2"`: 3 seeds per config (3 × 3 = 9 adaptation runs), so the comparisons are seed-paired over seeds 0–2 only.
- All three configs are the Table 2 setting: |Cs| = 20 (indices 0–19), |Ct| = 31 (20 shared + 11 private, indices 20–30). Source Amazon 1,805 images; target DSLR 498, evaluated on 487.
- `office20_a2d_ft.yaml` against `office20_a2d.yaml`: the ResNet-50 **not frozen** (whole network at Adam 1e-4 in both stages, the natural reading of the paper) against frozen, every other improvement on in both, so the seed-paired difference is what freezing alone is worth. It trains its own 20-class source model (`runs/office20_a2d_ft/`). Slower than the frozen runs (gradients through the backbone); not timed.
- `office20_a2d.yaml` against `office20_a2d_lpref.yaml`: the lp pseudo-label test. `office20_a2d.yaml` is also job 2's first config: job 2 skips seeds 0–2 and runs only seeds 3–4.
- The Table 3 lp test (`configs/office_a2d_lpref.yaml`, `STAGES=adapt`) is not in this job; run it before making lp the default everywhere.

**Table 2 with lp pseudo-labels, all six tasks, 5 seeds** (after job 56572 showed lp +9.9 ALL / +28.7 PRIV on A→D). One `office20_<task>_lpref.yaml` per task: main configuration plus `adapt.pseudo_label: lp`, run dirs tagged `lpref` so they never collide with the nearest-guide runs. Each job trains the task's 20-class source model if it is missing, then seeds 0–4 (A→D skips its finished seeds 0–2). The four jobs are independent:

Submitted first (2026-11-22): the DSLR/Webcam targets. A→D keeps its 3 seeds from job 56572; the Amazon targets are deferred for time.

```bash
# W->D (~3 h)
sbatch slurm/run.sbatch configs/office20_w2d_lpref.yaml
# Webcam targets (~6 h)
sbatch --export=ALL,CONFIGS="configs/office20_a2w_lpref.yaml configs/office20_d2w_lpref.yaml" slurm/run.sbatch
```

Deferred:

```bash
# A->D seeds 3-4 (~1 h)
sbatch --export=ALL,STAGES=adapt,SEEDS="3 4" slurm/run.sbatch configs/office20_a2d_lpref.yaml
# Amazon targets (~17 h each; the jobs requeue themselves at 6 h)
sbatch slurm/run.sbatch configs/office20_d2a_lpref.yaml
sbatch slurm/run.sbatch configs/office20_w2a_lpref.yaml
```

**Hand-picked images + lp + 20 epochs** (2026-11-22; SPEC "Under test: hand-picked images with lp, 20 epochs"). A→D and W→D (W→D is furthest below the paper), the DSLR picks of the `*_fixedshot.yaml` configs, `pseudo_label: lp`, `adapt.epochs: 20`, seeds 0–4 (the seed changes only the training RNG). Reuses the source models from jobs 56508 and 56614. About 2 h:

```bash
sbatch --export=ALL,STAGES=adapt,CONFIGS="configs/office20_a2d_fixedshot_lp20.yaml configs/office20_w2d_fixedshot_lp20.yaml" slurm/run.sbatch
```

**W→D with an unfrozen-backbone source model, 100 source epochs** (2026-11-23; SPEC "Under test: W→D source model with the ResNet unfrozen, 100 epochs"). `configs/office20_w2d_srcft100.yaml`: the W→D lp20 run above with a new source model (ResNet-50 trained at Adam 1e-4, 100 epochs + 1 warm-up) in its own folder `runs/office20_w2d_srcft100/`; adaptation unchanged (ResNet frozen at the new source weights). Seed 0 first (source ~10 min + adaptation ~12 min on the A40); add seeds 1–4 with `STAGES=adapt` once it looks sane:

```bash
sbatch --export=ALL,SEEDS="0" slurm/run.sbatch configs/office20_w2d_srcft100.yaml
# later, reusing that source model:
sbatch --export=ALL,STAGES=adapt,SEEDS="1 2 3 4" slurm/run.sbatch configs/office20_w2d_srcft100.yaml
```

```bash
# 0. Old Digits runs predate the deterministic pipeline: move them aside so they are re-run
mv runs/digits_u2m runs/digits_u2m_old; mv runs/digits_s2m runs/digits_s2m_old
# 1. Digits + Table 3 DSLR/Webcam targets (~15 h)
sbatch --export=ALL,CONFIGS="configs/digits_u2m.yaml configs/digits_s2m.yaml configs/office_w2d.yaml configs/office_a2w.yaml configs/office_d2w.yaml" slurm/run.sbatch
# 2. Table 2 DSLR/Webcam targets (~16 h)
sbatch --export=ALL,CONFIGS="configs/office20_a2d.yaml configs/office20_w2d.yaml configs/office20_a2w.yaml configs/office20_d2w.yaml" slurm/run.sbatch
# 3. Ablations on A->D (~15 h, two jobs)
sbatch --export=ALL,STAGES=adapt,ABLATIONS="a b c" slurm/run.sbatch configs/office_a2d.yaml
sbatch --export=ALL,CONFIGS="configs/office_a2d_nols1.yaml configs/office_a2d_noneg.yaml" slurm/run.sbatch
# 4. Table 3 Amazon targets (~35 h)
sbatch --export=ALL,CONFIGS="configs/office_d2a.yaml configs/office_w2a.yaml" slurm/run.sbatch
# 5. Table 2 Amazon targets (~35 h)
sbatch --export=ALL,CONFIGS="configs/office20_d2a.yaml configs/office20_w2a.yaml" slurm/run.sbatch
# 6. Table 3 few-shot 5% (Ours*), Webcam/Amazon targets; reuses the source models from jobs 1 and 4 (~45 h)
sbatch --export=ALL,STAGES=adapt,CONFIGS="configs/office_a2w.yaml configs/office_d2w.yaml configs/office_d2a.yaml configs/office_w2a.yaml" slurm/run.sbatch adapt.fewshot_frac=0.05
# 7. VisDA Syn->Real (days)
sbatch slurm/run.sbatch configs/visda_sy2re.yaml
```

- Ours* on the DSLR targets is not run: 5% of a DSLR class is one image, the same one the one-shot run draws (SPEC inconsistency 4).
- The table for the report: `python scripts/aggregate.py runs/` (ours next to the paper) and `python scripts/aggregate.py runs_vanilla/` (the paper as written).
- If the cluster accepts pending jobs, `sbatch --dependency=singleton ...` (all jobs are named `cida`) queues each job to start when the previous one ends.

### Hand-picked one-shot images

`configs/office_a2d_fixedshot.yaml` sets `adapt.fixed_shots`, one DSLR file per private class, so every seed uses the same labelled images and `--seed` changes only the training RNG:

| Class | Image | | Class | Image |
|---|---|---|---|---|
| phone | frame_0002.jpg | | ruler | frame_0007.jpg |
| printer | frame_0014.jpg | | scissors | frame_0018.jpg |
| projector | frame_0014.jpg | | speaker | frame_0011.jpg |
| punchers | frame_0003.jpg | | stapler | frame_0019.jpg |
| ring_binder | frame_0007.jpg | | tape_dispenser | frame_0020.jpg |

They were picked by eye (image quality, background, angle, visibility; the most common object model in each class) from sheets made by `python scripts/contact_sheet.py configs/office_a2d.yaml`. This uses the labels of every image in the class, so report these runs separately from the paper's protocol, which averages 5 random draws. `python scripts/shot_check.py runs/office_a2d --config configs/office_a2d_fixedshot.yaml` compares the picks with the random draws without training.

## 7. Getting identical numbers

- The pipeline is deterministic for a given config, seed and machine: re-running the Office A→D main runs from scratch reproduced 70.3 / 48.5 exactly, and two Digits runs gave bit-identical checkpoints.
- **Run through `slurm/run.sbatch`.** It passes `num_workers=6`. Data-augmentation randomness is drawn inside the DataLoader workers, so a different `num_workers`, GPU type or library version can change the numbers slightly.
- Train the source model once per task and keep it: every adaptation config of that task reuses `runs/<task>/source/final.pt` (rect and fixed-shot runs reuse `runs/office_a2d/source`).
- After a run finishes, `ckpt.pt` is deleted (`keep_ckpt: false`) and `final.pt` plus `final_metrics.json` stay. The scripts in section 6 read `final.pt`.
- `--force` reruns a finished stage; without it, finished runs are skipped.

## 8. Tried and not adopted

| Idea | Switch | Result |
|---|---|---|
| Private-guide rectification (Liu et al. 2020) | `configs/office_a2d_rect.yaml` (`adapt.rectify.enabled`) | ALL unchanged (70.3), PRIV +2.3 within noise, shared −2.3 |
| chi-square threshold for negatives | `src_train.neg_mode: chi2` (`*_chi2.yaml`) | identical checkpoints: both rules accept every global-Gaussian sample |
| Backbone at 0.1× lr instead of frozen | `configs/office_a2d_bb01.yaml` | alternative kept for comparison; frozen is the main setting |
| La1 temperature | `adapt.la1_temp` | logits / 100 peaks early then decays; default stays 1 (paper) |
| Gate / bias / boundary offset at prediction | `scripts/posthoc.py` | Table 3: moves images between shared and private without raising ALL. Table 2: gate +2.5 ALL, bias +0.5; both below `lp` |
| k-means re-centring of guides | `scripts/posthoc.py`, `guide_diag.py` (`km`, `km-bal`) | +2 to +3 ALL; below `lp` |
