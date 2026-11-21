# Handover: CIDA reimplementation, Office-31 A→D PRIV gap (state as of 2026-11-21)

This is for the next agent picking up the work. Read it first, then [SPEC.md](SPEC.md). SPEC is the record of every result and decision. This file covers the current state, what has not reached SPEC yet, the open decisions, and the working rules.

## 1. The project in one paragraph
This is a from-scratch PyTorch reimplementation of *Class-Incremental Domain Adaptation* (Kundu et al., ECCV 2020, arXiv 2008.01389). The paper has no official code. The PDF is at `D:\Documents\Deep Learning Project\CIDA\2008.01389v1.pdf` (main text plus supplementary). The repo belongs to teammate **Nikita Bachhas** (`https://github.com/nikita-bachhas/CIDA-reimplementation.git`). The current goal is to understand and close the gap on **Office-31 A→D, one-shot, Table 3 setting**:

| Run | ALL | PRIV | shared |
|---|---|---|---|
| Paper | 72.2 | 72.6 | – |
| Our main runs | 70.3 ± 4.5 | 48.5 ± 8.8 | 93.4 ± 3.5 |
| Hand-picked one-shot images (not the paper's protocol) | 81.9 ± 1.9 | 69.8 ± 2.7 | 94.7 ± 1.2 |

The paper doesn't report shared accuracy; its numbers imply about 71.8. Our ALL matches the paper, but the shared/private split is very different: we are far too good on shared classes and far too weak on private ones.

## 2. Working rules (from the user, durable)
- **Commit and push every change to GitHub without asking**, on branch **`office-rectify-sweep`**. Never commit to `master`, which is Nikita's.
- **Do not commit `posthoc_py/`.** It is a local folder of cluster logs and stays untracked.
- End commit messages with the attribution line the harness gives you.
- Keep the user's personal details (email etc.) out of anything sent externally.
- The user wants concepts explained at undergraduate level when they ask, with plain wording and short tables.
- Report results honestly, seed by seed. Earlier in this work I made three claims that were wrong, and the user caught or questioned them; see section 8.

## 3. Repo and branch state
- Local checkout: `D:\Documents\Deep Learning Project\CIDA\CIDA-reimplementation`. Branch `office-rectify-sweep` tracks origin. The working tree is clean apart from `posthoc_py/`.
- Commits on this branch, on top of Nikita's `d79b5c3`:
  - `9026b03`: private-guide rectification (`cida/rectify.py`, `adapt.rectify.*`, `configs/office_a2d_rect.yaml`), `scripts/guide_diag.py`, and the posthoc offset sweep.
  - `3a9d770`, `6978640`: k-means guide re-centring variants, plus SPEC.
  - `9446bed`: balanced k-means (Sinkhorn), over-clustering, label propagation, and `--label_frac` / `--shots` for guide_diag.
  - `a2318f7`: SPEC write-up of lp and the 3-shot results.
  - `3432a2d`: `scripts/confusion.py` prints a class × seed accuracy table for all classes, and `confusion.json` gains `per_class`.
  - `c719a33`: SPEC records section 7 below (class-averaged reading, class-wise accuracy).
  - `2b6fd33`: `scripts/posthoc.py` gains guide-relabelling prediction rules `nearest` / `km` / `km-bal` / `lp`, class-averaged metrics, and a class-wise table.
- `68a7af6` + next commit: `adapt.fixed_shots`, `contact_sheet.py`, `shot_check.py`, `configs/office_a2d_fixedshot.yaml`.
- Tests: `python -m pytest -q` passes all **48** (about 3 min on a laptop CPU).
- Local machine: Windows, RTX 3070. It has the trained Office A→D checkpoints under `runs/office_a2d/` (`source/`, the 5 main seeds and the 5 rect seeds, each with `final.pt` and `final_metrics.json`) and, since 2026-11-21, the Office-31 images under `data/office31/`. The local PyTorch is CPU-only, so short evaluation scripts run here; training goes to the cluster.

## 4. Cluster workflow (NTU CCDS GPU TC2)
- The cluster pulls from GitHub, so push first, then `git pull` there. Full setup and the cluster rules are in [README.md](README.md). In short: everything runs through `sbatch`, QoS `normal`, one GPU job at a time.
- Job scripts. Arguments pass straight through; each adds `--num_workers 3` itself.
  - `sbatch slurm/run.sbatch configs/<cfg>.yaml [key=value ...]`: source training, then seeds 0–4. Anything already finished is skipped.
  - `sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d --runs adapt_1shot_abl-none_rect [--stage start|final] [--label_frac 0.05 | --shots 3]`
  - `sbatch slurm/posthoc.sbatch runs/office_a2d --runs adapt_1shot_abl-none_rect`
  - `sbatch slurm/confusion.sbatch runs/office_a2d [--runs adapt_1shot_abl-none_rect]`
- `--runs` takes the run-name **prefix before `_seed`**, for example `adapt_1shot_abl-none_rect`. It is **not a glob**; I once suggested `..._seed*` and that was wrong.
- The user copies logs back into `logs/` (for example `logs/cida-confusion_56418.out`) and attaches them in chat.
- When a run finishes, `ckpt.pt` is deleted (`keep_ckpt: false`); `final.pt` stays.

## 5. Setup facts that matter for reading the numbers
- **Classes (A→D, `configs/office31.yaml`).** Shared: indices 0–9 (back_pack … file_cabinet). Private: indices 20–29 (phone, printer, projector, punchers, ring_binder, ruler, scissors, speaker, stapler, tape_dispenser).
  - The paper says private = "ranked 21–31", which is 11 classes, but Table 3 has |Ct| = 20. That is the paper's own inconsistency (SPEC, Inconsistencies item 1). We drop trash_can.
- **Evaluation set.** DSLR has 498 images. The 20 task classes leave 327, and removing the 10 labelled one-shot images leaves **317** (154 shared + 163 private). Evaluation is transductive: we adapt on and score the same unlabelled DSLR images. The paper doesn't specify a split.
- **Metrics** (`cida/metrics.py`):
  - ALL / PRIV / shared are per-image accuracies. The `*_cls` variants are class-averaged.
  - `priv_as_shared` / `shared_as_priv` are the cross-group error rates.
- **Seeds.** `--seed k` sets both the one-shot images and the training RNG (`set_seed(1000 + k)`), so seed-paired comparisons do **not** isolate the shot. Compare variants seed by seed, since the seed spread is large.
- **Backbone.** The ResNet-50 is frozen at its ImageNet weights (our choice, in SPEC). That raised A→D from 49.8 / 33.3 to 70.3 / 48.5.

## 6. What we know so far (details and tables in SPEC "Under test: …")
1. **Rectification** (Liu et al. 2020 prototype rectification of the private guides) gives no net gain: 70.3 / 50.8 / 91.0. It separates the private classes from each other better, but more private images leak into shared classes. Not adopted. The run does exist and was genuinely trained with rectification on: `rect_w0` was logged and the config is `office_a2d_rect.yaml`.
2. **Offset sweep** (`posthoc.py`): moving the private/shared decision boundary can't reach the paper. It costs shared accuracy faster than it gains private, and PRIV levels off around 59–64.
3. **Guide placement is the bottleneck, not the features.** With every guide placed at its true class mean (`oracle-all`, which uses labels), nearest-guide labelling on the same adapted features reaches **88 / 80 / 96**, versus about 70 / 48–51 for the method's guides. The one-shot private guides end up 3–11 class spreads away from their class means.
4. **Label-free relabelling** (`guide_diag.py`, rect runs, final features; ALL / PRIV / shared):

   | Method | Result |
   |---|---|
   | basic | 70.3 / 51.3 / 90.5 |
   | k-means (`km`) | 73.6 / 56.8 / 91.3, better on every seed |
   | balanced k-means (`km-bal`) | 73.4 / 60.0 / 87.7 |
   | **label propagation (`lp`)** | **75.8 / 62.3 / 90.1** (best label-free) |
   | lp from 3 labelled images per class | 77.4 / 64.5 / 89.5 |

   - Averaging 3 labelled images per class (basic) gives 62.5 PRIV, so one-shot lp is worth about two extra labels.
   - With 5% labels, DSLR gets exactly 1 image per class, which is identical to one-shot.
   - Caveat: the variants were chosen after seeing labelled scores, on 5 seeds of one task.
5. **The paper check is done.** The A→D config matches the paper on everything checked: class counts, one-shot / 5% labels, 5 runs, the prediction rule, n = 20%, Adam 1e-4, equal source/negative batches, ResNet-50 ImageNet, and the DSLR augmentation.

## 7. Class-averaged reading and class-wise accuracy (recorded in SPEC on 2026-11-21, kept here for reference)
**a) The paper's ALL / PRIV may be class-averaged.** The paper says "mean target accuracy over Ct (ALL) and C′t (PRIV)", and supp §4 computes precision class-averaged. Our class-averaged numbers:

| Run | ALL (class-avg) | PRIV (class-avg) | shared (class-avg) |
|---|---|---|---|
| Main | **72.4** | **51.9** | 93.0 |
| Rect | 71.9 | 53.2 | 90.6 |

The main runs' class-averaged ALL matches the paper's 72.2. PRIV is still about 20 short either way.

**b) Per-seed final results** (`runs/office_a2d/*/final_metrics.json`, epoch 60):

| Seed | Main ALL / PRIV / shared | Rect ALL / PRIV / shared |
|---|---|---|
| 0 | 76.3 / 57.7 / 96.1 | 77.6 / 63.2 / 92.9 |
| 1 | 69.1 / 49.1 / 90.3 | 67.2 / 47.9 / 87.7 |
| 2 | 64.4 / 38.0 / 92.2 | 64.7 / 42.9 / 87.7 |
| 3 | 72.9 / 56.4 / 90.3 | 72.6 / 54.0 / 92.2 |
| 4 | 68.8 / 41.1 / 98.1 | 69.7 / 46.0 / 94.8 |

**c) Class-wise accuracy** (logs `logs/cida-confusion_56418.out` main, `_56419.out` rect; mean over seeds):

| Class | n | Main | Rect | Where errors go |
|---|---|---|---|---|
| 9 of 10 shared classes | | 93–100 | 94–100 | – |
| desktop_computer (shared) | 15 | 44.0 | 21.3 | both ways with speaker / projector |
| scissors | 17 | 95.3 | 94.1 | – |
| ring_binder | 9 | 66.7 | 64.4 | projector / calculator |
| ruler | 6 | 60.0 | 60.0 | scissors (n = 6, noisy) |
| phone | 12 | 56.7 | 56.7 | calculator, **0% in seeds 3 and 4** |
| stapler | 20 | 55.0 | 73.0 | tape_dispenser |
| printer | 14 | 48.6 | 31.4 | file_cabinet |
| projector | 22 | 42.7 | 58.2 | desktop_computer |
| tape_dispenser | 21 | 39.0 | 30.5 | stapler / desktop_computer |
| punchers | 17 | 32.9 | 44.7 | stapler / tape_dispenser |
| speaker | 25 | 21.6 | 19.2 | desktop_computer (55–66%) |

What the class-wise table shows:
- **Fixed cross-group leaks.** Four private classes each lose images to one shared neighbour: speaker → desktop_computer, projector → desktop_computer, phone → calculator, printer → file_cabinet.
- **Winner-takes-all within private classes.** One guide absorbs its neighbours, most clearly in the punchers / stapler / tape_dispenser triangle:
  - Seed 0: stapler 100%, while punchers gets 12% and tape_dispenser 10%.
  - Seed 4: reversed, stapler 0% (all → tape_dispenser).
  - Seed 2: printer → ring_binder 93%.
- **Seed spread comes from whole classes flipping on or off,** and the same flips mostly repeat across the main and rect configs (phone seeds 3–4, printer seed 2, ring_binder seed 3). That points to the one-shot image, but RNG is confounded with it.
- **Fixing the boundary alone caps out.** `PRIV|priv` (argmax over private logits only) is 60.2 for main and 65.4 for rect. Even a perfect private/shared boundary would leave PRIV short of 72.6.
- **Rectification trades classes,** which is why ALL doesn't move. It helps stapler (+18), projector (+16) and punchers (+12), and hurts printer (−17), tape_dispenser (−9) and desktop_computer (−23).

## 8. Mistakes to avoid repeating
- I first said the PRIV gap was "just the operating point" of the private/shared boundary. The offset sweep disproved that: boundary moves lose accuracy.
- I said the one-shot draw drives the seed spread before there was evidence for it (seed confounds shot and RNG). The fixed-shot runs (2026-11-21) now support it: with the shots fixed, PRIV sd falls from 8.8 to 2.7. The lesson stands: say "not yet shown" until an experiment separates the causes.
- The user once suspected the rect runs were just the baseline. Check with evidence before agreeing or disagreeing: config path, logged `rect_w0`, timestamps.
- Shell heredocs with embedded Python failed in this environment. Write scripts to the scratchpad, or use the Edit tool.

## 9. Open decisions and next steps
0. **Current goal (2026-11-21): reproduce the paper's tables for the report; everything else is out of scope.** Headline numbers = main configuration, i.e. **with all five of our changes on** (frozen backbone, AdaBN, residual fe/fd, linear fe/fd last layer, La1 on B_t) plus the head fix, under the paper's protocol: a random one-shot draw per seed, the paper's prediction rule, last epoch. The vanilla runs (changes off) already exist for every task and serve as the "paper as written" comparison row. Hand-picked shots and `lp` are not part of the headline numbers. Status (commands and job order in REPRODUCE.md, "Report runs"):

   | Paper result | Done, main config | To run | Rough cost per task, 5 seeds (A40) |
   |---|---|---|---|
   | Table 3 one-shot (Cs 10 / Ct 20) | A→D 70.3 / 48.5 | W→D, A→W, D→W, D→A, W→A | DSLR target ~3 h, Webcam ~5 h, Amazon ~15–20 h |
   | Table 2 one-shot (Cs 20 / Ct 31) | none (the hand-picked A→D run is not the protocol) | all 6 | same |
   | Digits | M→U 70.5 / 53.5 | U→M, S→M (old runs predate the deterministic pipeline) | short |
   | Ablations on A→D: Ours-a / b / c, no Ls1, no negatives | none | 5 | ~3 h each |
   | Table 3 few-shot 5% (Ours*) | DSLR targets = the one-shot runs (SPEC inconsistency 4) | A→W, D→W, D→A, W→A | as above |
   | VisDA Syn→Real (Real→Syn last, if time) | none | 1 | days; not measured |
1. ~~Record section 7 in SPEC.md~~ — done (`c719a33`).
2. **lp as the prediction step: done** (logs 56471 main, 56472 rect; SPEC "Other label-free placements"). Rect `lp` reproduces guide_diag exactly (75.8 / 62.3 / 90.1); main `lp` 75.2 / 58.0 / 93.4. PRIV up on 5/5 seeds in both. Helps speaker, punchers, printer; does not fix desktop_computer, tape_dispenser, or phone in seeds 3–4 (0% under the paper rule and lp).
2b. **Hand-picked one-shot images: trained** (job 56473, `logs/cida_56473.out`; SPEC "Under test: hand-picked one-shot images"). `configs/office_a2d_fixedshot.yaml`: **ALL 81.9 ± 1.9 / PRIV 69.8 ± 2.7 / shared 94.7; class-averaged 83.3 / 72.3**, against 70.3 / 48.5 / 93.4 for the main runs (PRIV +12 to +36 on every seed). Not the paper's protocol (picked by eye from all labelled images): report separately from Table 3. Still to run on the cluster: `posthoc.sbatch` and `confusion.sbatch` with `--runs adapt_1shot_abl-none_fixedshot` for the class-wise picture. Tools: `adapt.fixed_shots`, `scripts/contact_sheet.py` (sheets into the gitignored `oneshot_review/`), `scripts/shot_check.py`. The laptop has `data/office31/` (all three domains), CPU-only PyTorch.
2c. **Table 2 setting (20 shared + 11 private).** `office20_a2d_fixedshot.yaml` and `office20_w2d_fixedshot.yaml` (Table 3 picks + `trash_can: frame_0004.jpg`); commands in REPRODUCE.md, "Table 2 setting". **A→D hand-picked ran** (job 56508, `logs/cida_56508.out`): **ALL 74.0 ± 2.1 / PRIV 43.1 ± 5.4 / shared 91.6** (paper 73.3 / 73.1). 54.5% of private images go to shared classes; B_t has ~90 images for 31 classes, smallest class 1. The user ran only the hand-picked config, so there is no random-shot comparison yet. Confusion (log 56525): **PRIV|priv is 78.3** (above the paper's 73.1), so gt separates the private classes well and almost all of the loss is private → shared: letter_tray absorbs punchers / stapler / tape_dispenser, pen absorbs ruler, monitor / laptop_computer absorb ring_binder (mostly new shared classes, absent in Table 3). **Posthoc done** (log 56528, SPEC table under "Table 2 setting"): **`lp` gives 82.1 ± 0.8 / 68.0 ± 2.8 / 90.1** (class-averaged 81.2 / 65.9), PRIV +22 to +31 on 5/5 seeds, shared −1.5. Adopted as the stated post-hoc step here, reported beside the paper rule. It beats the label-chosen offset on every seed; `bias` under-corrects (predicts 17.4% private, true 36.3%); the gate gives +10.9 PRIV here (useless in Table 3). Still weak under lp: stapler 30, ruler 20 (n = 6). Next: random-shot `configs/office20_a2d.yaml`, lp as B_t pseudo-labeller in training, class-balanced B_t (proposed, not implemented). W→D not run.
3. **lp in the training refresh: implemented, not yet run** (`adapt.pseudo_label=lp`; SPEC "Under test: lp pseudo-labels in training"). lp labels the refresh, B_t = top 20% per class by lp confidence, P_t and La1 use the lp labels. Test before the report queue, one job: `sbatch --export=ALL,CONFIGS="configs/office_a2d_ft.yaml configs/office20_a2d.yaml configs/office20_a2d_lpref.yaml" slurm/run.sbatch`. On 2026-11-22 the user swapped the lowest-priority run, the Table 3 lp test (`office_a2d_lpref.yaml`, kept for later), for `office_a2d_ft.yaml`: Table 3 A→D with the backbone **not frozen** (Adam 1e-4 on the whole network in both stages, own source model in `runs/office_a2d_ft/`), all other improvements on. It isolates the freezing choice, seed-paired with the main runs (70.3 / 48.5 / 93.4); the earlier 49.8 → 70.3 jump mixed freezing with the head fix. So the lp decision now rests on Table 2 only (lpref vs random-shot `office20_a2d`, seed by seed); run `office_a2d_lpref` before adopting it for all tasks. If it wins, it becomes part of the main configuration (set `pseudo_label: lp` in `base.yaml`) before the report queue runs; VisDA would keep `nearest` (dense graph limit). lp's B_t private precision on Table 3 (82.3 vs 79.5) means a gain there isn't guaranteed.
4. **Separating shot from RNG: partly answered by 2b.** With the shots fixed, PRIV spread over seeds is sd 2.7 (67.5–74.2), against 8.8 (38.0–57.7) with a random shot per seed, so the shot drives most of the spread. Measured for one (good) set of shots only; a fixed *random* set (e.g. seed 2's shots with seeds 0–4) would show whether bad shots are also stable.
5. **Still to look at:**
   - The desktop_computer / speaker / projector cluster: inspect the DSLR images, which may be ambiguous.
   - In the rect runs, desktop_computer loses about 43% of its images to *other shared* classes. Not explained yet.
6. Not implemented (SPEC "Not implemented yet"): Ours-d, the A→D→W two-step upgrade, and baseline numbers from Table 3. There is no main-configuration VisDA run yet; Nikita ran the vanilla preset on Digits, Office (Table 2 and 3) and VisDA.

## 10. Key files
| File | Purpose |
|---|---|
| `adapt.py`, `train_source.py` | the two training stages |
| `cida/rectify.py` | `rectify_guides`, `sinkhorn_assign`, `kmeans_guides`, `overcluster`, `label_propagation` |
| `cida/metrics.py` | ALL / PRIV / shared, per-image and class-averaged |
| `scripts/guide_diag.py` | nearest-guide labelling under guide variants (`VARIANTS`), asymmetry, B_t precision; `--stage`, `--label_frac`, `--shots`, `--km_iters` |
| `scripts/posthoc.py` | prediction-step variants and the q offset sweep (`print_sweep`) |
| `scripts/confusion.py` | confusion matrix, private-class error destinations, class × seed table |
| `scripts/aggregate.py`, `scripts/trajectory.py` | seed means vs paper; per-epoch curves |
| `configs/office_a2d.yaml`, `office_a2d_rect.yaml`, `office31.yaml`, `office20*.yaml` | A→D main / rect / class split / Table 2 setting |
| `configs/office_a2d_fixedshot.yaml`, `office20_a2d_fixedshot.yaml`, `office20_w2d_fixedshot.yaml` | hand-picked one-shot DSLR images (Table 3 / Table 2) |
| `REPRODUCE.md` | every change on top of the paper, config switches, commands, expected results |
| `tests/test_rectify.py`, `tests/test_smoke.py` | unit tests for the guide methods; end-to-end smoke tests for every script |
| `logs/` | cluster logs the user copied back (guide_diag 56360/56361/56363, confusion 56418/56419, …) |

`logs/` and `runs/` are **gitignored**: they exist only on the user's laptop and on the cluster, not on GitHub. The key numbers from them are copied into this file and SPEC.md. On another machine, ask the user for the log files.
