# Handover: CIDA reimplementation, Office-31 PRIV gap (state as of 2026-11-23)

This is for the next agent picking up the work. Read section 0 first, then the rest of this file, then [SPEC.md](SPEC.md). SPEC is the record of every result and decision. This file covers the current state, what has not reached SPEC yet, the open decisions, and the working rules.

## 0. Start here (state on 2026-11-23)
**Goal:** reproduce the paper's tables for the report (section 9, item 0). The side question now under test: **is the source model what holds PRIV back?** (evidence in section 7b).

**Decision (user, 2026-11-23): the report uses the off-protocol configuration only, from now on.** Hand-picked one-shot images + `pseudo_label: lp` + 20 adaptation epochs (+ step-matched source training where the source is small), because it gets closer to the paper than the paper's protocol does. Paper-protocol runs (random shots, 60 epochs) are no longer run; existing ones stay in SPEC for the record. Current off-protocol results (Table 2): A→D 82.8 / 71.2 / 89.4 (job 56686), W→D 88.5 / 74.1 / 96.7 (job 56769, 180-epoch source). Hand-picked images exist only for DSLR targets (A→D, W→D); other tasks would need Webcam or Amazon picks first.

**Update (later on 2026-11-23):** the srcft100 files were pushed (a82d859) and seed 0 ran as job 56726; results in item 3c and SPEC. The two sections below are kept for the record.

**Was blocking (resolved): the new W→D config was not on GitHub.**
- The working tree has a new `configs/office20_w2d_srcft100.yaml` (untracked) and uncommitted edits to `HANDOVER.md`, `REPRODUCE.md`, `SPEC.md` and `tests/test_smoke.py`. GitHub's `office-rectify-sweep` is still at `a2b382f`.
- Because of that, two cluster jobs failed at once with `no such config: configs/office20_w2d_srcft100.yaml`: **56715** and **56717** (`.err` files in `D:\Documents\Deep Learning Project\CIDA\`). Nothing ran; no GPU time used.
- All 57 tests pass with these changes (`C:\Python312\python -m pytest -q`, about 3 min).
- **Do this first:**
  1. Ask the user for a yes, then commit those five files and push (section 2 explains why the yes is needed). Never add `posthoc_py/`.
  2. Tell the user to run on the cluster: `git checkout office-rectify-sweep && git pull`
  3. Then: `ls configs/office20_w2d_srcft100.yaml && sbatch --export=ALL,SEEDS="0" slurm/run.sbatch configs/office20_w2d_srcft100.yaml` (the `ls` stops a third failed submission).

**The run waiting to go (section 9, item 3c; SPEC "Under test: W→D source model with the ResNet unfrozen, 100 epochs"):**
- Source: ResNet-50 **unfrozen** (`backbone_lr_mult: 1.0`), **100 epochs + 1 warm-up**, own folder `runs/office20_w2d_srcft100/` (the existing `runs/office20_w2d/source` is untouched).
- Adaptation: identical to the W→D lp20 run of job 56686 (hand-picked shots, `pseudo_label: lp`, 20 epochs), ResNet frozen at the new source weights (BatchNorm statistics still adapt via AdaBN). Tag `fixedshot_lp20_srcft100`.
- User's choices: 100 source epochs "so Ls1 can stabilise better", and "1 epoch for now", which was read as **1 seed (seed 0)**. If the user meant 1 adaptation epoch, the override is `adapt.epochs=1`; confirm with them if unsure.
- It changes two things against job 56686 (freezing and 50 → 100 source epochs), so a difference can't be pinned on either alone. A 50-epoch unfrozen run would separate them (not set up).
- Expected time on the A40: source ~10 min (estimate: frozen Webcam rate × 1.4 slowdown measured on Amazon × 101 epochs), adaptation ~12 min per seed.

**When its log arrives** (the user attaches `cida_<job>.out` from `D:\Documents\Deep Learning Project\CIDA\`; copy it into `logs/`, which is gitignored):
- Source stage: does `loss_ls1` settle (A→D unfrozen never did: median 38 in epochs 41–51 against 5.8 frozen; W→D frozen ends near 2.9)? Does `loss_ls2` fall below the frozen run's 2.17, and `neg_detected` rise above 38.9%? `src_acc` won't tell much: the frozen W→D source model already reached 100%.
- Adaptation: compare with job 56686 W→D seed 0: ALL / PRIV / shared **82.3 / 62.1 / 93.9**, priv_as_shared 17.5.
- If it looks sane: `sbatch --export=ALL,STAGES=adapt,SEEDS="1 2 3 4" slurm/run.sbatch configs/office20_w2d_srcft100.yaml`, then record seed-by-seed results in SPEC (under the "Under test" entry) and in item 3c here, and commit.

**Offered to the user, not done yet:**
- `scripts/feature_probe.py` on the Table 2 A→D and W→D source models: ImageNet features vs `fs`, private accuracy and private → shared errors. The Table 3 probe (section 7b) is the main evidence against the source model; it has never been run on Table 2 or W→D. Check the script's arguments before writing the command.
- `sbatch slurm/confusion.sbatch runs/office20_w2d --runs adapt_1shot_abl-none_fixedshot_lp20` to confirm that W→D's remaining PRIV loss is mostly private ↔ private confusion (section 7b).

**Latest results (details in item 3b and SPEC):** job 56686, hand-picked shots + lp + 20 epochs, Table 2: **A→D 82.8 / 71.2 / 89.4** (paper 73.3 / 73.1), **W→D 85.4 / 69.0 / 94.8** (paper 89.5 / 88.4). Against job 56614 (W→D lp, random shots, epoch 60: 81.4 / 63.2 / 91.8), W→D gained +4.0 / +5.8 / +3.0, but against that run's epoch 20 only +2.3 / +2.1 / +2.5: most of the gain is from stopping at 20 epochs, not from the shots.

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
  - **In practice (2026-11-23):** the Claude Code desktop app's auto-mode permission check blocks `git commit` / `git push` unless the user says yes in the same turn. It has blocked three attempts without one and allowed the one where the user had just said "Yes, commit and push". So finish the work, then ask for a yes (a wrap-up button works) and commit on that yes, or give the user the exact command. Don't retry a blocked commit in the same turn.
- **Do not commit `posthoc_py/`.** It is a local folder of cluster logs and stays untracked.
- End commit messages with the attribution line the harness gives you.
- Keep the user's personal details (email etc.) out of anything sent externally.
- The user wants concepts explained at undergraduate level when they ask, with plain wording and short tables.
- Report results honestly, seed by seed. Earlier in this work I made three claims that were wrong, and the user caught or questioned them; see section 8.

## 3. Repo and branch state
- Local checkout: `D:\Documents\Deep Learning Project\CIDA\CIDA-reimplementation`. Branch `office-rectify-sweep` tracks origin. `posthoc_py/` is always untracked; on 2026-11-23 there are also uncommitted changes (section 0).
- Commits on this branch, on top of Nikita's `d79b5c3`:
  - `9026b03`: private-guide rectification (`cida/rectify.py`, `adapt.rectify.*`, `configs/office_a2d_rect.yaml`), `scripts/guide_diag.py`, and the posthoc offset sweep.
  - `3a9d770`, `6978640`: k-means guide re-centring variants, plus SPEC.
  - `9446bed`: balanced k-means (Sinkhorn), over-clustering, label propagation, and `--label_frac` / `--shots` for guide_diag.
  - `a2318f7`: SPEC write-up of lp and the 3-shot results.
  - `3432a2d`: `scripts/confusion.py` prints a class × seed accuracy table for all classes, and `confusion.json` gains `per_class`.
  - `c719a33`: SPEC records section 7 below (class-averaged reading, class-wise accuracy).
  - `2b6fd33`: `scripts/posthoc.py` gains guide-relabelling prediction rules `nearest` / `km` / `km-bal` / `lp`, class-averaged metrics, and a class-wise table.
- `68a7af6` + next commit: `adapt.fixed_shots`, `contact_sheet.py`, `shot_check.py`, `configs/office_a2d_fixedshot.yaml`.
- Later commits: `de1f983` (lp20 configs), `a2b382f` (job 56686 recorded; the current tip on GitHub). Uncommitted: see section 0.
- Tests: `python -m pytest -q` passes all **57** (about 3 min on a laptop CPU).
- **Which Python:** `C:\Python312\python` (Git Bash: `/c/Python312/python`) has torch 2.10 CPU. Run it **without** `-I`: torch is not found in isolated mode, and the Windows Store `python` alias is a stub. Run repo scripts from the repo folder.
- Local machine: Windows, RTX 3070. It has the trained Office A→D checkpoints under `runs/office_a2d/` (`source/`, the 5 main seeds and the 5 rect seeds, each with `final.pt` and `final_metrics.json`) and, since 2026-11-21, the Office-31 images under `data/office31/`. The local PyTorch is CPU-only, so short evaluation scripts run here; training goes to the cluster.

## 4. Cluster workflow (NTU CCDS GPU TC2)
- The cluster pulls from GitHub, so push first, then `git pull` there. Full setup and the cluster rules are in [README.md](README.md). In short: everything runs through `sbatch`, QoS `normal`, one GPU job at a time.
- Job scripts. Arguments pass straight through; each adds `--num_workers 3` itself.
  - `sbatch slurm/run.sbatch configs/<cfg>.yaml [key=value ...]`: source training, then seeds 0–4. Anything already finished is skipped.
  - `sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d --runs adapt_1shot_abl-none_rect [--stage start|final] [--label_frac 0.05 | --shots 3]`
  - `sbatch slurm/posthoc.sbatch runs/office_a2d --runs adapt_1shot_abl-none_rect`
  - `sbatch slurm/confusion.sbatch runs/office_a2d [--runs adapt_1shot_abl-none_rect]`
- `--runs` takes the run-name **prefix before `_seed`**, for example `adapt_1shot_abl-none_rect`. It is **not a glob**; I once suggested `..._seed*` and that was wrong.
- The user copies logs back into `logs/` (for example `logs/cida-confusion_56418.out`) and attaches them in chat. Recently the logs land in the parent folder `D:\Documents\Deep Learning Project\CIDA\` (e.g. `cida_56686.out`, `cida_56715.err`) and the user attaches them **with no message**: read them against this file, summarise seed means and spread against the paper and the matching earlier job, and copy the `.out` into `logs/`. Logs are JSON lines, one per epoch, ending with an `"event": "done"` line per seed. `cida_56686_w2d.out` in the parent folder is just the W→D part of job 56686.
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
  - **Paper check (2026-11-23):** the paper never says the backbone is frozen. Supp §6.1 (p. 19) only says "ResNet-50, pretrained on ImageNet" for fs. Algorithm 2 (p. 29) updates all of θ_fs with `Adam{fs}` / `Adam{fs, gs}`, and Algorithm 4 (p. 31) all of θ_ft, so read literally the whole network trains. The only things it calls frozen are gs during adaptation (p. 8) and the shared guides in Ours-a. State freezing as our change in the report.
  - The earlier unfrozen run (job 56572, `office20_a2d_ft`) unfroze **both** stages (51 source epochs, 60 adaptation epochs). No run has yet unfrozen the source stage alone; `office20_w2d_srcft100` is the first.
- **With `fixed_shots`** the one-shot images are the same for every seed, so the seed changes only the training RNG.
- **Table 2 W→D sizes** (counted in `data/office31/`): Webcam source 528 images in the 20 shared classes (12–43 per class); DSLR target 498 (310 shared + 188 private); scored 487 = 310 shared + 177 private (the 11 one-shot images are excluded). One private image ≈ 0.56 PRIV points. DSLR private classes: ruler 7, ring_binder 10, phone 13, printer 15, trash_can 15, punchers 18, scissors 18, stapler 21, tape_dispenser 22, projector 23, speaker 26.
- **B_t** (`refresh()` in `adapt.py`): rebuilt after every epoch from the **target** images only. Each image gets a pseudo-label (lp or nearest guide), and for each pseudo-label class the top `max(1, int(0.20 × group size))` by confidence are kept. With about 16 DSLR images per class that is mostly 3 per class, hence `B_size` ≈ 84–90 for 31 classes.

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

## 7b. Source-model diagnostics (2026-11-23, from existing logs; not yet in SPEC)
Why the source model is a suspect for the PRIV gap. All numbers come from the source stage of earlier jobs (`logs/cida_56508.out`, `56572`, `56614`, `56615`). Source training = 1 warm-up epoch + 50 epochs (`configs/office31.yaml`), source_seed 0, trained once per task and reused by every adaptation seed and later config. **`src_acc` is measured on the same images it trains on** (`src_train_split: all`, `src_eval_split: all`): it is a fit measure, not generalisation.

**a) The feature probe (already in SPEC, row "ResNet backbone frozen").** Table 3 A→D, one-shot nearest-guide accuracy on DSLR, priv|priv / shared / all: ImageNet features **72.4** / 93.5 / 71.3; frozen source `fs` **46.3** / 90.9 / 58.0; fully fine-tuned extractor 27.0; adapted `ft` 67.0. Source training removes most of what separates the unseen private classes; adaptation wins much of it back but not all.

**b) Source accuracy (%)**

| Job / task | Source | ResNet | Ep 1 | Ep 10 | Ep 20 | Ep 51 | First ≥ 99.5 |
|---|---|---|---|---|---|---|---|
| 56572 A→D `_ft` | Amazon | unfrozen | 94.1 | 98.4 | 99.7 | 99.9 | ep 19 |
| 56508 A→D, 56615 A→W (identical) | Amazon | frozen | 87.3 | 94.2 | 96.3 | 98.3 | never |
| 56614 W→D | Webcam | frozen | 92.0 | 99.6 | 99.8 | 100.0 | ep 8 |
| 56615 D→W | DSLR | frozen | 71.6 | 99.7 | 99.7 | 100.0 | ep 7 |

**c) Second objective Ls2** (CE on source + CE on negatives labelled "unknown"; trains fs, gs). Source accuracy is high, so Ls2 mostly measures the negatives (ln 21 ≈ 3.0 = even guess over 21 outputs).

| Job / task | ResNet | Steps/epoch | Ls2 ep 10 / 20 / 30 / 51 | neg_detected ep 20 / 51 |
|---|---|---|---|---|
| 56508 A→D | frozen | ~55 | 3.28 / 2.00 / 1.21 / 0.61 | 58.3 / 97.6 |
| 56572 A→D `_ft` | unfrozen | ~55 | 2.96 / 1.43 / 0.77 / 0.33 | 76.4 / 99.4 |
| 56614 W→D | frozen | 16 | 4.04 / 3.42 / 3.11 / 2.17 | 1.2 / 38.9 |
| 56615 D→W | frozen | ~9 | 4.27 / 3.77 / 3.61 / 3.10 | 0.0 / 2.3 |

- `neg_detected` = share of 1,024 fresh negatives (from the end-of-epoch refit) that gs puts in "unknown" (`train_source.py` lines 43–44). It is 0% up to epoch 5 everywhere.
- W→D and D→W have too few steps (Ls2 runs on every second step, so W→D gets about 400 Ls2 steps in 51 epochs): their "unknown" class is unfinished.

**d) First objective Ls1 (cluster compaction / separability, Eq. 3; median of epoch averages)**

| Epochs | A→D frozen | A→D unfrozen | W→D frozen |
|---|---|---|---|
| 2 | 19.7 | 27.9 | 804 |
| 3–10 | 9.7 | 65 | 10.5 |
| 11–20 | 8.6 | 91 | 5.0 |
| 21–30 | 7.7 | 53 | 3.3 |
| 41–51 | 5.8 | 38 | 2.9 |

- Frozen: clusters tighten steadily (W→D has 5 epochs at exactly 0.0, i.e. huge margins). Unfrozen: Ls1 rises after warm-up and never settles, ending ~6× higher, even though its Ls2 and source accuracy are better.
- Untested reading: with the ResNet free, features move within each epoch while the Gaussians are refit only at its end, so the prototypes lag. Adaptation relies on those Gaussians (Lr1/Lr2 proxy samples, shared guides `fe(μ_s)`), which could explain part of the unfrozen A→D drop (ALL 66.1 → 57.7 in job 56572).

**e) Where W→D's PRIV goes (job 56686, means):** A→D: 71.2% correct, 24.4% → shared, ~4% → wrong private. W→D: 69.0% correct, 13.3% → shared, **~18% → wrong private**. W→D's main loss looks like private ↔ private confusion, not leakage into shared classes (to be confirmed with `confusion.sbatch`, section 0). Hand-picked shots helped A→D much more than W→D (+14.7 vs +2.1 PRIV over lp + random shots at epoch 20), and fixing the shots did not reduce W→D's PRIV spread (5.6 → 6.0), so on W→D the shot is not what drives the variation.

**f) Timing (A40, 51 source epochs):** Amazon unfrozen 9.9 min (0.194 min/epoch), Amazon frozen 7.2 (0.140), Webcam frozen 3.6 (0.071), DSLR frozen 4.2 (0.082). Unfreezing cost ×1.4 per epoch on Amazon. Estimates for 101 unfrozen epochs: Amazon ~20 min, Webcam ~10, DSLR ~11–12.

## 8. Mistakes to avoid repeating
- I first said the PRIV gap was "just the operating point" of the private/shared boundary. The offset sweep disproved that: boundary moves lose accuracy.
- I said the one-shot draw drives the seed spread before there was evidence for it (seed confounds shot and RNG). The fixed-shot runs (2026-11-21) now support it: with the shots fixed, PRIV sd falls from 8.8 to 2.7. The lesson stands: say "not yet shown" until an experiment separates the causes.
- The user once suspected the rect runs were just the baseline. Check with evidence before agreeing or disagreeing: config path, logged `rect_w0`, timestamps.
- Shell heredocs with embedded Python failed in this environment. Write scripts to the scratchpad, or use the Edit tool.

## 9. Open decisions and next steps
0. **Superseded on 2026-11-23 (see the decision in section 0): headline numbers are now the off-protocol configuration.** Earlier goal (2026-11-21): reproduce the paper's tables for the report; everything else is out of scope. Headline numbers = main configuration, i.e. **with all five of our changes on** (frozen backbone, AdaBN, residual fe/fd, linear fe/fd last layer, La1 on B_t) plus the head fix, under the paper's protocol: a random one-shot draw per seed, the paper's prediction rule, last epoch. The vanilla runs (changes off) already exist for every task and serve as the "paper as written" comparison row. Hand-picked shots and `lp` are not part of the headline numbers. Status (commands and job order in REPRODUCE.md, "Report runs"):

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
3. **Job 56572 results (Table 2 A→D, random shots, seeds 0–2; SPEC "Under test: lp pseudo-labels in training" → Results):** frozen + nearest **66.1 / 28.2 / 87.6**; frozen + lp **76.0 / 56.9 / 87.0** (PRIV +18.6 / +18.0 / +49.2, ALL up on 3/3 seeds, shared −0.6); **not frozen** + nearest **57.7 / 26.7 / 75.4** (ALL −8.4, shared −12.2, PRIV mixed) → keep the backbone frozen. lp passes the adoption rule on Table 2. **Next:** the Table 3 lp pair, `sbatch --export=ALL,STAGES=adapt,SEEDS="0 1 2" slurm/run.sbatch configs/office_a2d_lpref.yaml` (~2 h; compare with main seeds 0–2: 76.3 / 57.7, 69.1 / 49.1, 64.4 / 38.0). If PRIV also rises there without a large shared drop, set `pseudo_label: lp` in `base.yaml` before the report queue (VisDA keeps `nearest`).
3a. **Table 2 with lp, W→D / A→W / D→W, 5 seeds** (jobs 56614, 56615; SPEC "Table 2 with lp pseudo-labels"): ALL / PRIV / shared **W→D 81.4 / 63.2 / 91.8** (paper 89.5 / 88.4), **A→W 73.9 / 51.9 / 84.5** (paper 63.6 / 62.6), **D→W 77.3 / 56.2 / 87.6** (paper 80.3 / 79.4); A→D stays at 3 seeds (76.0 / 56.9 / 87.0, paper 73.3 / 73.1), as the user chose. PRIV is 11–25 points short on every task (private → shared leak 22–36%); ALL is above the paper on A→D and A→W. Late drift of 2–4 points after epoch 20 on all three new tasks; last epoch kept. Not run: D→A and W→A (~17 h each, user deferred), A→D seeds 3–4.
3b. **Hand-picked shots + lp + 20 epochs, Table 2 A→D and W→D, 5 seeds** (job 56686, `logs/cida_56686.out`; SPEC "Under test: hand-picked images with lp, 20 epochs" → Results). ALL / PRIV / shared: **A→D 82.8 ± 1.7 / 71.2 ± 2.3 / 89.4** (paper 73.3 / 73.1; class-averaged 82.6 / 69.5), **W→D 85.4 ± 2.0 / 69.0 ± 6.0 / 94.8** (paper 89.5 / 88.4; class-averaged 85.6 / 67.3). A→D is the closest we have come to the paper's PRIV (−1.9): +14.7 PRIV over lp + random shots at epoch 20, and the private → shared leak is down to 24.4% (from 54.5% for nearest + hand-picked). W→D gains only +2.1 PRIV over lp + random shots and stays 19.4 short; its PRIV spread (62.1–77.4) comes from the training RNG alone, since the shots are fixed. ALL peaks before epoch 20 on every seed (A→D by 0.9 on average, W→D by 2.4); last epoch kept. Not the paper's protocol (hand-picked shots, epoch count read off test curves): report apart from Table 2.
3c. **Next: W→D with an unfrozen-backbone source model, 100 source epochs** (`configs/office20_w2d_srcft100.yaml`; SPEC "Under test: W→D source model with the ResNet unfrozen, 100 epochs"; commands in REPRODUCE.md). The user chose 100 source epochs so Ls1/Ls2 can settle on W→D (frozen 50-epoch source: Ls2 2.17, 38.9% of negatives detected) and seed 0 first. Adaptation identical to the lp20 run in 3b (ResNet frozen at the new source weights), so it tests whether the source model holds PRIV back; it changes freezing and source length together. First run to unfreeze the source stage alone. **Seed 0 ran** (job 56726, SPEC entry → Results): **84.6 / 59.3 / 99.0** against 82.3 / 62.1 / 93.9 for job 56686 seed 0. Ls2 settled (0.70, 96% of negatives detected) but Ls1 never did; shared near-perfect, PRIV −2.8 (within W→D's seed noise), private loss moved from shared to wrong-private (25.4%). Pseudo-labels start at 67.1 (frozen 82.3) and PRIV is still rising at epoch 20, so the 20-epoch cut may be early. Options offered: 60 adaptation epochs on seed 0, seeds 1–4, a frozen 100-epoch source run.
3d. **W→D with a frozen source model, 180 source epochs** (`configs/office20_w2d_src180.yaml`; SPEC "Under test: W→D source model trained longer, ResNet frozen"). The user chose this over the srcft100 follow-ups: ~2,880 source steps (Amazon's 51 epochs give ~2,850), ResNet frozen, adaptation as in 3b, seeds 0–4. Separates source length from unfreezing when read beside 3c. **Ran** (job 56769, SPEC entry → Results): **88.5 / 74.1 / 96.7** against 85.4 / 69.0 / 94.8 (ALL up 5/5 seeds, PRIV +5.1, sd 6.0 → 3.5; paper 89.5 / 88.4). Ls1 and Ls2 both settle (Ls2 0.31, 99% negatives). Private → wrong-private falls 17.6 → 11.0%. Source length helps; unfreezing (3c) is what cost PRIV. Next candidate: step-matched source training for the small-source tasks in the paper-protocol runs (random shots, lp, epoch 60).
3e. **Step-matched source training, paper protocol** (`configs/office20_w2d_lpref_srcmatch.yaml`, 180 source epochs; `configs/office20_d2w_lpref_srcmatch.yaml`, 317; SPEC "Under test: step-matched source training, paper protocol"). The user said yes on 2026-11-23. Same as jobs 56614 / 56615 (random shots, lp, 60 epochs, seeds 0–4, same shots per seed) except the source model gets ~2,850 steps like Amazon's. Compare with W→D 81.4 / 63.2 / 91.8 and D→W 77.3 / 56.2 / 87.6. If it helps, W→A and D→A get the same rule. **Dropped, not run:** the user chose off-protocol only (section 0).
3f. **A→D with a frozen source model, 150 source epochs** (`configs/office20_a2d_src150.yaml`; SPEC "Under test: A→D source model trained longer, ResNet frozen"). The user chose this on 2026-11-23 over post-hoc lp. Off-protocol configuration (hand-picked shots, lp, 20 epochs), seeds 0–4, ~8,450 source steps (from ~2,850). Compare with job 56686 A→D 82.8 / 71.2 / 89.4. Expect a smaller gain than W→D's: A→D's source already detected 98% of negatives. Not yet run.
3g. **Final run on A→W, D→W, D→A, W→A** (`configs/office20_<task>_final.yaml`; SPEC "Final run: off-protocol configuration on the remaining Table 2 tasks"; commands in REPRODUCE.md). The user asked on 2026-11-23 to use the off-protocol setup for every remaining task and compare. New hand-picked Webcam and Amazon images (picked from contact sheets in the gitignored `oneshot_review/`, montage `oneshot_review/picks_webcam_amazon.png`), lp, 20 epochs, longer source per domain (Amazon 150, Webcam 180, DSLR 317 epochs). The Amazon-source length waits on 3f; if A→D src150 doesn't help, A→W could go back to 50. **Webcam targets ran** (job 56786): **A→W 80.6 / 65.5 / 88.0** (paper 63.6 / 62.6: above on both), **D→W 85.1 / 63.8 / 95.4** (paper 80.3 / 79.4: ALL above, PRIV −15.6). D→W peaks at epochs 6–14. D→A (job 56787) was still training its source model at the last log copy; A→D src150 not reported yet. **W→A (job 56791) crashed** at the start of adaptation: AdaBN's last batch over Amazon's 2,817 images holds one image (2,817 = 44 × 64 + 1), and BatchNorm cannot train on one sample. Fixed in `adabn()` by skipping batches of fewer than 2 images. Earlier DSLR and Webcam targets are unaffected. D→A (job 56787) fails the same way after its source stage. Resubmit both after `git pull`; their source models are reused. **Amazon targets ran** after the fix (jobs 56808, 56809): **W→A 63.8 / 49.8 / 71.5** (paper 63.7 / 62.8), **D→A 62.9 / 49.1 / 70.6** (paper 64.1 / 64.3): ALL at the paper, PRIV 13–15 short, shared stuck near 71 (barely above epoch 0). Only A→D src150 is still pending.
3i. **Final run in the Table 3 setting (10 shared + 10 private), all six tasks** (`configs/office_<task>_final.yaml`; SPEC "Final run: off-protocol configuration in the Table 3 setting"; commands in REPRODUCE.md). The user asked on 2026-11-23 for the Table 2 final-run results in the Table 3 setting too. Same configuration (hand-picked shots without trash_can, lp, 20 epochs, ResNet frozen); source epochs raised to keep the Table 2 step counts (Amazon 340, Webcam 410, DSLR 712). Each config trains its own source model. Not yet run.
3h. **Deferred until the final-run results are in (user, 2026-11-23): augmented private guides.** Average each private guide over ~8 augmented copies of its one labelled image (flip, mild crop; careful with colour jitter, since evaluation uses the plain transform). Today the guide is `ft` of the plain image (`adapt.py` `private_guides` / `labelled_batch`). Plan: off-by-default switch `adapt.guide_aug_views`; test on W→D first, built on `configs/office20_w2d_src180.yaml` with only the tag changed so its source model is reused, seed-paired with job 56769 (88.5 / 74.1 / 96.7). Expected 0–3 PRIV, cost ~1.5–2× per seed.
   Background: **lp in the training refresh** (`adapt.pseudo_label=lp`; SPEC "Under test: lp pseudo-labels in training"). lp labels the refresh, B_t = top 20% per class by lp confidence, P_t and La1 use the lp labels. Test before the report queue, one job: `sbatch --export=ALL,SEEDS="0 1 2",CONFIGS="configs/office20_a2d_ft.yaml configs/office20_a2d.yaml configs/office20_a2d_lpref.yaml" slurm/run.sbatch` (seeds 0–2, all three in the Table 2 setting, |Cs| = 20, |Ct| = 31). On 2026-11-22 the user swapped the lowest-priority run, the Table 3 lp test (`office_a2d_lpref.yaml`, kept for later), for an unfrozen-backbone run, then asked for it in the Table 2 split like the others: `office20_a2d_ft.yaml`, backbone **not frozen** (Adam 1e-4 on the whole network in both stages, own source model in `runs/office20_a2d_ft/`), all other improvements on. It isolates the freezing choice, seed-paired with `office20_a2d` (frozen) in the same job; the earlier Table 3 49.8 → 70.3 jump mixed freezing with the head fix. So the lp decision now rests on Table 2 only (lpref vs random-shot `office20_a2d`, seed by seed); run `office_a2d_lpref` before adopting it for all tasks. If it wins, it becomes part of the main configuration (set `pseudo_label: lp` in `base.yaml`) before the report queue runs; VisDA would keep `nearest` (dense graph limit). lp's B_t private precision on Table 3 (82.3 vs 79.5) means a gain there isn't guaranteed.
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
