# CIDA reimplementation spec

Paper: Kundu, Venkatesh, Venkat, Revanur, Babu. *Class-Incremental Domain Adaptation.* ECCV 2020 (arXiv 2008.01389), plus its supplementary material.
This file records what the paper fixes, what we chose ourselves, and where the paper contradicts itself. The report's "deviations" section should be built from it.

## Fixed by the paper

| Item | Value | Source |
|---|---|---|
| Optimiser | Adam, lr 1e-4, a separate optimiser per loss | §4b, supp §3.2 and §3.4 |
| Source warm-up | 1 epoch of plain cross-entropy before fitting prototypes | supp Algo 2, lines 2–5 |
| Source losses | Ls1 (softmax over class prototype densities, trains fs) and Ls2 (CE on source + CE on negatives labelled \|Cs\|, trains fs and gs), alternated with `iter % 2` | Eq. 3–4, Algo 2 |
| Batch composition | Nsrc = Nneg | §4b, Fig. 4C |
| Negatives | Sample the global Gaussian; keep a sample if its density under its argmax class is below the density at μ + 3√λ_max·ν_max | supp Algo 1 |
| Prototype refresh | End of every epoch (source and target) | §3.1c, Algo 2 and Algo 4 |
| fe, fd initialisation | Pretrained to near-identity on proxy-source samples, one Adam each | §3.2e, Algo 4 lines 3–7 |
| Adaptation losses | Round-robin over [Lr1, Lr2, Lc, La1, La2] (`iter % 5`). Each trains: Lr1 → {fe, fd, gt}; Lr2 → {fe, fd}; Lc → {ft, gt}; La1 → {ft}; La2 → {fe, ft}. gs stays frozen. | Algo 4 lines 8–10 |
| Guides | fe(μ_s^c) for shared classes; ft(one-shot image) for private classes (class mean for few-shot) | §3.2c, §4.1b |
| Confident set | n = 20% nearest per guide | §4b, Algo 3 |
| Prediction | argmax over [gs∘fd∘ft restricted to Cs ‖ gt∘ft] | §4b |
| Evaluation | ALL (over Ct) and PRIV (over C′t), mean of 5 runs with different one-shot samples | §4a |
| Architectures | supp Table 2 (see `cida/models.py`) | supp §6 |
| Digits | 0–4 shared, 5–9 private; RGB input (greyscale copied to 3 channels) | supp §5, §6.2 |
| VisDA | Shared = {bicycle, bus, car, motorcycle, train, truck}; the other 6 classes are private | supp §5 |
| Office augmentation | Flip, rotation (±5/10/15°), crop, colour jitter, RGB channel swap | supp Fig. 2 |

## Our choices (not specified in the paper)

| Item | Choice | Why |
|---|---|---|
| "Likelihood" inside the Ls1 / La1 softmax | Log-density log N(u; μ_c, Σ_c) used as the logit | The paper writes exp(P(u)); with raw densities in 100–256 dimensions this underflows. |
| Covariance regularisation | Σ ← 0.9Σ + (0.1 + 1e-4)·(tr Σ / D)·I | Office classes have about 30–90 images in 256 dimensions, so the raw covariance is singular. |
| Batch size | 64 (Digits), 32 (Office, VisDA) | Not given |
| Epochs | Source 10 / 50 / 5, adaptation 30 / 60 / 10 (Digits / Office / VisDA) | Not given. Tune on Digits M→U, then fix. |
| Adaptation "epoch" | len(target loader) × number of losses iterations, so each loss sees about one pass | Not defined in the paper |
| AdaBN before adaptation | ft's BatchNorm running statistics are re-estimated on the unlabelled target set right after ft ← fs (`adapt.adabn`) | With source BN statistics, a shared USPS digit is closer to any one-shot USPS guide than to its MNIST class mean. 74–78% of shared samples got private pseudo-labels (accuracy 28–38%). Target statistics give 14–21% and 61–69%. ft is the *target-specific* extractor, so target BN statistics fit the method. |
| fe/fd identity initialisation | Residual form u + MLP(u), last layer zero-initialised, so fe = fd = identity exactly at the start. The paper's pretraining still runs but starts at zero error (`model.proj_residual`). | With a plain MLP, 2000 pretraining steps left ~13% relative error, and the final result on Digits M→U was ALL 44.8 ± 5.2 (cluster run, 5 seeds) against 70.5 for the main runs. (An earlier version of this row also cited shared accuracy of 28–85% *before* adaptation. That is not evidence: the epoch-0 evaluation runs with an untrained, randomly initialised gt, and the main runs with residual fe/fd show the same 23–84% range there. See the note under the vanilla results.) |
| La1 on the confident set | La1 batches come from B_t instead of the whole target set (`adapt.la1_samples: confident`; `all` restores the paper). The label is still the nearest guide, computed online, and P_t is still fit on all pseudo-labelled samples. | As written (Eq. 6, Algo 4 lines 19–22), La1 trains ft on ~65%-accurate nearest-guide labels for every target sample and reinforces its own errors, moving shared samples into private clusters. Ablations on Digits M→U, 5 seeds, shared accuracy at epochs 2 / 15: full method 83.0 → 62.4; no Lr (b) 78.6 → 62.0; no La2 (c) 64.2 → 2.1; no La1 85.7 → 82.5. La1 variants, ALL / PRIV / shared at epoch 30: paper 55.1 / 58.4 / 52.8; **B_t 71.9 / 55.3 / 83.4** (flat from epoch 10; the current deterministic main run gives 70.5 / 53.5 / 82.2, see inconsistency 2); logits / 100 67.1 / 53.0 / 76.8 (peaks 73.1 at epoch 8, then decays); both 68.4 / 56.1 / 76.9. The `adapt.la1_temp` switch (default 1, the paper) stays for further tests. |
| fe/fd last layer | Linear (no activation) | The table lists an ELU/LeakyReLU last layer, which cannot be the identity on BatchNorm features (negative values). Toggle with `model.proj_final_act`. |
| ResNet backbone frozen | `src_train.backbone_lr_mult` / `adapt.backbone_lr_mult` = 0 in `configs/office31.yaml` and `visda.yaml`: the ResNet-50 keeps its ImageNet weights in both stages (its BatchNorm statistics still adapt); the FC head, classifiers and fe/fd train at 1e-4. `configs/office_a2d_bb01.yaml` keeps the 0.1× alternative. | The paper only says "ResNet-50 pretrained on ImageNet". Feature probe on Office A→D (`scripts/feature_probe.py`, one-shot nearest-guide accuracy on DSLR, mean of 5 seeds), priv|priv / shared / all: ImageNet features **72.4 / 93.5 / 71.3**. With the whole network fine-tuned (one-FC256 head, `runs/office_a2d_v0`): source backbone 33.1 / 86.1 / 54.3, full source extractor 27.0 / 82.2 / 48.3, adapted ft 46.1 / 72.9 / 52.2. Fine-tuning the whole ResNet for 50 epochs on ~900 images of 10 classes (100% source accuracy) discards what separates the unseen private classes. Source-only, two-FC256 head: frozen fs 46.3 / 90.9 / 58.0 (source acc 99.8); 0.1× backbone 71.2 / 94.5 / 73.3, fs 42.2 / 91.8 / 56.6 (source acc 100). **Full A→D run, frozen (5 seeds): ALL 70.3 ± 4.5, PRIV 48.5 ± 8.8, shared 93.4** (paper 72.2 / 72.6), against ALL 49.8 / PRIV 33.3 for `office_a2d_v0`; adapted ft probe 67.0 / 93.4 / 71.8. PRIV peaks at epoch 30 (ALL 72.2, PRIV 52.3) and drifts down to 48.5 at epoch 60; we keep the last epoch (no target labels for selection). The remaining PRIV loss is mostly private samples assigned to shared classes (26% at epoch 60; shared → private 6%). Private-guide rectification does not reduce it (ALL 70.3 / PRIV 50.8 / shared 91.0, priv_as_shared 32%); see "Under test: private-guide rectification". |
| ResNet head (fixed) | ResNet-50 → FC1024-ELU → BN → FC256-ELU → FC256-ELU → BN, as in supp Table 2A (`model.resnet_fc_out: 2`) | Our first Office A→D run (`runs/office_a2d_v0`, ALL 49.8 / PRIV 33.3) had only one FC256 layer by mistake. Saved runs without the key still load with one layer. |
| LeNet BatchNorm | On | "Similar to [OSBP]", whose LeNet uses BN |
| LeNet layer sizes | conv5, conv5, conv3 stride 2, conv3 stride 2 | Inferred from the 28→24→11→5 feature-map sizes in supp Table 2 |
| BN when computing private guides | The one-shot images go through ft in the same batch as the target samples | Avoids BatchNorm statistics from a batch of one image per class |
| Digits data | Adapt on the target train split, evaluate on the target test split | Not given |
| Office data | Transductive: adapt and evaluate on the whole target domain, excluding the labelled one-shot images from evaluation | Not given |
| Model selection | Last epoch (no target labels are used for selection) | Source-free and unsupervised |
| ALL averaging | Report both per-sample and class-averaged accuracy | Not given |
| Source warm-up | Trains fs and gs (Algo 2 line 4 writes only θ_fs, but gs must be trained too) | Algo 2 typo |

## Vanilla preset (the paper as written)

`preset=vanilla` (`configs/presets/vanilla.yaml`) switches off every change we made to the method, so the implementation can be tested exactly as the paper describes it. Runs go to `runs_vanilla/<task>/`; results with `python scripts/aggregate.py runs_vanilla/`.

| Our change (main runs) | Vanilla (paper) |
|---|---|
| La1 on the confident set B_t | La1 on all target samples (`adapt.la1_samples: all`) |
| AdaBN on ft before adaptation | ft ← fs unchanged, BatchNorm statistics included (`adapt.adabn: false`) |
| Residual fe/fd, exact identity at the start | Plain MLP pretrained to near-identity (`model.proj_residual: false`) |
| fe/fd last layer linear | Last layer with the activation from supp Table 2 (`model.proj_final_act: true`) |
| ResNet-50 backbone frozen (Office, VisDA) | Whole network trained with Adam 1e-4 (`backbone_lr_mult: 1.0` in both stages) |

Kept in vanilla, because the paper fixes them: the two-FC256 ResNet head (supp Table 2A), the 3σ negative test (`neg_mode: paper`), La1 temperature 1, and everything under "Fixed by the paper".
Kept in vanilla, because the paper is silent and the method needs *some* value: log-density logits, covariance regularisation, batch sizes, epochs and the adaptation-epoch length, LeNet BatchNorm and layer sizes, the private-guide batch, data splits, last-epoch model selection, the warm-up training gs (Algo 2 typo), and the Office class indices (inconsistency 1).

What we already know about these settings one at a time (Digits M→U unless stated): plain-MLP fe/fd gave ALL 44.8 ± 5.2; paper La1 gave 55.1 / 58.4 (ALL / PRIV); full fine-tuning on Office A→D gave 49.8 / 33.3 (with the one-FC256 head). Vanilla combines all of them.

**Vanilla results, Digits (5 seeds, ALL / PRIV):** M→U 43.7 ± 4.6 / 50.5 ± 15.5; U→M 47.2 ± 7.3 / 60.7 ± 6.5; S→M 32.8 ± 5.5 / 46.5 ± 6.0. Paper: 72.5 / 71.6, 69.4 / 68.6, 46.4 / 45.7. Our main runs: 70.5 ± 10.5 / 53.5 ± 17.4 (M→U, current deterministic run, 2026-11-18), 65.5 / 51.0, 58.8 / 44.0 (U→M and S→M from earlier runs, not yet re-run since the pipeline became deterministic). ALL is 14–28 points below the paper, while PRIV is close to or above our main runs. Shared samples are pushed into the private classes: in S→M seed 4, 44.9% of shared samples are predicted private, and shared accuracy is 32.3. This is the failure each of our changes addressed.
Per-epoch means (shared accuracy, epoch 0 → 1 → 30; nearest-guide pseudo-label accuracy at epoch 1): M→U 47.2 → 45.8 → 39.1, 40.6%; U→M 80.3 → 47.0 → 34.5, 43.9%; S→M 61.8 → 42.9 → 20.0, 37.5%. The loss happens in three steps:
1. **Before any adaptation loss (epoch 0):** shared accuracy is already low on M→U (47.2) and S→M (61.8), and 52% (M→U) and 21% (S→M) of shared samples go to gt. **This is not caused by the vanilla settings.** The epoch-0 evaluation ([adapt.py](adapt.py), `log.log(epoch=0, ...)`) runs with gt still at its random initialisation, so its logits compete with gs for every sample. The main runs (AdaBN, residual fe/fd) show the same thing on M→U: shared 45.4, shared → private 54.1% (per seed 15–76%). Epoch-0 prediction accuracy says nothing about the method. What does differ at epoch 0 is the nearest-guide pseudo-label accuracy, which does not use gt: **27.9% vanilla vs 63.3% main** on M→U. That is where the plain fe/fd and source BatchNorm statistics hurt.
2. **First epoch:** pseudo-labels are 38–44% accurate, against 58–65% in the main runs. Without AdaBN, shared target samples lie closer to the one-shot target guides than to fe(μ_s). About half of all shared samples end up predicted private.
3. **Later epochs:** La1 trains ft on these labels for every target sample, so shared samples keep moving into the private clusters. shared_as_priv ends at 57–60%. PRIV stays at 47–61 only because most samples are predicted private.

**Vanilla results, Office-31 (Table 3 setting, 5 seeds, ALL / PRIV at epoch 60; paper in brackets):** A→D 45.2 / 16.9 (72.2 / 72.6); A→W 51.3 / 26.9 (62.1 / 62.0); D→A 25.2 / 3.2 (62.6 / 61.8); D→W 56.1 / 17.8 (78.5 / 78.7); W→A 20.9 / 7.4 (62.1 / 62.4); W→D 54.3 / 19.6 (87.8 / 87.6). The main configuration on A→D: 70.3 / 48.5.
On Office the failure runs the other way from Digits: **private samples end up in shared classes** (priv_as_shared 53–94% at epoch 60). Shared accuracy stays at 91–95 on D↔W and 75 on A→D/W, and falls to 36–50 with Amazon as target.
- At epoch 0, no shared samples go to gt: shared_as_priv is 0. On Digits, by contrast, the untrained gt takes 15–76% of shared samples at epoch 0 in both vanilla and main runs (see step 1 above), so the fe/fd last-layer activation (ELU here, LeakyReLU on Digits) does not explain the difference. Why the random gt wins so rarely on Office is not yet checked (the scale of gt's initial logits relative to gs is a likely candidate).
- PRIV peaks early and then decays steadily. Peak (epoch) → epoch 60: A→D 35.6 (10) → 16.9; A→W 40.6 (10) → 26.9; D→W 42.3 (5) → 17.8; W→D 48.0 (5) → 19.6; D→A 36.9 (2) → 3.2; W→A 30.7 (5) → 7.4. Pseudo-label accuracy decays with it (e.g. D→W 72.6 → 57.5). La1 trains the whole ResNet-50 on every target sample with these labels, so the features drift towards the shared guides. The source model starts from features that already separate the private classes badly (feature probe: 27 priv|priv on A→D with full fine-tuning).
- Even the best epoch, chosen with target labels (which the setting does not allow), stays below the paper's ALL: 56.8, 60.9, 52.2, 71.0, 47.2, 72.6 for A→D, A→W, D→A, D→W, W→A, W→D. The epoch count (ours; the paper gives none) explains part of the gap, not all of it.
- `trajectory.py` reads each seed's log from its last restart. So a seed resumed after the 6 h requeue only appears from that epoch on (n = 4 in early W→D / W→A rows).

**Vanilla results, Office-31 (Table 2 setting, |Cs| = 20, |Ct| = 31, 5 seeds, ALL / PRIV at epoch 60; paper in brackets):** A→D 47.7 ± 4.3 / 16.3 ± 6.7 (73.3 / 73.1); A→W 42.5 ± 5.4 / 20.9 ± 4.1 (63.6 / 62.6); D→A 9.9 ± 7.9 / 4.4 ± 5.6 (64.1 / 64.3); D→W 53.2 ± 18.1 / 17.7 ± 10.4 (80.3 / 79.4); W→A 14.7 ± 11.8 / 1.4 ± 1.7 (63.7 / 62.8); W→D 60.0 ± 3.5 / 12.7 ± 6.1 (89.5 / 88.4). Seed 4 of A→D was restarted from scratch after its resume checkpoint was damaged by the full home quota.
The failure has the same shape as in the Table 3 setting:
- At epoch 0, gt has not been trained, so every private sample is predicted shared (priv_as_shared 100) and no shared sample is predicted private.
- PRIV rises for a few epochs, then decays while priv_as_shared climbs back to 60–99%. Peak (epoch) among epochs 0–30: A→D 34.4 (10), A→W 39.1 (10), D→A 25.6 (2), D→W 34.2 (3), W→A 21.9 (2), W→D 32.5 (3).
- With Amazon as the target, shared accuracy collapses as well (at epoch 30: D→A 40.1, W→A 28.9, from 70.3 and 55.7 at epoch 0). This brings ALL at epoch 60 down to 9.9 and 14.7, below the Table 3 runs (25.2, 20.9).
- The best epoch, chosen with target labels, is still below the paper's ALL: 54.5, 52.6, 51.2, 76.3, 44.8, 74.4 for A→D, A→W, D→A, D→W, W→A, W→D. D→W comes closest (76.3 against 80.3).
- B_min_class (the smallest number of confident samples B_t gives any class) is 0 in every seed on A→D, D→A (from epoch 15) and W→A: in each of those runs at least one class has no samples in B_t.

**Vanilla results, VisDA Syn→Real (5 seeds, ALL / PRIV at epoch 10; paper in brackets):** 6.2 ± 2.4 / 16.1 ± 6.2 (56.4 / 56.3); ALL(cls) 8.5 ± 0.5. The model collapses onto one class: ALL(cls) is close to 1/12 (8.3, every image given the same label), and at epoch 10 shared accuracy is 0.0 with shared_as_priv 99.9. Every target image is predicted as the same private class, which differs between seeds. PRIV is then that class's share of the private images. Source training itself works (src_acc 97 on synthetic).
Per-epoch means (`trajectory.py`; n = 4 up to epoch 3 because one seed resumed after a requeue):

| epoch | ALL | PRIV | shared | shared_as_priv | priv_as_shared | pl_acc_all | B_precision_priv |
|---|---|---|---|---|---|---|---|
| 0 | 23.0 | 0.0 | 37.3 | 0.0 | 100.0 | 10.4 | 8.9 |
| 1 | 16.4 | 9.6 | 20.7 | 48.4 | 43.6 | 11.4 | 9.2 |
| 3 | 10.6 | 3.7 | 14.9 | 21.4 | 81.6 | 8.2 | 7.6 |
| 5 | 10.7 | 5.7 | 13.8 | 25.0 | 73.3 | 6.8 | 7.7 |
| 8 | 6.4 | 11.0 | 3.5 | 70.6 | 29.6 | 6.0 | 7.2 |
| 10 | 6.2 | 16.1 | 0.0 | 99.9 | 0.2 | 6.5 | 7.0 |

- Before adaptation (epoch 0), the source model classifies the shared real images at 37.3% (6 classes, chance 16.7), so the pipeline works. The nearest-guide pseudo-labels, however, are only 10.4% accurate, below 12-class chance.
- The shared guides are already displaced before adaptation. After the 2000-iteration near-identity pre-training, the plain-MLP fe/fd (with ELU on the last layer) still reconstruct the source features with a relative error of 0.24 (seed 0: loss_fe 32.5, loss_fd 32.4). The residual fe/fd of the main configuration are exact identities at the start.
- The guides come from two domains: fe(μ_s) for shared classes is built from synthetic features, while the private guides ft(one-shot image) are real images. Without AdaBN, real shared images lie closer to the real one-shot private guides than to the synthetic class means. Most of them are pseudo-labelled private (step 2 of the Digits failure), and B_t is 91–93% wrong for private classes.
- La1 then trains the whole network on these labels for every target sample. Shared accuracy falls from 37.3 to 0 within 10 epochs, and the private predictions merge into a single class.
- This is the Digits failure (shared samples absorbed into private classes) taken to the end, made worse by the larger synthetic-to-real gap. We have no main-configuration VisDA run to compare with yet.

## Inconsistencies in the paper

1. **Office private classes.** Supp §5 says the private classes are those "ranked 21–31", which is 11 classes. Table 3 has |Ct| = 20, which means 10 private classes. We use indices 20–29 for Table 3 (`configs/office31.yaml`) and 20–30 for Table 2, where |Cs| = 20 and |Ct| = 31 (`configs/office20.yaml`).
2. **The 3σ rule in high dimensions.** For a Gaussian, the point μ + 3√λ·ν has squared Mahalanobis distance exactly 9, so the Algo 1 test reduces to "Mahalanobis distance > 3 from the most likely class". In D dimensions a typical in-distribution sample has squared distance about D (100 or 256). So almost every global sample, and almost every real class sample, counts as "negative": the acceptance rate is about 100%. `src_train.neg_mode=chi2` instead uses the chi-square(D) quantile as the threshold. **Result (Digits M→U):** the chi2 threshold (143 for D = 100) also accepts 100% of samples. The global Gaussian's covariance includes the spread *between* classes, so its samples lie far outside every class prototype under either rule. Both source models trained identically, so the threshold is irrelevant here: the negatives are simply samples of the global Gaussian. In the current runs (`runs/digits_m2u` and `runs/digits_m2u_chi2`, 2026-11-18) the source and adaptation checkpoints are **bit-identical tensor for tensor**, and both give ALL 70.5 / PRIV 53.5. The pipeline is deterministic for a given config and seed. An earlier main run gave 71.9 / 55.3, and this section used to attribute the gap to GPU nondeterminism. The current code shows no run-to-run variation at all, so a code or setting change between the two runs is the more likely cause; the older run's exact setup was not recorded here.
**Noise is the one-shot draw, not the run.** Across the 5 seeds of M→U main: ALL sd 10.5 (57.0–84.5), PRIV sd 17.4 (36.8–79.4), so the standard error of a 5-seed mean is about 4.7 (ALL) and 7.8 (PRIV). The private pseudo-label precision in B_t tracks this closely (seed 4: 99% → PRIV 79.4; seeds 0 and 1: 60–62% → PRIV 37–40). Each seed fixes the one-shot images, so compare variants **seed by seed** (paired differences) rather than mean against mean. Example: main minus vanilla on M→U is +8.7 to +39.3 ALL on every seed (mean +26.8).
3. **Algo 1 line 9** writes P(u_s) where it means P(u_n).
4. **Few-shot "5% labelled" on DSLR.** Taken per class and rounded (`adapt.fewshot_frac`, at least 1 image), 5% of a DSLR private class (6–25 images) is always 1 image, drawn by the same RNG as the one-shot image. So Ours* on the DSLR-target tasks (A→D, W→D) is the one-shot run, yet the paper reports different numbers (A→D 78.8 / 74.3 against 72.2 / 72.6). The paper must count the 5% another way (for example over the whole target set); it does not say how. We run Ours* only where 5% gives more than one image per class (Webcam and Amazon targets) and report the DSLR rows as identical to one-shot.

## Under test: private-guide rectification and guide placement (2026-11-20)

**Hypothesis.** The two kinds of guide are built differently. A shared guide fe(μ_c) is a mean; a private guide is one labelled image, a random point of its class cloud. For a target class with mean m_c and spread tr Σ_c, a sample's expected squared distance to its own guide is tr Σ_c + ‖g_c − m_c‖², which is about 2 tr Σ_c for a one-shot guide and tr Σ_c + ‖δ_c‖² for a shared guide (δ_c = domain-shift offset of fe(μ_c)). With small δ (frozen ImageNet features, Office) private samples drift to the shared classes: priv_as_shared 26% on A→D, shared → private 6%. With large δ (MNIST → USPS) the drift reverses, which matches the Digits failure. Not yet measured on real runs.

**Diagnostic** (`scripts/guide_diag.py`, no retraining). For each seed it reports two things.
- **Asymmetry:** ‖g_c − m_c‖² / tr Σ_c and the mean distance to the own guide, per group.
- **Nearest-guide pseudo-label accuracy** under five sets of guides:
  - **basic:** the method's guides.
  - **rect:** private guides rectified with label-free members.
  - **rect-m:** members chosen by margin.
  - **rect-all:** shared guides rectified too.
  - **oracle:** true target class means. This is an upper bound and uses labels.

`--stage start` reconstructs the features adaptation starts from (fs + AdaBN, fe = identity). `--stage final` uses each seed's adapted ft. Results go to `<run>/guide_diag_<stage>.json`.

**Method** (`cida/rectify.py`, after Liu et al., ECCV 2020, "Prototype Rectification for Few-Shot Learning"). At every refresh, each private guide g_c becomes a weighted mean of itself and its confident members:
- Members are the top n_percent of the samples whose nearest guide is c, chosen as B_t is.
- Weights are w_i = softmax(ε · cos(x_i, g_c)), and the labelled image keeps at least `anchor` of the weight.
- This is repeated `iters` times, re-assigning samples to the current guides each time.

During training the live guide is w0 · g_live + corr, so L_a2 still moves the one-shot part. The labelled images are excluded from the members. Switches live under `adapt.rectify.*`, off by default, so the main runs are unchanged. `configs/office_a2d_rect.yaml` turns it on (`adapt.tag=rect`). Logged per epoch: `rect_w0`, `rect_shift`, and `pl_acc_basic` (the pseudo-label accuracy the basic guides would give). eps 10, anchor 0.3 and iters 3 are our choices and have not been tuned.

**Toy check** (`tests/test_rectify.py`):
- **Setup:** 20-d Gaussians, mean separation 8, and the least typical private sample as the one-shot.
- **Before:** nearest-guide private accuracy is 81%.
- **After rectification:** 99.7%, and the guide is 2–3 units from the class mean, down from 6.7.

**Office A→D results** (`runs/office_a2d/adapt_1shot_abl-none_rect_seed0-4` against the main runs, which were re-run from scratch and reproduce ALL 70.3 / PRIV 48.5 exactly). The table gives means of 5 seeds at epoch 60, with paired t over the 5 seed differences:

| Metric | Main | Rect | Change (t) | Per-seed change |
| --- | --- | --- | --- | --- |
| ALL | 70.3 ± 4.5 | 70.3 ± 5.0 | +0.1 (0.1) | +1.3 −1.9 +0.3 −0.3 +0.9 |
| PRIV | 48.5 ± 8.8 | 50.8 ± 8.0 | +2.3 (1.4) | +5.5 −1.2 +4.9 −2.5 +4.9 |
| shared | 93.4 ± 3.5 | 91.0 ± 3.2 | −2.3 (−2.1) | −3.2 −2.6 −4.5 +1.9 −3.2 |
| priv_as_shared | 26.4 | 32.3 | +5.9 (1.6) | +1.8 +20.9 +1.2 +4.3 +1.2 |
| private → other private class | 25.2 | 16.9 | −8.2 (−2.7) | −7.4 −19.6 −6.1 −1.8 −6.1 |
| PRIV among predicted-private | 66.1 | 75.1 | +9.0 (3.6) | +8.7 +16.7 +8.1 +1.1 +10.2 |
| shared_as_priv | 6.2 | 4.0 | −2.2 (−1.1) | |
| shared → other shared class | 0.4 | 4.9 | +4.5 (3.5) | +5.2 +7.8 0.0 +5.8 +3.9 |

What the results show:
- **No net gain.** The PRIV gain is within seed noise, and ALL is unchanged because shared falls by as much as PRIV rises.
- **The hypothesis is not confirmed for training.** The drift it was meant to fix gets worse: priv_as_shared rises from 26.4 to 32.3. Rectification only re-centres a guide on samples it already owns. Private samples that a shared guide has captured never become its members.
- **What improves is separating the private classes from each other**, in all 5 seeds. B_t is cleaner: B_precision_priv at epoch 60 is 82.6 against 70.5 for the main runs. In the main runs it decays from 93 to 70; in the rect runs it holds at about 83 from epoch 5 on.
- **Rectified and basic guides give the same pseudo-label accuracy.** Within the rect runs, pl_acc_all equals pl_acc_basic: 71.3 vs 71.3 at epoch 60, and at most +0.7 at epoch 0. Yet rect_w0 settles at 0.49. The features probably adapt around whichever guide is live, so the effect comes through which samples enter B_t, not through different labels.
- **Shared → other shared errors are unexplained.** They rise from 0.4 to 4.9 although the shared guides are not rectified. Not yet checked with `scripts/confusion.py`.
- **Same late drift.** Rect peaks earlier (epoch 14: ALL 72.6, PRIV 54.1, against epoch 28: 72.3 / 52.3 for the main runs). It then falls to 50.8 at epoch 60, again through private → shared.

Not adopted: the main configuration keeps rectify off. eps, anchor and iters stay untuned; tuning them is not worth it while the private → shared drift is unchanged.

**Diagnostics on the Office A→D runs** (logs 56305 / 56309 / 56251 / 56253, no retraining).

*Offset sweep* (`scripts/posthoc.py`). Seed means, ALL / PRIV / shared, with q the share of DSLR images predicted private:

| Point | Main runs | Rect runs |
| --- | --- | --- |
| Paper rule (q = 0.41 / 0.37) | 70.3 / 48.5 / 93.4 | 70.3 / 50.8 / 91.0 |
| Best ALL on the sweep | 69.4 / 47.4 / 92.7 (q 0.40) | 71.1 / 57.2 / 85.8 (q 0.46) |
| q = 0.50 (balanced-class prior: 10 of 20 classes private) | 68.4 / 53.1 / 84.5 | 70.2 / 58.4 / 82.7 |
| PRIV ≈ shared | 57.3 / 57.9 / 56.6 | 63.0 / 62.3 / 63.8 |
| Paper (shared implied) | 72.2 / 72.6 / ≈ 71.8 | |

- **The boundary alone cannot reach the paper.** If ALL is per image, the paper's numbers mean about 229 of 317 DSLR images correct against our 223, so totals are close and the split differs. But moving the boundary loses accuracy: it costs shared images faster than it rescues private ones. PRIV levels off at 59 (main) and 64 (rect) even at q = 0.80, because the remaining private errors go to other private classes.
- **Rectification shifts the whole curve.** At equal shared accuracy, the rect runs get 4–5 more PRIV points (shared ≈ 85: 53 vs 57; shared ≈ 72: 56 vs 61). Their paper-rule boundary sits further toward shared (36.8% predicted private against 40.9%), which hides the gain at the default point.
- **The label-free gate and bias barely help** (main: PRIV +0.3 / +0.6, rect: +0.6 / −0.4). The bias calibrates to the share of samples whose nearest guide is private, which is itself too low (36–42% against the true 51%).

*Guide diagnostic* (`scripts/guide_diag.py`, main runs). Nearest-guide pseudo-labels on the unlabelled target, ALL / PRIV / B_t private precision:

| Guides | Start (fs + AdaBN, fe = id) | Final (adapted ft, fe) |
| --- | --- | --- |
| basic | 57.2 / 28.2 / 69.3 | 69.5 / 47.6 / 64.0 |
| rect | 57.7 / 29.8 / 72.0 | 70.1 / 48.8 / 65.0 |
| rect-all | 59.1 / 31.5 / 67.0 | 70.3 / 49.4 / 64.6 |
| oracle (true private class means) | 76.8 / 69.1 / 87.4 | 81.1 / 77.9 / 92.2 |

- **The adapted features are good enough.** With the private guides at the true class means, nearest-guide labelling reaches PRIV 77.9 / ALL 81.1, above the paper. The method's guides reach 47.6. The gap is guide placement.
- **Asymmetry, off/tr = ‖g − m‖² / tr Σ.** At the start it is 0.90 (shared) and 0.95–2.02 (private), close to the theory (about 1 for a one-shot guide). After adaptation the private guides are 3.5–11.3 class spreads off their means, against 1.1–1.6 for shared: the offset grows during training. Rectification barely moves it (2.8–11.1), because it averages over members chosen by the current guide, a third of which are wrong.
- **The seed spread is not set by the shot draw.** At the start, nearest-guide PRIV is 27.0–28.8 on every seed, and the shot's typicality does not track final PRIV (seeds 3 and 4 have the least typical shots, ratio about 2.0, and end at 56.4 and 41.1). The spread (38–58) arises during training. `--seed` sets both the one-shot images and the training RNG (`set_seed(1000 + seed)`), so seed-paired comparisons do not isolate the shot.

**K-means re-centring** (`kmeans_guides` in `cida/rectify.py`, variants in `scripts/guide_diag.py`; logs 56326 / 56346 / 56347). Lloyd's k-means over all unlabelled target samples, started at the basic guides, each cluster keeping its starting guide's label (label-free, 30 rounds at most). Variants: `km` (every guide moves), `km-priv` (shared guides fixed at fe(μ_c)), `km-anch` (the starting guide keeps weight 0.3), and `oracle-all` (every guide at its true class mean, the ceiling for `km`). Nearest-guide labels on the unlabelled target, which on Office is also the evaluation set. Seed means, ALL / PRIV / shared:

| Guides | Main, final | Rect, final | Main, start |
| --- | --- | --- | --- |
| basic | 69.5 / 47.6 / 92.6 | 70.3 / 51.3 / 90.5 | 57.2 / 28.2 / 87.8 |
| rect-all | 70.3 / 49.4 / 92.5 | 70.4 / 51.8 / 90.1 | 59.1 / 31.5 / 88.2 |
| **km** | **71.9 / 52.4 / 92.5** | **73.6 / 56.8 / 91.3** | 58.3 / 30.6 / 87.7 |
| km-priv | 69.2 / 53.7 / 85.6 | 71.9 / 59.4 / 85.2 | 57.2 / 33.4 / 82.3 |
| km-anch | 70.7 / 49.8 / 92.7 | 72.9 / 55.7 / 91.2 | 59.1 / 32.1 / 87.5 |
| oracle (private means) | 81.1 / 77.9 / 84.4 | 79.2 / 76.1 / 82.5 | 76.8 / 69.1 / 85.1 |
| oracle-all | 87.9 / 80.2 / 96.0 | 88.3 / 80.2 / 96.9 | 81.7 / 73.0 / 90.9 |

- **`km` is the first label-free change that raises both ALL and PRIV on every seed.** On the final features: main runs +2.4 ALL / +4.8 PRIV (per seed PRIV +3.7 +4.9 +8.0 +1.8 +5.5), rect runs +3.3 / +5.5 (+5.5 +7.4 +6.7 +4.3 +3.7), with shared unchanged. Rect + km labelling gives 73.6 / 56.8 / 91.3, above the paper's ALL (72.2) though PRIV stays well below (72.6). B_t private precision rises from 64.0 to 74.3 (main) and from 79.5 to 86.1 (rect).
- **Fixing the shared guides costs shared accuracy.** `km-priv` gets the most PRIV (59.4 on rect) but loses 5–7 points of shared, because private clusters take over shared samples that the fixed shared guides cannot win back.
- **It closes only a sixth of the gap.** `oracle-all` reaches 88 / 80 / 96 on the same features, against 72–74 / 52–57 for `km`. K-means stops at a poor local optimum: on seeds 3 and 4 the private guides stay 9–10 class spreads off-centre after k-means (seeds 0 and 1: 0.8–1.5). Those are the seeds whose one-shot images were least typical at the start (ratio about 2.0 against 1.0), so shot quality matters for where k-means ends, even though it did not set the starting pseudo-label accuracy.
- **It does nothing at the start** (58.3 / 30.6 against 57.2 / 28.2). Before adaptation the target classes are not yet compact enough for k-means to find them; it helps only on adapted features.

**Other label-free placements, and more labelled images** (`cida/rectify.py`, `scripts/guide_diag.py`, no retraining; logs 56360 / 56361 / 56363, rect runs, final features). New variants:
- `km-bal`: k-means with equal cluster sizes (Sinkhorn assignment, entropy weight 0.1 relative to the mean squared distance), the uniform class prior behind the sweep's q = 0.5.
- `km-over`: k-means with 3 free clusters per class, each labelled by the majority `km` label of its members, so a class can own several clusters.
- `lp`: label spreading (Zhou et al., NeurIPS 2004) from the guides over a cosine 10-NN graph, alpha = 0.99, class-balanced. It is dense N x N, so it is skipped above 20,000 samples.

`--label_frac 0.05` / `--shots k` build the private guides from more labelled images per class: the run's shot plus extra images drawn with the run's seed, all excluded from scoring. The models were still trained with one shot, so this measures the guides alone (the trained few-shot runs are `adapt.fewshot_frac=0.05`).

Seed means, ALL / PRIV / shared, with the per-seed PRIV change from basic:

| Labelling | Run's one-shot | 3 labelled images per private class |
| --- | --- | --- |
| basic | 70.3 / 51.3 / 90.5 | 75.7 / 62.5 / 87.9 |
| km | 73.6 / 56.8 / 91.3 (PRIV +3.7 to +7.4) | 75.4 / 60.3 / 89.4 (−9.1 to +7.7) |
| km-bal | 73.4 / 60.0 / 87.7 (+5.5 to +12.9) | 75.0 / 60.4 / 88.6 (−4.9 to +3.5) |
| km-over | 74.1 / 57.5 / 91.7 (+4.3 to +8.6) | 74.9 / 60.0 / 88.7 (−9.1 to +7.7) |
| **lp** | **75.8 / 62.3 / 90.1** (+1.2 to +19.0) | **77.4 / 64.5 / 89.5** (−4.2 to +8.4) |
| oracle-all | 88.3 / 80.2 / 96.9 | 89.3 / 80.6 / 97.4 |

- **5% is one-shot on DSLR.** Every private class got exactly one labelled image (about 17 images per class, rounded per class with a minimum of 1), so the 5% run is identical to the one-shot run. 5% only means more labels on larger targets such as Amazon.
- **Label propagation is the best label-free labelling.** From the one-shot guides it gives 75.8 / 62.3 / 90.1, against 70.3 / 48.5 / 93.4 for the main runs' classifier and 70.3 / 50.8 / 91.0 for the rect runs' classifier. ALL and PRIV rise on every seed (PRIV per seed 77.3 / 60.7 / 46.6 / 61.3 / 65.6). ALL is above the paper's 72.2; PRIV is still 10 points below its 72.6. `km-bal` gives the most even gain on PRIV (+5.5 to +12.9 on every seed) but costs 3 points of shared.
- **One-shot label propagation is worth about two extra labels.** Averaging 3 labelled images per class (basic, 3 shots) gives PRIV 62.5, the same as `lp` from one shot. It also confirms the asymmetry theory: the private guides' offset off/tr drops from 2.8–9.2 to 0.5–1.4.
- **Re-centring helps bad guides and hurts good ones.** With 3-shot guides, every k-means variant lowers PRIV on average (−2 to −2.5) and `lp` adds only +2.0, with 2 of 5 seeds down. K-means pulls well-placed guides toward the wrong clusters.
- **Caveat.** The variant was chosen after seeing scores that use labels, on 5 seeds of one task. lp's settings (10 neighbours, alpha 0.99) were not tuned.

Next: use `lp` as the prediction step (transductive, no retraining; the numbers above already are its scores on the evaluation set), then as the pseudo-labeller in the refresh during training. lp's B_t private precision (82.3) is close to basic's (79.5) and below `km` (86.1), so the training gain is not guaranteed. 

**lp as the prediction rule** (`scripts/posthoc.py`, logs 56471 main / 56472 rect; ALL / PRIV / shared, class-averaged in brackets). The rect `lp` row reproduces the guide_diag numbers above exactly, as it should (same features, same guides).

| Rule | Main runs | Rect runs |
| --- | --- | --- |
| paper (trained classifiers) | 70.3 / 48.5 / 93.4 (72.4 / 51.9) | 70.3 / 50.8 / 91.0 (71.9 / 53.2) |
| nearest guide | 69.5 / 47.6 / 92.6 | 70.3 / 51.3 / 90.5 |
| km | 71.9 / 52.4 / 92.5 | 73.6 / 56.8 / 91.3 |
| km-bal | 72.7 / 56.9 / 89.4 | 73.4 / 60.0 / 87.7 |
| **lp** | **75.2 / 58.0 / 93.4 (76.9 / 60.6)** | **75.8 / 62.3 / 90.1 (76.9 / 64.2)** |

- `lp` raises PRIV on every seed of both configs (main +3.1 to +17.8, rect +3.7 to +19.6), without losing shared accuracy on the main runs.
- Class-wise (main, paper → lp): speaker 21.6 → 44.8, punchers 32.9 → 54.1, printer 48.6 → 60.0, stapler 55.0 → 66.0. Not fixed: desktop_computer (44.0 → 40.0), tape_dispenser (39.0 → 41.0), projector (42.7 → 44.5), and phone stays at 0% on seeds 3 and 4 under both the paper rule and `lp`. Those errors are in the features or the one-shot guide itself, not in how labels are assigned.

**Per-image or class-averaged ALL / PRIV?** The paper reports "mean target accuracy over Ct (ALL) and C′t (PRIV)" and does not say how it averages. Supp §4 computes its precision class-averaged, so the paper's ALL / PRIV may be class-averaged too. Our runs, both readings (`final_metrics.json`, epoch 60, mean of 5 seeds):

| Run | ALL | PRIV | shared | ALL (cls) | PRIV (cls) | shared (cls) |
| --- | --- | --- | --- | --- | --- | --- |
| Main | 70.3 | 48.5 | 93.4 | **72.4** | 51.9 | 93.0 |
| Rect | 70.3 | 50.8 | 91.0 | 71.9 | 53.2 | 90.6 |
| Paper | 72.2 | 72.6 | – | | | |

Per seed, main (ALL / PRIV / shared per image; ALL / PRIV class-averaged): 76.3 / 57.7 / 96.1, 79.6 / 63.8; 69.1 / 49.1 / 90.3, 71.1 / 52.1; 64.4 / 38.0 / 92.2, 65.1 / 38.5; 72.9 / 56.4 / 90.3, 74.6 / 59.3; 68.8 / 41.1 / 98.1, 71.8 / 45.5. Rect: 77.6 / 63.2 / 92.9, 80.6 / 68.6; 67.2 / 47.9 / 87.7, 68.6 / 50.3; 64.7 / 42.9 / 87.7, 64.2 / 41.5; 72.6 / 54.0 / 92.2, 74.4 / 56.8; 69.7 / 46.0 / 94.8, 71.8 / 49.0.

Class-averaged, the main runs' ALL (72.4) matches the paper's 72.2. PRIV is about 20 points short under either reading, so the averaging question does not explain the PRIV gap.

**Class-wise accuracy** (`scripts/confusion.py`, logs 56418 main / 56419 rect, epoch 60). n is the number of evaluation images per class; "→ shared" is the share of a private class predicted as any shared class.

| Class | n | Main, per seed | Main mean | Rect mean | Where the errors go |
| --- | --- | --- | --- | --- | --- |
| 9 of 10 shared classes | 12–24 | | 93–100 | 94–100 | – |
| desktop_computer (shared) | 15 | 80 / 7 / 47 / 7 / 80 | 44.0 | 21.3 | to private (52% main, 36% rect), mainly speaker / projector |
| scissors | 17 | 94 / 94 / 100 / 94 / 94 | 95.3 | 94.1 | – |
| ring_binder | 9 | 100 / 44 / 22 / 100 / 67 | 66.7 | 64.4 | projector, calculator |
| ruler | 6 | 83 / 33 / 17 / 83 / 83 | 60.0 | 60.0 | scissors, bottle (n = 6, noisy) |
| phone | 12 | 92 / 100 / 92 / 0 / 0 | 56.7 | 56.7 | calculator (100% in seeds 3 and 4) |
| stapler | 20 | 100 / 100 / 30 / 45 / 0 | 55.0 | 73.0 | tape_dispenser |
| printer | 14 | 50 / 64 / 7 / 64 / 57 | 48.6 | 31.4 | file_cabinet; ring_binder 93% in seed 2 |
| projector | 22 | 77 / 23 / 0 / 23 / 91 | 42.7 | 58.2 | desktop_computer |
| tape_dispenser | 21 | 10 / 19 / 76 / 67 / 24 | 39.0 | 30.5 | stapler; desktop_computer in seed 4 |
| punchers | 17 | 12 / 35 / 29 / 53 / 35 | 32.9 | 44.7 | stapler, tape_dispenser |
| speaker | 25 | 20 / 8 / 12 / 64 / 4 | 21.6 | 19.2 | → shared 55% main / 66% rect, almost all desktop_computer; projector 92% in seed 1 |

- **Fixed cross-group leaks.** Four private classes lose images to one shared neighbour: speaker and projector → desktop_computer, phone → calculator, printer → file_cabinet. desktop_computer leaks back the other way. These five classes carry most of priv_as_shared.
- **Winner-takes-all among private classes.** One private guide absorbs its neighbours. In the punchers / stapler / tape_dispenser triangle, main seed 0 gives stapler 100% while punchers gets 12% and tape_dispenser 10% (both > 80% → stapler); main seed 4 reverses it, stapler 0% (95% → tape_dispenser).
- **The seed spread comes from whole classes switching on or off** (phone 0% in seeds 3–4, projector 0–91%, stapler 0–100%). The same switches mostly repeat in the main and rect runs (phone seeds 3–4, printer seed 2, ring_binder seeds 1–2), which points at the one-shot images. Seed sets both the shot and the training RNG, so this is suggestive, not proof.
- **A perfect boundary would not be enough.** PRIV|priv (argmax over the private logits only, i.e. every private image correctly routed to gt) is 60.2 for main and 65.4 for rect, both below the paper's 72.6.
- **Rectification trades classes,** which is why ALL does not move. It helps stapler (+18), projector (+16) and punchers (+12), and hurts printer (−17), tape_dispenser (−9) and desktop_computer (−23).
- **Open:** in the rect runs desktop_computer loses 43% of its images to *other shared* classes (21.3% correct, 36.0% to private), which accounts for most of the rise in shared → other shared (0.4 → 4.9). Not explained yet. The desktop_computer / speaker / projector images may be visually ambiguous on DSLR; not yet inspected.

## Under test: hand-picked one-shot images (2026-11-21)

**Idea.** Instead of a random one-shot image per private class, use the image that shows the object best. `adapt.fixed_shots` (`cida/data.py: fixed_labelled`) takes {class name: file name} and replaces the random draw; `--seed` then changes only the training RNG. This also separates the two things `--seed` normally changes together: with the shots fixed, the spread over seeds 0–4 is the training RNG alone.

**Not the paper's protocol.** The paper averages 5 random draws (§4a). Choosing the best-looking image gives a best-case number, so these runs are a separate setting and are not compared with Table 3. The picks were made by looking at every DSLR image of each class, which uses the labels of the whole class.

**Picks** (`configs/office_a2d_fixedshot.yaml`, made from `scripts/contact_sheet.py` sheets). Criteria: image quality, background, angle, visibility; within a class, the most common object model, shown from an angle that displays its distinctive parts.

| Class | Image | Reason |
| --- | --- | --- |
| phone | frame_0002 | sharp 3/4 view, handset, display and keypad visible, plain dark desk |
| printer | frame_0014 | boxy laser printer (the most common model), 3/4, output tray, plain wall |
| projector | frame_0014 | tabletop silver model (13 of 23 images), lens front and centre |
| punchers | frame_0003 | two-hole punch (11 of 18), arch handle and paper guide |
| ring_binder | frame_0007 | lever-arch binder, ring mechanism and spine visible |
| ruler | frame_0007 | wooden ruler, sharp tick marks, plain background |
| scissors | frame_0018 | blue scissors (8 of 18), open, blades and handles visible |
| speaker | frame_0011 | grille fills the view (what separates it from desktop_computer), 3/4 |
| stapler | frame_0019 | black Swingline (11 of 21), throat and anvil visible |
| tape_dispenser | frame_0020 | rectangular black model (7 of 22), roll and cutter visible |

**Pre-check without training** (`scripts/shot_check.py`, run locally on the saved main runs). Nearest-guide and lp labelling with the private guides from each seed's random shot or from the picks. Scored on the same 275 DSLR images (every set's labelled images excluded), so the numbers are not comparable with the 317-image tables above. ALL / PRIV / shared:

| Features | Rule | Random shots (seeds 0–4) | Picks |
| --- | --- | --- | --- |
| Start (fs + AdaBN, fe = id), identical for every set | nearest | 61.5 / 24.6 / 90.5 (PRIV 23.1–25.6) | **68.0 / 38.0 / 91.6** |
| | lp | 62.4 / 28.8 / 88.8 (PRIV 24.8–32.2) | **66.5 / 40.5 / 87.0** |
| Final (each seed's adapted ft, fe) | nearest | 72.7 / 47.3 / 92.6 | **77.7 / 65.8 / 87.1** |
| | lp | 76.2 / 55.2 / 92.7 | **78.4 / 64.5 / 89.4** |

Per seed on the final features, own shot → picks, PRIV: nearest 57.9 → 69.4, 48.8 → 71.1, 37.2 → 67.8, 53.7 → 66.1, 38.8 → 54.5; lp 71.1 → 76.0, 61.2 → 76.9, 39.7 → 51.2, 57.0 → 57.9, 47.1 → 60.3. ALL rises on every seed with nearest, and on 4 of 5 with lp (seed 3: 75.6 → 72.0).

- **The picks are better guides than any of the five random draws** on identical features (start PRIV 38.0 against 23–26), and still better on the final features, which were trained around each seed's own shot and so favour it.
- **They cut the private → shared leak** (nearest, final: priv_as_shared 24.8 → 9.6) and cost shared accuracy (92.6 → 87.1).
- **Not uniform across classes.** At the start, the picks help phone, scissors, speaker and tape_dispenser most; the printer pick is weak (10% at the start, against 0–60% for the random shots).
**Table 2 setting.** `configs/office20_a2d_fixedshot.yaml` and `office20_w2d_fixedshot.yaml` use the same DSLR picks plus `trash_can: frame_0004.jpg` (a single dark bin with a liner, the most common kind; 7 of 15 images are dark lined bins, 6 blue recycling bins).

*A→D results* (job 56508, `runs/office20_a2d/adapt_1shot_abl-none_fixedshot_seed0-4`, epoch 60, 487 evaluation images: 310 shared + 177 private; 20-class source model, source accuracy 98.3). No random-shot main-configuration runs exist for this setting, so the effect of the picks here is not measured.

| Seed | ALL / PRIV / shared | ALL / PRIV (cls) | priv_as_shared |
| --- | --- | --- | --- |
| 0 | 77.4 / 49.2 / 93.5 | 76.6 / 45.4 | 45.2 |
| 1 | 74.3 / 47.5 / 89.7 | 72.5 / 44.7 | 47.5 |
| 2 | 73.5 / 40.7 / 92.3 | 72.5 / 40.6 | 58.2 |
| 3 | 72.5 / 42.4 / 89.7 | 72.6 / 41.4 | 57.6 |
| 4 | 72.1 / 35.6 / 92.9 | 72.4 / 35.1 | 63.8 |
| **Mean ± sd** | **74.0 ± 2.1 / 43.1 ± 5.4 / 91.6 ± 1.8** | **73.3 / 41.4** | **54.5** |
| Paper (Table 2, random shots) | 73.3 / 73.1 | | |
| Our vanilla baseline | 47.7 / 16.3 | | |

- **ALL matches the paper, PRIV is 30 points short.** ALL is carried by the shared classes (64% of the evaluation images, 91.6% correct). Half of the private images (54.5%) are predicted as shared classes; shared → private is 0.2%.
- **The same picks do much worse than in Table 3** (PRIV 43.1 against 69.8). Table 2 has twice as many shared guides, including classes close to the private ones (mobile_phone, laptop_computer, monitor, keyboard, letter_tray), so more private images are nearer a shared guide.
- **B_t starves the private classes.** B_t holds about 90 images for 31 classes, and the smallest class has 1 throughout. Its private part is 100% correct but tiny, so Lc, La1 and La2 see few private images. PRIV rises to 45 by epoch 10, peaks at 47.0 (epoch 30), ends at 43.1; priv_as_shared falls from 99 (epoch 0, untrained gt) to about 50 by epoch 15 and stays there.
- **The seed spread is larger than in Table 3** with fixed shots (PRIV sd 5.4 against 2.7), so the training RNG matters more in this setting.
- **Where the errors go** (`scripts/confusion.py`, log 56525). Private-class accuracy, seed mean, with its main error destination: phone 90.0, scissors 91.8, speaker 62.4 (desktop_computer, mobile_phone), projector 47.3 (desktop_computer), tape_dispenser 39.0 (letter_tray, mobile_phone; 5–76 across seeds), ruler 36.7 (pen, 50–83%), trash_can 31.4 (back_pack, file_cabinet), printer 27.1 (letter_tray, file_cabinet, desktop_computer), stapler 14.0 (letter_tray, 60–75%), punchers 11.8 (letter_tray, 53–94%), ring_binder 4.4 (monitor, laptop_computer; 96% to shared). Shared classes almost never go private (0–4%); the weak shared classes lose to other shared classes (bookcase 46.7, desktop_computer 53.3, keyboard 60.0).
- **The private classifier works; the shared/private decision does not.** PRIV|priv (argmax over the private logits only) is 78.5 / 75.1 / 79.7 / 77.4 / 80.8, mean **78.3**, above the paper's 73.1, against 43.1 with the paper's rule. In Table 3 the same cap was 60–65 (random shots), so there the private classes were also confused with each other; here nearly all the loss is private → shared. letter_tray alone takes most of punchers, stapler and (in seeds 2–3) tape_dispenser; pen takes ruler; monitor / laptop_computer take ring_binder. Most of these are new shared classes, absent from the Table 3 data, and they match the nearest classes in ImageNet feature space.
*Prediction rules on the same models* (`scripts/posthoc.py`, log 56528; no retraining; ALL / PRIV / shared, seed mean; class-averaged ALL / PRIV; priv_as_shared). Only the label-free rules are candidates; the offset sweep picks q with labels and is shown for reference.

| Rule | ALL / PRIV / shared | cls ALL / PRIV | p→s | PRIV change per seed |
| --- | --- | --- | --- | --- |
| paper | 74.0 / 43.1 / 91.6 | 73.3 / 41.4 | 54.5 | – |
| gate | 76.5 / 54.0 / 89.3 | 75.6 / 52.5 | 40.0 | +9.0 to +12.4 (5/5) |
| bias | 74.5 / 44.7 / 91.5 | 73.8 / 43.1 | 52.5 | −1.1 to +4.5 (3/5) |
| km | 77.7 / 53.8 / 91.4 | 77.4 / 53.1 | 42.1 | +5.1 to +18.1 (5/5) |
| km-bal | 79.3 / 68.2 / 85.7 | 80.8 / 69.8 | 22.8 | +22.0 to +30.5 (5/5) |
| **lp** | **82.1 ± 0.8 / 68.0 ± 2.8 / 90.1** | **81.2 / 65.9** | **26.3** | **+22.0 to +30.5 (5/5)** |
| offset q = 0.36 (≈ equal class shares, 11/31) | 78.4 / 68.0 / 84.4 | | | |
| offset, best q per seed (uses labels) | 79.5 / 61.8 / 89.6 | | | |

- **`lp` is adopted as the stated post-hoc step for this setting,** reported beside the paper-rule numbers. It raises PRIV by 22–31 points and ALL by 5.7–9.7 on every seed, for −1.5 shared. Per seed, lp ALL / PRIV: 83.2 / 71.2, 82.5 / 70.6, 81.1 / 65.0, 82.1 / 67.2, 81.5 / 66.1.
- **lp beats the label-chosen boundary on every seed** (ALL 83.2 / 82.5 / 81.1 / 82.1 / 81.5 against the per-seed best offset 81.9 / 79.9 / 79.1 / 76.6 / 79.9). The equal-share offset reaches the same PRIV (68.0) but loses 5.7 more shared accuracy, so it is not added as a named rule.
- **Less selection bias than in Table 3.** `lp` and its settings (10 neighbours, alpha 0.99, class-balanced) were chosen on the Table 3 runs and named as the candidate before this log was read; nothing was tuned on these scores. Still one task, 5 seeds, scored on the images it adapts on.
- **Against the paper (73.3 / 73.1):** ALL is 8.8 above, PRIV 5.1 below (class-averaged 65.9). PRIV|priv, the cap with a perfect shared/private decision, is 78.3; lp recovers 25 of the 35 points the paper rule loses.
- **The label-free `bias` under-corrects, as expected.** It predicts 17.4% of images private (true 36.3%), because the nearest-guide shares it calibrates to leak to the shared classes too.
- **The gate works here, unlike in Table 3** (+10.9 PRIV, 5/5 seeds; Table 3 +0.3). gs's "unknown" class now flags 21–28% of target images. Still below `lp`.
- **Class-wise, paper → lp:** trash_can 31.4 → 88.6, punchers 11.8 → 62.4, ring_binder 4.4 → 53.3, printer 27.1 → 68.6, tape_dispenser 39.0 → 67.6, speaker 62.4 → 77.6, projector 47.3 → 63.6. Still weak: stapler 30.0 and ruler 20.0 (n = 6; 36.7 under the paper rule, 76.7 under km-bal). Shared classes that pay: monitor 93.6 → 85.5, desktop_computer 53.3 → 40.0, letter_tray 92.5 → 86.2, mobile_phone 92.9 → 88.4.
- **km-bal** has the highest class-averaged PRIV (69.8; it fixes ruler and stapler) but costs 5.9 shared, so ALL is 2.8 below lp.
- **To do:** `configs/office20_a2d.yaml` with random shots for the comparison; lp as the B_t pseudo-labeller in training (B_t starvation is a training-side problem lp at prediction cannot fix); posthoc on the Table 3 fixed-shot runs.

**Training results** (`configs/office_a2d_fixedshot.yaml`, job 56473, `runs/office_a2d/adapt_1shot_abl-none_fixedshot_seed0-4`, epoch 60, paper prediction rule, scored on the usual 317 images):

| Seed | Main (random shot): ALL / PRIV / shared | Picks: ALL / PRIV / shared | Picks, class-averaged ALL / PRIV | PRIV change |
| --- | --- | --- | --- | --- |
| 0 | 76.3 / 57.7 / 96.1 | 82.0 / 69.9 / 94.8 | 83.6 / 72.8 | +12.3 |
| 1 | 69.1 / 49.1 / 90.3 | 80.4 / 67.5 / 94.2 | 81.7 / 69.6 | +18.4 |
| 2 | 64.4 / 38.0 / 92.2 | 84.9 / 74.2 / 96.1 | 86.1 / 76.3 | +36.2 |
| 3 | 72.9 / 56.4 / 90.3 | 82.0 / 69.3 / 95.5 | 83.5 / 72.1 | +12.9 |
| 4 | 68.8 / 41.1 / 98.1 | 80.1 / 68.1 / 92.9 | 81.7 / 70.8 | +27.0 |
| **Mean ± sd** | 70.3 ± 4.5 / 48.5 ± 8.8 / 93.4 ± 3.5 | **81.9 ± 1.9 / 69.8 ± 2.7 / 94.7 ± 1.2** | **83.3 / 72.3** | **+21.3** |

- **Large gain on every seed:** ALL +5.7 to +20.5, PRIV +12.3 to +36.2, shared unchanged on average (+1.3). Unlike the no-training pre-check, shared accuracy does not drop.
- **The private → shared leak shrinks:** priv_as_shared 15.6 against 26.4; shared_as_priv 1.0 against 6.2.
- **Against the paper (different protocol, so not a like-for-like comparison):** ALL 81.9 is above its 72.2; class-averaged PRIV 72.3 matches its 72.6, per-image PRIV 69.8 is just below.
- **No late drift.** Seed-mean PRIV at epochs 5 / 10 / 20 / 30 / 60: 56.1 / 67.9 / 71.4 / 71.8 / 69.8; the peak (epoch 26: ALL 83.8, PRIV 72.8) is within 2 points of the last epoch. B_t private precision is 100% from epoch 5 on (main runs: 93 → 70). With clean B_t, La1 / Lc reinforce correct labels instead of errors.
- **The one-shot image drives most of the seed spread.** With the shots fixed, only the training RNG varies, and PRIV sd is 2.7 (67.5–74.2) against 8.8 (38.0–57.7) with a random shot per seed. This is measured for one good set of shots; whether a poor fixed set is equally stable is not tested.
- **Not yet run:** posthoc / confusion on these runs (class-wise accuracy; whether `lp` adds anything on top).
- **Reading.** One labelled image per class is enough for the method to reach the paper's private-class accuracy if that image is typical of its class. Which images get drawn is therefore the largest factor we have found in the gap between our random-shot runs and the paper. How the paper's 5 random draws reach 72.6 is still unexplained; we have not tested whether their draws, split or evaluation differ from ours.

## Under test: lp pseudo-labels in training (2026-11-21)

**Why.** In the Table 2 hand-picked runs, B_t holds about 90 images for 31 classes (smallest class 1), and 54.5% of private images are pseudo-labelled shared, so they never enter the private part of B_t and Lc / La1 / La2 rarely train on them. Label propagation as the *prediction* step on the same models moved PRIV from 43.1 to 68.0 and the private → shared leak from 54.5% to 26.3% (log 56528). The question is whether the same labels help *during* training.

**Switch** (`adapt.pseudo_label`, default `nearest` = the paper; `lp_knn` 10, `lp_alpha` 0.99, the settings used post hoc, untuned). With `lp`, at every refresh:
- every target image gets its label from `cida.rectify.label_propagation` (the guides as labelled nodes, cosine 10-NN graph, class-balanced);
- B_t is the 20% of each class's members with the highest lp confidence (instead of the 20% nearest the guide);
- P_t is fit on the lp labels, and La1 uses the image's lp label from the last refresh instead of the online nearest guide (otherwise La1 would pull images that lp moved to a private class back to their nearest, shared, guide);
- the guides themselves, Lr1 / Lr2 / Lc / La2 and the prediction rule are unchanged.
The log adds `pl_acc_nearest` (what nearest-guide labelling would have scored) next to `pl_acc_all`. Dense N × N graph, so it refuses targets above 20,000 images (VisDA would keep `nearest`).

**Expected risk.** On the Table 3 random-shot runs, lp's B_t private precision was 82.3 against 79.5 for nearest guide (guide_diag), so wrong labels may still reinforce themselves; the gain may be large on Table 2 and small on Table 3.

**Test** (random shots, seed-paired; adopt only if PRIV rises on most seeds in both settings without a large shared drop): `configs/office_a2d_lpref.yaml` against the main runs (70.3 / 48.5 / 93.4), and `configs/office20_a2d_lpref.yaml` against `configs/office20_a2d.yaml`. Not yet run. The first test job runs only the Table 2 pair; the Table 3 one is deferred.

**Backbone freezing, isolated** (`configs/office20_a2d_ft.yaml`, Table 2 setting, in the same job; results below: freezing is worth +8.4 ALL). The 49.8 → 70.3 jump on Table 3 A→D (row "ResNet backbone frozen" above) came from freezing *and* the head fix together. This config keeps the fixed head and every other change of the main configuration, but trains the whole ResNet-50 at Adam 1e-4 in both stages (its own 20-class source model), so the seed-paired difference with `configs/office20_a2d.yaml` (frozen, same job, seeds 0–2) is the effect of freezing alone. Not yet run.

**Results** (job 56572, `logs/cida_56572.out`; Table 2 A→D, random one-shot draws, seeds 0–2, epoch 60, 487 evaluation images, paper prediction rule; ALL / PRIV / shared):

| Seed | `office20_a2d` (frozen, nearest) | `office20_a2d_lpref` (frozen, lp) | `office20_a2d_ft` (not frozen, nearest) |
| --- | --- | --- | --- |
| 0 | 69.4 / 32.8 / 90.3 | 75.2 / 51.4 / 88.7 | 56.7 / 26.6 / 73.9 |
| 1 | 64.7 / 34.5 / 81.9 | 72.9 / 52.5 / 84.5 | 56.7 / 24.9 / 74.8 |
| 2 | 64.1 / 17.5 / 90.6 | 80.1 / 66.7 / 87.7 | 59.8 / 28.8 / 77.4 |
| **Mean** | **66.1 / 28.2 / 87.6** | **76.0 / 56.9 / 87.0** | **57.7 / 26.7 / 75.4** |
| Class-averaged ALL / PRIV | 66.2 / 30.5 | 76.3 / 58.7 | 60.1 / 26.3 |
| priv_as_shared / shared_as_priv | 68.7 / 1.3 | 38.6 / 5.3 | 58.9 / 5.4 |
| Paper (Table 2) | 73.3 / 73.1 | | |

*lp pseudo-labels (lpref − frozen nearest, per seed):* ALL +5.8 / +8.2 / +16.0, PRIV +18.6 / +18.0 / +49.2, shared −1.6 / +2.6 / −2.9.
- **PRIV rises on 3 of 3 seeds, shared unchanged on average (−0.6).** This passes the adoption rule for Table 2. Seed 2 (a bad draw: PRIV 17.5 with nearest) gains the most, so lp also reduces the seed spread of ALL's lower end; PRIV sd stays large (8.5 against 9.3) with 3 seeds.
- **The training gain is in the features, not only the labels.** At epoch 60 nearest-guide labelling on the lpref features scores 76.0 (`pl_acc_nearest`) against 67.9 on the frozen-nearest features; lp labels themselves reach 80.2. B_t private precision is 84.6 (85.0 with nearest), and B_t still has about 86 images with a smallest class of 1, so lp fixes *which* images enter B_t, not its size.
- **No late drift:** seed-mean PRIV 45.2 / 52.5 / 56.5 / 56.1 / 56.9 at epochs 5 / 10 / 20 / 30 / 60, against nearest's peak of 35.6 at epoch 20 falling to 28.2.
- **Against the paper:** ALL 76.0 is above 73.3; PRIV 56.9 (class-averaged 58.7) is still 16 points short.
- **Against the hand-picked shots** (frozen, nearest, 5 seeds: 74.0 / 43.1 / 91.6): lp with random shots is higher on ALL and PRIV, without picking images by hand.
- **Still to check:** the Table 3 pair (`configs/office_a2d_lpref.yaml` against main seeds 0–2) before lp becomes the default; whether post-hoc `lp` adds anything on top of the lpref models (`scripts/posthoc.py`).

*Backbone not frozen (ft − frozen, per seed):* ALL −12.7 / −8.0 / −4.3, PRIV −6.2 / −9.6 / +11.3, shared −16.4 / −7.1 / −13.2.
- **Freezing is worth +8.4 ALL here, mostly through shared accuracy** (+12.2 on average, all 3 seeds); PRIV is mixed (2 of 3 seeds better frozen, mean +1.5). This isolates freezing from the head fix: the head is identical in both runs.
- The fine-tuned source model reaches 99.9% source accuracy (frozen: 98.3), and its target pseudo-labels start lower (epoch 0: 52.7 against 64.9) and stay lower (59.1 against 67.9 at epoch 60), with classes missing from B_t (smallest class 0 from epoch 1). Fine-tuning on the 20 Amazon classes fits the source and damages the target features, as the Table 3 feature probe showed.
- Cost: about 38 min per seed and 10 min for the source model, against 33–34 min frozen.
- **Conclusion:** keep the backbone frozen.

### Table 2 with lp pseudo-labels: W→D, A→W, D→W (2026-11-22)

Jobs 56614 (W→D) and 56615 (A→W, D→W), `logs/cida_56614.out` and `logs/cida_56615.out`; `configs/office20_<task>_lpref.yaml` (main configuration + `adapt.pseudo_label: lp`), random one-shot draws, seeds 0–4, epoch 60, paper prediction rule. Each task trained its own 20-class source model (source accuracy W 100.0, A 98.3, D 100.0). A→D keeps its 3 seeds from job 56572; D→A and W→A are not run (about 17 h each).

Per seed, ALL / PRIV / shared:

| Seed | A→D (3 seeds) | A→W | D→W | W→D |
| --- | --- | --- | --- | --- |
| 0 | 75.2 / 51.4 / 88.7 | 71.8 / 45.3 / 84.7 | 75.4 / 59.0 / 83.3 | 81.5 / 58.2 / 94.8 |
| 1 | 72.9 / 52.5 / 84.5 | 75.5 / 59.4 / 83.3 | 73.7 / 45.7 / 87.3 | 85.0 / 71.2 / 92.9 |
| 2 | 80.1 / 66.7 / 87.7 | 75.9 / 52.0 / 87.5 | 79.7 / 51.2 / 93.6 | 84.0 / 65.5 / 94.5 |
| 3 | | 68.4 / 43.8 / 80.3 | 77.0 / 54.3 / 88.1 | 81.3 / 57.6 / 94.8 |
| 4 | | 77.7 / 59.0 / 86.7 | 80.7 / 70.7 / 85.6 | 75.2 / 63.3 / 81.9 |
| **Mean** | **76.0 / 56.9 / 87.0** | **73.9 / 51.9 / 84.5** | **77.3 / 56.2 / 87.6** | **81.4 / 63.2 / 91.8** |
| sd ALL / PRIV | 3.7 / 8.5 | 3.7 / 7.3 | 2.9 / 9.4 | 3.8 / 5.6 |
| Class-averaged ALL / PRIV | 76.3 / 58.7 | 72.5 / 52.4 | 76.5 / 57.6 | 81.2 / 61.8 |
| Paper (Table 2) ALL / PRIV | 73.3 / 73.1 | 63.6 / 62.6 | 80.3 / 79.4 | 89.5 / 88.4 |
| Vanilla (5 seeds) ALL / PRIV | 47.7 / 16.3 | 42.5 / 20.9 | 53.2 / 17.7 | 60.0 / 12.7 |

- **ALL against the paper:** above it on A→D (+2.7) and A→W (+10.3), below on D→W (−3.0) and W→D (−8.1). The paper's ALL and PRIV are almost equal on every task; ours have PRIV 29–33 points under shared.
- **PRIV is the gap everywhere:** 52–63 against the paper's 63–88; short by 10.7 (A→W) to 25.2 (W→D). The leak is private → shared (priv_as_shared: A→W 36.4, D→W 30.5, W→D 21.5 on average); shared → private stays small (3.5 / 5.6 / 4.1); the worst seed is W→D seed 4 (13.5, B_t smallest class 0, B_t precision 73.9).
- **Against vanilla:** +21 to +31 ALL and +31 to +51 PRIV on every task.
- **Late drift remains on these tasks.** Seed-mean ALL / PRIV at epoch 20 against epoch 60: A→W 75.7 / 54.5 → 73.9 / 51.9; D→W 81.0 / 60.1 → 77.3 / 56.2 (ALL peaks at 82.5 at epoch 5); W→D 83.1 / 66.9 → 81.4 / 63.2. PRIV peaks between epochs 9 and 27 on 12 of 15 seeds. Stopping earlier would gain 2–4 points, but the epoch would be chosen with target labels, so we keep the last epoch.
- Time: 26–34 min per adaptation seed.

### Under test: hand-picked images with lp, 20 epochs (2026-11-22)

`configs/office20_a2d_fixedshot_lp20.yaml` and `configs/office20_w2d_fixedshot_lp20.yaml`: the main configuration with the hand-picked DSLR one-shot images (`office20_*_fixedshot.yaml`), `adapt.pseudo_label: lp`, and `adapt.epochs: 20`. Seeds 0–4. Not the paper's protocol on two counts (shots picked by eye; the epoch count read off test curves), so it is reported apart from Table 2.

- **Why:** the two best levers found so far, together. Hand-picked shots gave +10 PRIV over random shots with nearest guide on A→D (epoch 20: 46.1 against 35.6); lp gave +21 (56.5). All runs plateau around epochs 15–20 and drift down afterwards (Lr2 rising, La2 still falling).
- **Tasks:** A→D, and W→D as the task furthest below the paper (lp, random shots, epoch 20: 83.1 / 66.9 against 89.5 / 88.4). A→W has the lowest absolute accuracy but has no hand-picked Webcam images.
- **20 epochs equals epoch 20 of a 60-epoch run:** the adaptation has no learning-rate schedule, so the only difference from reading epoch 20 off a 60-epoch log is that training stops there.
- **Compare with:** lp + random shots at epoch 20 (A→D 76.7 / 56.5 / 88.2, 3 seeds; W→D 83.1 / 66.9 / 92.3, 5 seeds) and nearest + hand-picked at epoch 20 (A→D 75.9 / 46.1 / 92.8).

**Results (job 56686, `logs/cida_56686.out`, 2026-11-23).** ALL / PRIV / shared per seed, last epoch (20):

| Seed | A→D | W→D |
| --- | --- | --- |
| 0 | 85.6 / 74.6 / 91.9 | 82.3 / 62.1 / 93.9 |
| 1 | 81.5 / 70.1 / 88.1 | 85.6 / 70.1 / 94.5 |
| 2 | 81.7 / 72.3 / 87.1 | 87.7 / 77.4 / 93.5 |
| 3 | 82.8 / 70.1 / 90.0 | 86.2 / 71.2 / 94.8 |
| 4 | 82.1 / 68.9 / 89.7 | 85.2 / 64.4 / 97.1 |
| **Mean** | **82.8 / 71.2 / 89.4** | **85.4 / 69.0 / 94.8** |
| sd ALL / PRIV / shared | 1.7 / 2.3 / 1.9 | 2.0 / 6.0 / 1.4 |
| Class-averaged ALL / PRIV | 82.6 / 69.5 | 85.6 / 67.3 |
| priv_as_shared / shared_as_priv | 24.4 / 4.3 | 13.3 / 2.7 |
| Paper (Table 2) ALL / PRIV | 73.3 / 73.1 | 89.5 / 88.4 |

- **A→D:** PRIV is 1.9 under the paper and ALL 9.5 over it. Against lp + random shots at epoch 20: +6.1 ALL, +14.7 PRIV, +1.2 shared. Against nearest + hand-picked at epoch 20: +6.9 ALL, +25.1 PRIV, −3.4 shared. Against the earlier post-hoc lp on the 60-epoch hand-picked models (82.1 / 68.0 / 90.1): +0.7 / +3.2 / −0.7. The private → shared leak falls from 54.5% (nearest, hand-picked, epoch 60) to 24.4%.
- **W→D:** still 4.1 ALL and 19.4 PRIV under the paper. Against lp + random shots at epoch 20: +2.3 ALL, +2.1 PRIV, +2.5 shared, so the hand-picked shots help much less here than on A→D. PRIV varies more across seeds (62.1–77.4, sd 6.0) even with the shots fixed; only the training RNG differs between seeds.
- **Drift within 20 epochs:** ALL peaks at epochs 14–17 on A→D (0.6–1.5 above the last epoch, mean 0.9) and at epochs 11–14 on W→D (1.7–3.5 above, mean 2.4). The last epoch is kept.
- Time: about 11.7 min per adaptation seed on the A40, about 2 h for the job.
- Still not the paper's protocol (hand-picked shots; the epoch count read off test curves): report apart from Table 2.

### Under test: W→D source model with the ResNet unfrozen, 100 epochs (2026-11-23)

`configs/office20_w2d_srcft100.yaml`: `office20_w2d_fixedshot_lp20.yaml` with `src_train: {epochs: 100, backbone_lr_mult: 1.0}`, its own folder `runs/office20_w2d_srcft100/` and tag `fixedshot_lp20_srcft100`. Adaptation is unchanged: `adapt.backbone_lr_mult: 0.0`, so ft's ResNet stays fixed at the *fine-tuned* source weights (BatchNorm statistics still follow DSLR via AdaBN). No earlier run unfroze the source stage alone: `office20_a2d_ft` (job 56572) unfroze both stages.

- **Why:** the Table 3 feature probe shows source training costs most of the private separability (one-shot priv|priv 72.4 with ImageNet features → 46.3 with the frozen fs), so the source model is a candidate cause of the PRIV gap. On W→D the frozen 50-epoch source model has not finished its objectives: Ls2 ends at 2.17 (Amazon: 0.61) with 38.9% of negatives detected (Amazon: 97.6%), because Webcam's 528 images give only 16 source steps per epoch. 100 epochs doubles the steps so that Ls1 and Ls2 can settle.
- **Two changes against job 56686 W→D, not one:** unfrozen backbone and 100 instead of 50 source epochs. A difference cannot be attributed to either alone.
- **Watch:** `loss_ls1` (on A→D, unfrozen source training never settled: median 38 in epochs 41–51 against 5.8 frozen), `loss_ls2` and `neg_detected`; then PRIV and priv_as_shared against job 56686 seed by seed.
- **Compare with:** job 56686 W→D (85.4 / 69.0 / 94.8; seed 0: 82.3 / 62.1 / 93.9).
- **Plan:** seed 0 first, then seeds 1–4 reusing the source model (REPRODUCE.md).

**Results, seed 0** (job 56726, `logs/cida_56726.out`; ALL / PRIV / shared at epoch 20, against job 56686 W→D seed 0):

| | Frozen source, 50 epochs (job 56686) | Unfrozen source, 100 epochs (job 56726) |
| --- | --- | --- |
| ALL / PRIV / shared | 82.3 / 62.1 / 93.9 | **84.6 / 59.3 / 99.0** |
| Class-averaged ALL / PRIV | 82.7 / 58.9 | 84.6 / 58.5 |
| Private → shared / private → wrong private / shared → private | 17.5 / 20.3 / 3.2 | 15.3 / 25.4 / 0.3 |
| Pseudo-label accuracy, epoch 0 → 20 | 82.3 → 85.9 | 67.1 → 86.3 |
| ALL / PRIV at epoch 10 | 85.2 / 66.7 | 79.1 / 42.9 |
| Source: Ls2, negatives detected (last epoch) | 2.17, 38.9% | 0.70, 95.9% |
| Source: Ls1 median, last ~25 epochs | 2.9 (steady) | 9.5 (12 of 26 epochs at 0, spikes up to 827) |

- **Ls2 settled, Ls1 did not.** 100 unfrozen epochs finished the "unknown" class (96% of negatives detected, against 39%), but Ls1 kept jumping between 0 and several hundred to the end, as on unfrozen A→D. Source accuracy was ~100% from epoch 4.
- **Shared improved, PRIV did not:** shared +5.1 (99.0, nearly every shared image right), PRIV −2.8, ALL +2.3. Class-averaged PRIV is unchanged (58.5 against 58.9). One seed, and W→D's PRIV seed spread is about 6, so the PRIV change is within noise.
- **The private loss moved, it did not shrink:** fewer private images go to shared classes (15.3 against 17.5), more to the wrong private class (25.4 against 20.3).
- **Adaptation starts from worse target features and is still climbing at epoch 20.** Pseudo-labels start at 67.1 (frozen: 82.3); PRIV is 42.9 at epoch 10 (frozen: 66.7) and still rising at epoch 20 (55.9 at 19 → 59.3), while the frozen run peaked at epoch 12 (69.5). The 20-epoch cut, chosen from the frozen runs' curves, may stop this run early.
- **Reading:** a better-trained source model makes the shared side near-perfect but does not separate the private classes better; this fits the feature probe (fine-tuning on the source classes costs target separability). Two changes at once (unfrozen, 100 epochs), so not attributable to either alone.
- **Next options:** the same seed with 60 adaptation epochs (`adapt.epochs=60 adapt.tag=fixedshot_lp60_srcft100`, `STAGES=adapt`, reusing this source model), to see whether PRIV keeps rising; seeds 1–4 before any conclusion; a frozen 100-epoch source run to separate source length from unfreezing.

## Investigated, not adopted

**Prediction-step changes for PRIV** (`scripts/posthoc.py`, re-evaluating the 15 saved Digits models; no target labels used). Mean change from the paper's rule (ALL / PRIV / shared):

| Variant | M→U | U→M | S→M |
| --- | --- | --- | --- |
| gate: gs "unknown" hands the sample to gt | −0.1 / +0.3 / −0.5 | −1.7 / +0.4 / −3.6 | −0.3 / +0.1 / −0.6 |
| bias: offset on gt logits, matched to the nearest-guide private share | −3.3 / +5.5 / −9.4 | −0.7 / +3.0 / −4.2 | −0.9 / +5.1 / −6.7 |

- **The gate barely fires on the samples that matter.** gs's unknown class was trained on global-Gaussian negatives far from every class (finding 2 above), so it does not recognise the new digits.
- **The bias raises PRIV on all 15 seeds but costs about as much in shared accuracy.** So ALL never improves. The offset only moves the boundary between overlapping shared and private features. The calibration target is also high: 46–63% private, against 41% (USPS test) and 49% (MNIST test) in truth.
- **Conclusion:** the PRIV gap comes from the features and pseudo-labels, not from the prediction step. The paper's prediction rule stays.

## Not implemented yet
- **Ours-d.** This variant uses real source features instead of proxy samples, so it needs source data during adaptation.
- **The two-step A→D→W upgrade** (supp §8).
- **Prior-art baselines.** Quote their numbers from Table 3.
