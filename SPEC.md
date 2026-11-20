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
