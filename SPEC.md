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
| fe/fd identity initialisation | Residual form u + MLP(u), last layer zero-initialised, so fe = fd = identity exactly at the start. The paper's pretraining still runs but starts at zero error (`model.proj_residual`). | With a plain MLP, 2000 pretraining steps left ~13% relative error. On the 10-epoch Digits source model this dropped shared accuracy to 28–85% **before** adaptation (cluster run, 5 seeds: final ALL 44.8 ± 5.2). |
| La1 on the confident set | La1 batches come from B_t instead of the whole target set (`adapt.la1_samples: confident`; `all` restores the paper). The label is still the nearest guide, computed online, and P_t is still fit on all pseudo-labelled samples. | As written (Eq. 6, Algo 4 lines 19–22), La1 trains ft on ~65%-accurate nearest-guide labels for every target sample and reinforces its own errors, moving shared samples into private clusters. Ablations on Digits M→U, 5 seeds, shared accuracy at epochs 2 / 15: full method 83.0 → 62.4; no Lr (b) 78.6 → 62.0; no La2 (c) 64.2 → 2.1; no La1 85.7 → 82.5. La1 variants, ALL / PRIV / shared at epoch 30: paper 55.1 / 58.4 / 52.8; **B_t 71.9 / 55.3 / 83.4** (flat from epoch 10); logits / 100 67.1 / 53.0 / 76.8 (peaks 73.1 at epoch 8, then decays); both 68.4 / 56.1 / 76.9. The `adapt.la1_temp` switch (default 1, the paper) stays for further tests. |
| fe/fd last layer | Linear (no activation) | The table lists an ELU/LeakyReLU last layer, which cannot be the identity on BatchNorm features (negative values). Toggle with `model.proj_final_act`. |
| ResNet backbone frozen | `src_train.backbone_lr_mult` / `adapt.backbone_lr_mult` = 0 in `configs/office31.yaml` and `visda.yaml`: the ResNet-50 keeps its ImageNet weights in both stages (its BatchNorm statistics still adapt); the FC head, classifiers and fe/fd train at 1e-4. `configs/office_a2d_bb01.yaml` keeps the 0.1× alternative. | The paper only says "ResNet-50 pretrained on ImageNet". Feature probe on Office A→D (`scripts/feature_probe.py`, one-shot nearest-guide accuracy on DSLR, mean of 5 seeds), priv|priv / shared / all: ImageNet features **72.4 / 93.5 / 71.3**. With the whole network fine-tuned (one-FC256 head, `runs/office_a2d_v0`): source backbone 33.1 / 86.1 / 54.3, full source extractor 27.0 / 82.2 / 48.3, adapted ft 46.1 / 72.9 / 52.2. Fine-tuning the whole ResNet for 50 epochs on ~900 images of 10 classes (100% source accuracy) discards what separates the unseen private classes. Source-only, two-FC256 head: frozen fs 46.3 / 90.9 / 58.0 (source acc 99.8); 0.1× backbone 71.2 / 94.5 / 73.3, fs 42.2 / 91.8 / 56.6 (source acc 100). **Full A→D run, frozen (5 seeds): ALL 70.3 ± 4.5, PRIV 48.5 ± 8.8, shared 93.4** (paper 72.2 / 72.6), against ALL 49.8 / PRIV 33.3 for `office_a2d_v0`; adapted ft probe 67.0 / 93.4 / 71.8. PRIV peaks at epoch 30 (ALL 72.2, PRIV 52.3) and drifts down to 48.5 at epoch 60; we keep the last epoch (no target labels for selection). The remaining PRIV loss is mostly private samples assigned to shared classes (26% at epoch 60; shared → private 6%). |
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

**Vanilla results, Digits (5 seeds, ALL / PRIV):** M→U 43.7 ± 4.6 / 50.5 ± 15.5; U→M 47.2 ± 7.3 / 60.7 ± 6.5; S→M 32.8 ± 5.5 / 46.5 ± 6.0. Paper: 72.5 / 71.6, 69.4 / 68.6, 46.4 / 45.7. Our main runs: 71.9 / 55.3, 65.5 / 51.0, 58.8 / 44.0. ALL is 14–28 points below the paper, while PRIV is close to or above our main runs. Shared samples are pushed into the private classes: in S→M seed 4, 44.9% of shared samples are predicted private, and shared accuracy is 32.3. This is the failure each of our changes addressed.
Per-epoch means (shared accuracy, epoch 0 → 1 → 30; nearest-guide pseudo-label accuracy at epoch 1): M→U 47.2 → 45.8 → 39.1, 40.6%; U→M 80.3 → 47.0 → 34.5, 43.9%; S→M 61.8 → 42.9 → 20.0, 37.5%. The loss happens in three steps:
1. **Before any adaptation loss (epoch 0):** shared accuracy is already low on M→U (47.2) and S→M (61.8). At this point the only changes from the source model are the plain-MLP fe/fd with an activation on the last layer, and the source BatchNorm statistics. 52% (M→U) and 21% (S→M) of shared samples go to the still-untrained private classifier gt.
2. **First epoch:** pseudo-labels are 38–44% accurate, against 58–65% in the main runs. Without AdaBN, shared target samples lie closer to the one-shot target guides than to fe(μ_s). About half of all shared samples end up predicted private.
3. **Later epochs:** La1 trains ft on these labels for every target sample, so shared samples keep moving into the private clusters. shared_as_priv ends at 57–60%. PRIV stays at 47–61 only because most samples are predicted private.

**Vanilla results, Office-31 (Table 3 setting, 5 seeds, ALL / PRIV at epoch 60; paper in brackets):** A→D 45.2 / 16.9 (72.2 / 72.6); A→W 51.3 / 26.9 (62.1 / 62.0); D→A 25.2 / 3.2 (62.6 / 61.8); D→W 56.1 / 17.8 (78.5 / 78.7); W→A 20.9 / 7.4 (62.1 / 62.4); W→D 54.3 / 19.6 (87.8 / 87.6). The main configuration on A→D: 70.3 / 48.5.
On Office the failure runs the other way from Digits: **private samples end up in shared classes** (priv_as_shared 53–94% at epoch 60). Shared accuracy stays at 91–95 on D↔W and 75 on A→D/W, and falls to 36–50 with Amazon as target.
- At epoch 0, no shared samples go to gt: shared_as_priv is 0. With ELU, the activation on the fe/fd last layer only clips values below −1, whereas Digits' LeakyReLU scales every negative value by 0.01. The epoch-0 loss on Digits is consistent with that.
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

**Vanilla results, VisDA Real→Syn (5 seeds, ALL / PRIV at epoch 10; paper in brackets):** 7.7 ± 1.4 / 13.9 ± 7.9 (55.8 / 55.7); ALL(cls) 8.4 ± 0.2. As in Syn→Real, every seed ends with a single predicted class. At epoch 10, shared_as_priv is 80.0 and priv_as_shared 20.0: four seeds put everything into one private class, and one seed puts everything into one shared class.
Each Real→Syn seed takes about 6 h, so every seed was resumed after a requeue. `trajectory.py` therefore has only 1–2 seeds for epochs 0–8 (n in the table):

| epoch | n | ALL | PRIV | shared | shared_as_priv | priv_as_shared | pl_acc_all | B_precision_priv |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 29.5 | 0.0 | 56.3 | 0.0 | 100.0 | 34.2 | 38.4 |
| 1 | 1 | 25.6 | 15.4 | 34.9 | 19.0 | 28.8 | 31.5 | 39.0 |
| 2 | 1 | 27.2 | 18.5 | 35.2 | 18.1 | 22.7 | 30.2 | 40.1 |
| 3 | 1 | 15.2 | 0.0 | 29.1 | 0.0 | 100.0 | 27.5 | 33.1 |
| 5 | 1 | 10.3 | 0.0 | 19.7 | 0.0 | 100.0 | 19.4 | 14.2 |
| 8 | 2 | 9.4 | 8.4 | 10.3 | 50.0 | 42.2 | 12.7 | 11.4 |
| 10 | 5 | 7.7 | 13.9 | 2.1 | 80.0 | 20.0 | 9.5 | 18.4 |

- The start is better than Syn→Real: the source model trained on real images gets 56.3 on the shared synthetic classes, and the pseudo-labels are 34.2% accurate (Syn→Real: 37.3 and 10.4).
- It still collapses. PRIV rises for two epochs and then drops to 0 by epoch 3, when the seed predicts every image as shared. Pseudo-label accuracy falls from 34 to 10, and by epoch 10 each seed predicts a single class.
- The paper reports 55.8 / 55.7 with the same architecture, so at least one of the choices the paper leaves open must differ from the authors' code. Our main suspects are the fully fine-tuned ResNet-50 (on Office A→D, freezing it took ALL from 49.8 to 70.3) and the fe/fd warm-up, which ends 24% away from identity (Syn→Real seed 0).

**All vanilla runs are complete.** ALL / PRIV against the paper: Digits 33–47 / 47–61 (paper 46–73 / 46–72); Office-31 Table 3 21–56 / 3–27 (paper 62–88 / 62–88); Office-31 Table 2 10–60 / 1–21 (paper 64–90 / 63–88); VisDA 6–8 / 14–16 (paper 56 / 56).

## Main configuration on Office-31

All four changes on, with the ResNet-50 frozen (`configs/office31.yaml`, `configs/office20.yaml`; runs in `runs/<task>/`). 5 seeds, ALL / PRIV at epoch 60. Best epoch = highest mean ALL over seeds, chosen with target labels (`scripts/epochs.py runs/`), so an upper bound only.

| Setting | Task | Main config | Best epoch (ep) | Vanilla | Paper |
|---|---|---|---|---|---|
| Table 3 (10/20) | A→D | **70.3 ± 4.5 / 48.5 ± 8.8** | 72.3 / 52.3 (28) | 45.2 / 16.9 | 72.2 / 72.6 |
| | A→W | **67.1 ± 7.2 / 44.4 ± 12.8** | 69.9 / 49.0 (17) | 51.3 / 26.9 | 62.1 / 62.0 |
| | D→A | 41.2 ± 2.6 / 11.4 ± 5.9 | 50.3 / 25.6 (8) | 25.2 / 3.2 | 62.6 / 61.8 |
| | D→W | 72.3 ± 3.1 / 48.6 ± 6.2 | 75.8 / 53.1 (9) | 56.1 / 17.8 | 78.5 / 78.7 |
| | W→A | 43.4 ± 6.5 / 12.2 ± 8.1 | 55.1 / 31.3 (8) | 20.9 / 7.4 | 62.1 / 62.4 |
| | W→D | 73.5 ± 5.2 / 52.1 ± 10.5 | 77.5 / 57.7 (11) | 54.3 / 19.6 | 87.8 / 87.6 |
| Table 2 (20/31) | A→D | 65.3 ± 2.9 / 26.6 ± 7.1 | 69.5 / 35.6 (21) | 47.7 / 16.3 | 73.3 / 73.1 |
| | A→W | **66.6 ± 2.8 / 25.5 ± 6.1** | 68.3 / 32.5 (25) | 42.5 / 20.9 | 63.6 / 62.6 |
| | D→A | 32.9 ± 4.6 / 2.5 ± 2.3 | 56.4 / 24.7 (3) | 9.9 / 4.4 | 64.1 / 64.3 |
| | D→W | 71.7 ± 0.7 / 27.1 ± 1.2 | 79.6 / 43.2 (8) | 53.2 / 17.7 | 80.3 / 79.4 |
| | W→A | 40.6 ± 4.9 / 4.5 ± 2.2 | 54.6 / 24.1 (4) | 14.7 / 1.4 | 63.7 / 62.8 |
| | W→D | 77.5 ± 1.9 / 42.9 ± 4.8 | 82.6 / 53.7 (11) | 60.0 / 12.7 | 89.5 / 88.4 |

- **ALL rises by 16 to 26 points over vanilla on every transfer.** It matches the paper on A→D (70.3 against 72.2, within noise) and exceeds it on A→W (67.1 against 62.1 in Table 3, 66.6 against 63.6 in Table 2). It stays 6 to 31 points below on the others.
- **Shared accuracy holds except for Table 2 with Amazon as target.** It stays at 77–100 through adaptation on Table 3 and on Table 2 A→D, A→W, D→W, W→D (93–100 on the last two), and shared_as_priv stays below 9%. On Table 2 D→A it falls from 71.6 at epoch 0 to 68.6 at epoch 30 and 49.7 at epoch 60; on W→A to 60.7. In vanilla, shared accuracy fell to 36–50 with Amazon as target.
- **PRIV is still the gap: 2.5–52.1 against 62–88 in the paper.** It improves over vanilla on every transfer except Table 2 D→A (2.5 against 4.4, within noise). The remaining loss is private samples predicted as shared, 16–94% at epoch 30. ALL peaks between epochs 3 and 28 and decays by 2 to 24 points until epoch 60; PRIV falls further (Table 2 D→A: 24.7 at epoch 3 → 2.5).
- **Amazon targets are the weakest** (PRIV 2.5–12.2, and the four largest decays from the best epoch: Table 2 D→A 23.5, Table 2 W→A 14.0, W→A 11.7, D→A 9.1). In Table 3, the private part of B_t is only 54–60% correct there, against 73–93% with DSLR or Webcam as target. In Table 2 D→A it falls from 73.0 at epoch 0 to 38.2 at epoch 60, and pl_acc_all from 51.6 to 39.8: the errors build up late.
- **Table 2 PRIV is lower than Table 3 on every transfer** (A→D 26.6 against 48.5, D→W 27.1 against 48.6). With 20 shared classes, more shared guides compete for each private sample: priv_as_shared at epoch 30 is 42–94% (Table 2 W→D lowest, D→A highest).
- Table 2 D→A first crashed in AdaBN: Amazon with 31 classes has 2,817 images = 44 × 64 + 1, and BatchNorm cannot train on a final batch of one. Fixed in 7c71273 (the batch is skipped).
- Earlier versions of `trajectory.py` and `epochs.py` dropped a resumed seed's epochs from before the restart (n = 4 in early rows). Both now keep the last logged value per epoch.

## Inconsistencies in the paper

1. **Office private classes.** Supp §5 says the private classes are those "ranked 21–31", which is 11 classes. Table 3 has |Ct| = 20, which means 10 private classes. We use indices 20–29 for Table 3 (`configs/office31.yaml`) and 20–30 for Table 2, where |Cs| = 20 and |Ct| = 31 (`configs/office20.yaml`).
2. **The 3σ rule in high dimensions.** For a Gaussian, the point μ + 3√λ·ν has squared Mahalanobis distance exactly 9, so the Algo 1 test reduces to "Mahalanobis distance > 3 from the most likely class". In D dimensions a typical in-distribution sample has squared distance about D (100 or 256). So almost every global sample, and almost every real class sample, counts as "negative": the acceptance rate is about 100%. `src_train.neg_mode=chi2` instead uses the chi-square(D) quantile as the threshold. **Result (Digits M→U):** the chi2 threshold (143 for D = 100) also accepts 100% of samples. The global Gaussian's covariance includes the spread *between* classes, so its samples lie far outside every class prototype under either rule. Both source models trained identically (same losses to 5 decimals), so the threshold is irrelevant here: the negatives are simply samples of the global Gaussian. The two adaptation runs then differ only through GPU nondeterminism: ALL 71.9 vs 70.5, PRIV 55.3 vs 53.5. **Treat differences under about 2 points between variants as noise.**
3. **Algo 1 line 9** writes P(u_s) where it means P(u_n).

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
