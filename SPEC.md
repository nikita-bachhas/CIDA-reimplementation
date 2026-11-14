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
| LeNet BatchNorm | On | "Similar to [OSBP]", whose LeNet uses BN |
| LeNet layer sizes | conv5, conv5, conv3 stride 2, conv3 stride 2 | Inferred from the 28→24→11→5 feature-map sizes in supp Table 2 |
| BN when computing private guides | The one-shot images go through ft in the same batch as the target samples | Avoids BatchNorm statistics from a batch of one image per class |
| Digits data | Adapt on the target train split, evaluate on the target test split | Not given |
| Office data | Transductive: adapt and evaluate on the whole target domain, excluding the labelled one-shot images from evaluation | Not given |
| Model selection | Last epoch (no target labels are used for selection) | Source-free and unsupervised |
| ALL averaging | Report both per-sample and class-averaged accuracy | Not given |
| Source warm-up | Trains fs and gs (Algo 2 line 4 writes only θ_fs, but gs must be trained too) | Algo 2 typo |

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
