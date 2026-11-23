"""Prediction-step variants on finished runs (no retraining).

The paper predicts argmax over [gs.fd.ft restricted to Cs | gt.ft]. Two label-free changes:
  gate   gs was trained with an extra "unknown" class (the negatives, index |Cs|). If gs scores
         "unknown" above every shared class, the sample goes to the private classifier gt.
  bias   gs is fully trained and gives larger logits than the new gt, so close calls go to the
         shared side. Add one offset b to gt's logits, chosen so the share of target *train*
         samples predicted private equals the share whose nearest guide is private (the
         paper's own pseudo-labelling). No target labels are used.
  both   gate, then bias for the samples the gate leaves alone.

Label-free relabelling of the evaluation images from the guides (fe(mu_s) for shared classes,
mean ft(labelled images) for private ones), ignoring the classifiers gs / gt. Transductive: the
labels come from the unlabelled evaluation images themselves (on Office these are also the
adaptation images; on Digits the target test split). Same functions as scripts/guide_diag.py:
  nearest  nearest guide (guide_diag "basic")
  km       k-means started at the guides, every guide moves; then nearest guide
  km-bal   the same with equal cluster sizes (Sinkhorn); then nearest guide
  lp       label spreading from the guides over a cosine 10-NN graph, class-balanced (skipped
           above LP_MAX images)

Prints ALL / PRIV / shared per seed (per image and class-averaged), the seed-paired change from the
paper's rule, and a class-wise table (every class: seed-mean accuracy under each rule, and per seed
under lp).

Sweep (diagnostic, uses labels only to score): the gt offset is swept so that a share q of the
evaluation images is predicted private, q = 0.20 .. 0.80. The offset for each q is computed from the
logits alone, so seeds are averaged at equal q. The curve shows the best PRIV each shared level allows
by moving the shared/private boundary; the label-free "bias" point is marked by its predicted share.
  sbatch slurm/posthoc.sbatch runs/digits_m2u runs/digits_u2m runs/digits_s2m
  sbatch slurm/posthoc.sbatch runs/office_a2d --runs adapt_1shot_abl-none_rect
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import CIDA  # noqa: E402
from cida.common import get_device, load_ckpt  # noqa: E402
from cida.data import base_dataset, build_domain, resolve_classes  # noqa: E402
from cida.metrics import cida_metrics  # noqa: E402
from cida.prototypes import GaussianPrototypes  # noqa: E402
from cida.rectify import kmeans_guides, label_propagation  # noqa: E402

CLS_METHODS = ["paper", "gate", "bias", "both"]  # use the classifiers gs / gt
GUIDE_METHODS = ["nearest", "km", "km-bal", "lp"]  # relabel from the guides, classifiers unused
METHODS = CLS_METHODS + GUIDE_METHODS
LP_MAX = 20000  # label propagation is dense N x N
KM_ITERS = 30
SWEEP_Q = [round(0.20 + 0.02 * k, 2) for k in range(31)]  # predicted-private share of the eval images


@torch.no_grad()
def outputs(model, ds, device, nw, excl=frozenset()):
    """Shared-head logits incl. unknown [N, |Cs|+1], private logits [N, |C't|], features, labels."""
    S, P, V, Y = [], [], [], []
    for x, y, i in DataLoader(ds, 256, num_workers=nw):
        keep = torch.tensor([j.item() not in excl for j in i])
        if not keep.any():
            continue
        v = model.ft(x[keep].to(device))
        S.append(model.gs(model.fd(v)).cpu()), P.append(model.gt(v).cpu()), V.append(v.cpu()), Y.append(y[keep])
    return torch.cat(S), torch.cat(P), torch.cat(V), torch.cat(Y)


def predict(S, P, ns, method, b=0.0):
    shared, unknown = S[:, :ns], S[:, ns]
    priv_logits = P + (b if method in ("bias", "both") else 0.0)
    pred = torch.cat([shared, priv_logits], 1).argmax(1)
    if method in ("gate", "both"):
        gated = unknown > shared.max(1).values
        pred[gated] = P[gated].argmax(1) + ns
    return pred


def guide_labels(V, G, ns, method):
    """Label-free labels of the samples V from the guides G (see the module docstring); None if skipped."""
    n = len(G)
    if method == "nearest":
        return torch.cdist(V, G).argmin(1)
    if method in ("km", "km-bal"):
        return torch.cdist(V, kmeans_guides(V, G, range(n), KM_ITERS, balanced=method == "km-bal")).argmin(1)
    if method == "lp":
        return label_propagation(V, G)[0] if len(V) <= LP_MAX else None
    raise ValueError(method)


def full_scores(pred, y, ns, n):
    """cida_metrics (per image, class-averaged, cross-group) plus per-class accuracy and predicted-private share."""
    out = cida_metrics(pred, y, ns, n)
    out["per_class"] = [100 * (pred[y == c] == c).float().mean().item() if (y == c).any() else float("nan")
                        for c in range(n)]
    out["n_per_class"] = [int((y == c).sum()) for c in range(n)]
    out["pred_priv_share"] = (pred >= ns).float().mean().item()
    return out


def scores(pred, y, ns):
    p = y >= ns
    ok = (pred == y).float()
    return {"all": 100 * ok.mean().item(), "priv": 100 * ok[p].mean().item(), "shared": 100 * ok[~p].mean().item()}


@torch.no_grad()
def analyse(run_dir, device, nw):
    ck = load_ckpt(run_dir / "final.pt")
    cfg, labelled = ck["cfg"], ck["labelled"]
    dom = cfg["tgt_domain"]
    tgt_eval = build_domain(cfg, dom, cfg["tgt_eval_split"], train_aug=False)
    tgt_train = build_domain(cfg, dom, cfg["tgt_train_split"], train_aug=False)
    ns, n = tgt_eval.n_shared, tgt_eval.n_shared + tgt_eval.n_private
    model = CIDA(cfg, ns, n - ns, cfg["adapt"].get("ablation", "none")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    lab = {i for c in range(ns, n) for i in labelled[c]}
    excl = lab if cfg["tgt_eval_split"] == cfg["tgt_train_split"] else frozenset()

    # Calibration on the unlabelled target train split: share of samples whose nearest guide is private.
    src = load_ckpt(run_dir.parent / "source" / "final.pt")  # works after the task folder is renamed
    mu_s = GaussianPrototypes.from_state_dict(src["protos"], device).means
    g_priv = torch.stack([model.ft(torch.stack([tgt_train[i][0] for i in labelled[c]]).to(device)).mean(0)
                          for c in range(ns, n)])
    guides = torch.cat([model.fe(mu_s), g_priv]).cpu()
    S_tr, P_tr, V_tr, _ = outputs(model, tgt_train, device, nw, lab)
    pi = (torch.cdist(V_tr, guides).argmin(1) >= ns).float().mean().item()
    d = P_tr.max(1).values - S_tr[:, :ns].max(1).values  # private wins iff d + b > 0
    b = -torch.quantile(d, 1 - pi).item()
    gate_rate = (S_tr[:, ns] > S_tr[:, :ns].max(1).values).float().mean().item()

    S, P, V, y = outputs(model, tgt_eval, device, nw, excl)
    V = V.float()
    res = {}
    for m in CLS_METHODS:
        res[m] = full_scores(predict(S, P, ns, m, b), y, ns, n)
    for m in GUIDE_METHODS:
        pred = guide_labels(V, guides.float(), ns, m)
        res[m] = full_scores(pred, y, ns, n) if pred is not None else None
    _, _, names = base_dataset(cfg, dom, cfg["tgt_eval_split"], False)
    classes = [str(names[c]) for c in resolve_classes(cfg["shared"], names) + resolve_classes(cfg["private"], names)]

    # Sweep the offset over the predicted-private share q (labels only score the result).
    d_ev = P.max(1).values - S[:, :ns].max(1).values
    sweep = []
    for q in SWEEP_Q:
        bq = -torch.quantile(d_ev, 1 - q).item()
        sweep.append({"q": q, "offset": bq, **scores(predict(S, P, ns, "bias", bq), y, ns)})
    out = {"run": run_dir.name, "classes": classes, "n_shared": ns, "pi_private": pi, "offset": b, "gate_rate_train": gate_rate,
           "true_priv_share": (y >= ns).float().mean().item(), **res, "sweep": sweep}
    with open(run_dir / "posthoc.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="+")
    ap.add_argument("--runs", default="adapt_1shot_abl-none", help="run-name prefix before _seed<k>")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    device = get_device("auto")
    for task in args.task_dirs:
        runs = sorted(p for p in Path(task).glob(f"{args.runs}_seed*") if (p / "final.pt").exists())
        print(f"\n===== {task}  runs={args.runs}  ({len(runs)} runs)   ALL / PRIV / shared")
        rows = [analyse(r, device, args.num_workers) for r in runs]
        if not rows:
            continue
        print(f"{'seed':6}{'priv share':>11}{'offset':>7}{'gated':>7}")
        for o in rows:
            print(f"{o['run'][-5:]:6}{100 * o['pi_private']:10.1f}%{o['offset']:7.2f}{100 * o['gate_rate_train']:6.1f}%")
        methods = [m for m in METHODS if all(o[m] is not None for o in rows)]
        for block in (CLS_METHODS, GUIDE_METHODS):
            ms = [m for m in block if m in methods]
            print(f"\n{'':6}" + "".join(f"{m:>20}" for m in ms))
            for o in rows:
                print(f"{o['run'][-5:]:6}" + "".join(
                    f"{o[m]['all']:6.1f} /{o[m]['priv']:5.1f} /{o[m]['shared']:5.1f}" for m in ms))
            for label, sfx in (("mean", ""), ("cls", "_cls")):  # per image; class-averaged
                print(f"{label:6}" + "".join(
                    f"{st.mean(o[m]['all' + sfx] for o in rows):6.1f} /{st.mean(o[m]['priv' + sfx] for o in rows):5.1f} /"
                    f"{st.mean(o[m]['shared' + sfx] for o in rows):5.1f}" for m in ms))
            print(f"{'p->s':6}" + "".join(f"{st.mean(o[m]['priv_as_shared'] for o in rows):20.1f}" for m in ms))
        print("  rows: per seed; mean = seed mean per image; cls = class-averaged; p->s = priv_as_shared %")
        print("\n  Seed-paired change from the paper's rule (per image)")
        for m in methods[1:]:
            dall = [o[m]["all"] - o["paper"]["all"] for o in rows]
            dpriv = [o[m]["priv"] - o["paper"]["priv"] for o in rows]
            up = sum(x > 0 for x in dpriv)
            print(f"  {m:7} ALL {' '.join(f'{x:+5.1f}' for x in dall)} | "
                  f"PRIV {' '.join(f'{x:+5.1f}' for x in dpriv)}  (PRIV up on {up}/{len(rows)} seeds)")
        print_class_table(rows, [m for m in ("paper", *GUIDE_METHODS) if m in methods],
                          "lp" if "lp" in methods else "nearest")
        print_sweep(rows)


def print_class_table(rows, methods, per_seed):
    """Every class: seed-mean accuracy under each rule, and per-seed accuracy under `per_seed`."""
    classes, ns = rows[0]["classes"], rows[0]["n_shared"]
    seeds = [o["run"].rsplit("_", 1)[-1] for o in rows]
    print(f"\n  Class-wise accuracy (%), seed mean per rule | per seed under {per_seed}")
    print(f"  {'class':>16} {'grp':>4} {'n':>3} " + "".join(f"{m:>8}" for m in methods) + " |"
          + "".join(f"{s:>7}" for s in seeds))
    for c, name in enumerate(classes):
        means = [st.mean(o[m]["per_class"][c] for o in rows) for m in methods]
        print(f"  {name:>16} {'shar' if c < ns else 'priv':>4} {rows[0][methods[0]]['n_per_class'][c]:>3} "
              + "".join(f"{x:8.1f}" for x in means) + " |"
              + "".join(f"{o[per_seed]['per_class'][c]:7.1f}" for o in rows))


def print_sweep(rows):
    """Seed-mean ALL / PRIV / shared against the predicted-private share q, with marked points."""
    curve = [{k: st.mean(o["sweep"][j][k] for o in rows) for k in ("q", "all", "priv", "shared")}
             for j in range(len(SWEEP_Q))]
    share = {m: st.mean(o[m]["pred_priv_share"] for o in rows) for m in ("paper", "bias")}
    true = st.mean(o["true_priv_share"] for o in rows)
    best = max(curve, key=lambda c: c["all"])
    bal = min(curve, key=lambda c: abs(c["priv"] - c["shared"]))
    print(f"\n  Offset sweep (labels used to score only), seed mean. Predicted-private share: "
          f"paper rule {100 * share['paper']:.1f}%, label-free bias {100 * share['bias']:.1f}%, "
          f"true {100 * true:.1f}%")
    print(f"  {'q':>5} {'ALL':>6} {'PRIV':>6} {'shared':>7}")
    for c in curve:
        tags = [t for t, hit in (("best ALL", c is best), ("PRIV~shared", c is bal),
                                 ("~paper rule", abs(c["q"] - share["paper"]) < 0.01),
                                 ("~bias", abs(c["q"] - share["bias"]) < 0.01),
                                 ("~true share", abs(c["q"] - true) < 0.01)) if hit]
        print(f"  {c['q']:5.2f} {c['all']:6.1f} {c['priv']:6.1f} {c['shared']:7.1f}  {', '.join(tags)}")
    per_seed = [max(o["sweep"], key=lambda s: s["all"]) for o in rows]
    print("  per-seed best-ALL point (q: ALL/PRIV/shared): "
          + "  ".join(f"{s['q']:.2f}: {s['all']:.1f}/{s['priv']:.1f}/{s['shared']:.1f}" for s in per_seed))


if __name__ == "__main__":
    main()
