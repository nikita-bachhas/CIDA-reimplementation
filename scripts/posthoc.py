"""Prediction-step variants on finished runs (no retraining).

The paper predicts argmax over [gs.fd.ft restricted to Cs | gt.ft]. Two label-free changes:
  gate   gs was trained with an extra "unknown" class (the negatives, index |Cs|). If gs scores
         "unknown" above every shared class, the sample goes to the private classifier gt.
  bias   gs is fully trained and gives larger logits than the new gt, so close calls go to the
         shared side. Add one offset b to gt's logits, chosen so the share of target *train*
         samples predicted private equals the share whose nearest guide is private (the
         paper's own pseudo-labelling). No target labels are used.
  both   gate, then bias for the samples the gate leaves alone.

Prints ALL / PRIV / shared per seed and the seed-paired change from the paper's rule.
  sbatch slurm/posthoc.sbatch runs/digits_m2u runs/digits_u2m runs/digits_s2m
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
from cida.data import build_domain  # noqa: E402
from cida.prototypes import GaussianPrototypes  # noqa: E402

METHODS = ["paper", "gate", "bias", "both"]


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
    src = load_ckpt(Path(cfg["out_dir"]) / "source" / "final.pt")
    mu_s = GaussianPrototypes.from_state_dict(src["protos"], device).means
    g_priv = torch.stack([model.ft(torch.stack([tgt_train[i][0] for i in labelled[c]]).to(device)).mean(0)
                          for c in range(ns, n)])
    guides = torch.cat([model.fe(mu_s), g_priv]).cpu()
    S_tr, P_tr, V_tr, _ = outputs(model, tgt_train, device, nw, lab)
    pi = (torch.cdist(V_tr, guides).argmin(1) >= ns).float().mean().item()
    d = P_tr.max(1).values - S_tr[:, :ns].max(1).values  # private wins iff d + b > 0
    b = -torch.quantile(d, 1 - pi).item()
    gate_rate = (S_tr[:, ns] > S_tr[:, :ns].max(1).values).float().mean().item()

    S, P, _, y = outputs(model, tgt_eval, device, nw, excl)
    res = {m: scores(predict(S, P, ns, m, b), y, ns) for m in METHODS}
    out = {"run": run_dir.name, "pi_private": pi, "offset": b, "gate_rate_train": gate_rate, **res}
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
        print(f"\n===== {task}  ({len(runs)} runs)   ALL / PRIV / shared")
        print(f"{'seed':6}{'priv share':>11}{'offset':>7}{'gated':>7} " + "".join(f"{m:>20}" for m in METHODS))
        rows = []
        for r in runs:
            o = analyse(r, device, args.num_workers)
            rows.append(o)
            print(f"{r.name[-5:]:6}{100 * o['pi_private']:10.1f}%{o['offset']:7.2f}{100 * o['gate_rate_train']:6.1f}% "
                  + "".join(f"{o[m]['all']:6.1f} /{o[m]['priv']:5.1f} /{o[m]['shared']:5.1f}" for m in METHODS))
        if rows:
            print(f"{'mean':32}" + "".join(
                f"{st.mean(o[m]['all'] for o in rows):6.1f} /{st.mean(o[m]['priv'] for o in rows):5.1f} /"
                f"{st.mean(o[m]['shared'] for o in rows):5.1f}" for m in METHODS))
            for m in METHODS[1:]:
                dall = [o[m]["all"] - o["paper"]["all"] for o in rows]
                dpriv = [o[m]["priv"] - o["paper"]["priv"] for o in rows]
                up = sum(x > 0 for x in dpriv)
                print(f"  {m:5} vs paper, per seed: ALL {' '.join(f'{x:+5.1f}' for x in dall)} | "
                      f"PRIV {' '.join(f'{x:+5.1f}' for x in dpriv)}  (PRIV up on {up}/{len(rows)} seeds)")


if __name__ == "__main__":
    main()
