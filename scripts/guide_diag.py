"""Guide asymmetry and guide rectification on finished runs (no retraining).

Question 1: are private samples farther from their own guide than shared samples are from theirs?
A private guide is one labelled image, so for a target class with mean m_c and spread tr(S_c)
    E ||g_c - m_c||^2 = tr(S_c)            (one-shot guide: one random sample of the class)
    ||g_c - m_c||^2   = ||delta_c||^2      (shared guide fe(mu_c): the domain-shift offset)
and a sample's expected squared distance to its own guide is tr(S_c) + ||g_c - m_c||^2. Reported
per group (shared / private), averaged over classes:
    off/tr   ||g_c - m_c||^2 / tr(S_c)     theory: ~1 for one-shot guides; small = good guide
    d_own    mean squared distance of a sample to its own class's guide, / mean tr(S)
If private off/tr >> shared off/tr, nearest-guide labels drift private -> shared (Office);
the reverse (large shared offset, i.e. domain shift) gives shared -> private (Digits).

Question 2: would rectified private guides give cleaner pseudo-labels? Nearest-guide labelling of
the unlabelled target train split (target labels used only to score), with guides:
    basic      the method's guides: fe(mu_c) and the one-shot features
    rect       private guides rectified (cida/rectify.py, select=distance), label-free
    rect-m     same, members chosen by the d1/d2 margin
    rect-all   shared guides rectified too
    km         k-means over all samples started at the basic guides, every guide moves (label-free)
    km-priv    same, shared guides fixed at fe(mu_c)
    km-anch    km with the starting guide keeping weight --anchor in each update
    oracle     private guides = true target class means (upper bound; uses labels)
    oracle-all every guide = true target class mean (the ceiling for km; uses labels)
Columns: pseudo-label accuracy ALL / PRIV / shared, priv->shared %, and B_t private precision
(top n_percent of each guide's members, as adapt.py selects B_t).

--stage final   each seed's adapted ft and fe (features shaped by training with basic guides)
--stage start   the source fs with target BatchNorm statistics (AdaBN) and fe = identity, i.e. the
                features adaptation starts from (main config only: needs model.proj_residual)

  sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d
  sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d --stage start
  sbatch slurm/py.sbatch scripts/guide_diag.py runs/office_a2d --runs adapt_1shot_abl-none_rect
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import CIDA, adabn  # noqa: E402
from cida.common import get_device, load_ckpt  # noqa: E402
from cida.data import build_domain  # noqa: E402
from cida.prototypes import GaussianPrototypes  # noqa: E402
from cida.rectify import kmeans_guides, rectify_guides  # noqa: E402

VARIANTS = ["basic", "rect", "rect-m", "rect-all", "km", "km-priv", "km-anch", "oracle", "oracle-all"]
ASYM = ("basic", "rect", "km")


@torch.no_grad()
def features(ft, ds, device, nw):
    ft.eval()
    V, Y = [], []
    for x, y, _ in DataLoader(ds, 256, num_workers=nw):
        V.append(ft(x.to(device)).float().cpu())
        Y.append(torch.as_tensor(y))
    return torch.cat(V), torch.cat(Y)


def asymmetry(V, Y, G, ns, n):
    """Per-group off/tr and d_own (see module docstring)."""
    rows = {"shared": [], "private": []}
    d_own = {"shared": [], "private": []}
    tr_all = []
    for c in range(n):
        Xc = V[Y == c]
        if len(Xc) < 2:
            continue
        m = Xc.mean(0)
        tr = (Xc - m).pow(2).sum(1).mean().item()
        tr_all.append(tr)
        grp = "shared" if c < ns else "private"
        rows[grp].append((G[c] - m).pow(2).sum().item() / tr)
        d_own[grp] += (Xc - G[c]).pow(2).sum(1).tolist()
    tr_mean = st.mean(tr_all)
    return {g: {"off/tr": st.mean(rows[g]) if rows[g] else float("nan"),
                "d_own": st.mean(d_own[g]) / tr_mean if d_own[g] else float("nan")} for g in rows}


def pseudo_label_scores(V, Y, G, ns, n, n_percent):
    dist = torch.cdist(V, G)
    d, k = dist.min(1)
    p = Y >= ns
    ok = (k == Y).float()
    b_ok, b_n = 0.0, 0
    for c in range(ns, n):
        members = (k == c).nonzero().squeeze(1)
        if len(members) == 0:
            continue
        keep = members[d[members].argsort()[:max(1, int(n_percent / 100 * len(members)))]]
        b_ok += (Y[keep] == c).sum().item()
        b_n += len(keep)
    return {"all": 100 * ok.mean().item(), "priv": 100 * ok[p].mean().item(), "shared": 100 * ok[~p].mean().item(),
            "priv_as_shared": 100 * (k[p] < ns).float().mean().item(),
            "B_prec_priv": 100 * b_ok / b_n if b_n else float("nan")}


@torch.no_grad()
def analyse(run_dir, device, nw, stage="final", eps=10.0, anchor=0.3, iters=3, km_iters=30):
    ck = load_ckpt(run_dir / "final.pt")
    cfg, labelled = ck["cfg"], ck["labelled"]
    acfg = cfg["adapt"]
    tgt_plain = build_domain(cfg, cfg["tgt_domain"], cfg["tgt_train_split"], train_aug=False)
    ns, n = tgt_plain.n_shared, tgt_plain.n_shared + tgt_plain.n_private
    model = CIDA(cfg, ns, n - ns, acfg.get("ablation", "none")).to(device)
    model.load_state_dict(ck["model"])
    src = load_ckpt(run_dir.parent / "source" / "final.pt")
    mu_s = GaussianPrototypes.from_state_dict(src["protos"], device).means
    if stage == "start":
        if not cfg["model"].get("proj_residual", True):
            return None  # fe is not exactly the identity at the start; not reconstructable here
        model.ft.load_state_dict(src["fs"])
        if acfg.get("adabn", True):
            adabn(model.ft, DataLoader(tgt_plain, 256, num_workers=nw), device)
        fe = nn.Identity()
    else:
        fe = model.fe.eval()
    V_all, Y_all = features(model.ft, tgt_plain, device, nw)
    lab = [i for c in range(ns, n) for i in labelled[c]]
    lab_y = torch.tensor([c for c in range(ns, n) for _ in labelled[c]])
    g_priv = torch.stack([V_all[lab][lab_y == c].mean(0) for c in range(ns, n)])
    keep = torch.ones(len(Y_all), dtype=torch.bool)
    keep[lab] = False  # labelled images are neither members nor scored
    V, Y = V_all[keep], Y_all[keep]
    base = torch.cat([fe(mu_s).float().cpu(), g_priv])
    npct = acfg.get("n_percent", 20)
    priv_rows, all_rows = range(ns, n), range(n)
    guides = {
        "basic": base,
        "rect": rectify_guides(V, base, priv_rows, npct, eps, anchor, iters)[0],
        "rect-m": rectify_guides(V, base, priv_rows, npct, eps, anchor, iters, select="margin")[0],
        "rect-all": rectify_guides(V, base, all_rows, npct, eps, anchor, iters)[0],
        "km": kmeans_guides(V, base, all_rows, km_iters),
        "km-priv": kmeans_guides(V, base, priv_rows, km_iters),
        "km-anch": kmeans_guides(V, base, all_rows, km_iters, anchor=anchor),
    }
    means = torch.stack([V[Y == c].mean(0) if (Y == c).any() else base[c] for c in range(n)])
    guides["oracle"] = torch.cat([base[:ns], means[ns:]])
    guides["oracle-all"] = means
    out = {"run": run_dir.name, "stage": stage,
           "asym": {v: asymmetry(V, Y, guides[v], ns, n) for v in ASYM},
           **{v: pseudo_label_scores(V, Y, g, ns, n, npct) for v, g in guides.items()}}
    with open(run_dir / f"guide_diag_{stage}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dirs", nargs="+")
    ap.add_argument("--runs", default="adapt_1shot_abl-none", help="run-name prefix before _seed<k>")
    ap.add_argument("--stage", choices=["final", "start"], default="final")
    ap.add_argument("--eps", type=float, default=10.0)
    ap.add_argument("--anchor", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--km_iters", type=int, default=30, help="max k-means rounds (km variants)")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    device = get_device("auto")
    for task in args.task_dirs:
        runs = sorted(p for p in Path(task).glob(f"{args.runs}_seed*") if (p / "final.pt").exists())
        print(f"\n===== {task}  runs={args.runs}  stage={args.stage}  ({len(runs)} runs)  eps={args.eps} "
              f"anchor={args.anchor} iters={args.iters} km_iters={args.km_iters}")
        rows = [o for r in runs if (o := analyse(r, device, args.num_workers, args.stage, args.eps, args.anchor,
                                                 args.iters, args.km_iters)) is not None]
        if not rows:
            print("  no runs (or --stage start on a run without residual fe)")
            continue
        print("Guide asymmetry  off/tr shared / private   d_own shared / private   (basic | rect | km)")
        for o in rows:
            print(f"  {o['run'][-5:]:6}" + " | ".join(
                f"{o['asym'][v]['shared']['off/tr']:5.2f} /{o['asym'][v]['private']['off/tr']:6.2f}  "
                f"{o['asym'][v]['shared']['d_own']:5.2f} /{o['asym'][v]['private']['d_own']:6.2f}" for v in ASYM))
        print("Nearest-guide pseudo-labels   ALL / PRIV / shared / priv->shared / B_t priv precision")
        for v in VARIANTS:
            for o in rows:
                s = o[v]
                print(f"  {v:9}{o['run'][-5:]:6}{s['all']:6.1f} /{s['priv']:5.1f} /{s['shared']:5.1f} /"
                      f"{s['priv_as_shared']:5.1f} /{s['B_prec_priv']:5.1f}")
            m = {k: st.mean(o[v][k] for o in rows) for k in rows[0][v]}
            print(f"  {v:9}{'mean':6}{m['all']:6.1f} /{m['priv']:5.1f} /{m['shared']:5.1f} /"
                  f"{m['priv_as_shared']:5.1f} /{m['B_prec_priv']:5.1f}")
        for v in VARIANTS[1:]:
            dp = [o[v]["priv"] - o["basic"]["priv"] for o in rows]
            da = [o[v]["all"] - o["basic"]["all"] for o in rows]
            print(f"  {v:9} vs basic per seed: ALL {' '.join(f'{x:+5.1f}' for x in da)} | "
                  f"PRIV {' '.join(f'{x:+5.1f}' for x in dp)}")


if __name__ == "__main__":
    main()
