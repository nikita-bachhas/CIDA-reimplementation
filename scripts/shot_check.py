"""How good are hand-picked one-shot images as guides, compared with the runs' random draws? (no training)

Nearest-guide and label-propagation labelling of the target images (labels used only to score), with
the private guides built from
  seed k   the one-shot images of run <runs>_seed<k> (the random draw of the main runs)
  fixed    adapt.fixed_shots of --config
All sets are scored on the same images: every set's labelled images are excluded from scoring.
  start    source fs + AdaBN, fe = identity: identical features for every set, only the shot differs
  final    each seed's adapted ft and fe, with its own shot and with the fixed shot (features were
           trained around the seed's own shot, so this favours the own shot)
  python scripts/shot_check.py runs/office_a2d --config configs/office_a2d_fixedshot.yaml
"""
import argparse
import statistics as st
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import CIDA, adabn  # noqa: E402
from cida.common import get_device, load_ckpt, load_config  # noqa: E402
from cida.data import build_domain, fixed_labelled  # noqa: E402
from cida.prototypes import GaussianPrototypes  # noqa: E402
from cida.rectify import label_propagation  # noqa: E402
from scripts.guide_diag import features, label_scores  # noqa: E402


def score_sets(V_all, Y_all, shared_guides, sets, keep, ns, n):
    """{set: {rule: scores}} for nearest guide and lp; per-class private accuracy included."""
    V, Y = V_all[keep], Y_all[keep]
    out = {}
    for name, lab in sets.items():
        g_priv = torch.stack([V_all[lab[c]].mean(0) for c in range(ns, n)])
        G = torch.cat([shared_guides, g_priv])
        d, k_nn = torch.cdist(V, G).min(1)
        k_lp, c_lp = label_propagation(V, G)
        out[name] = {}
        for rule, k, conf in (("nearest", k_nn, -d), ("lp", k_lp, c_lp)):
            s = label_scores(k, conf, Y, ns, n, 20)
            s["per_class"] = [100 * (k[Y == c] == c).float().mean().item() for c in range(ns, n)]
            out[name][rule] = s
    return out


def show(title, res, names):
    print(f"\n{title}   ALL / PRIV / shared / priv->shared")
    for rule in ("nearest", "lp"):
        for name, r in res.items():
            s = r[rule]
            print(f"  {rule:8}{name:7}{s['all']:6.1f} /{s['priv']:5.1f} /{s['shared']:5.1f} /{s['priv_as_shared']:5.1f}")
        for label, rows in (("rnd avg", [r[rule] for nm, r in res.items() if not nm.endswith(("fx", "fixed"))]),
                            ("fix avg", [r[rule] for nm, r in res.items() if nm.endswith(("fx", "fixed"))])):
            if rows:
                print(f"  {rule:8}{label:7}" + " /".join(f"{st.mean(s[k] for s in rows):5.1f}"
                                                       for k in ("all", "priv", "shared", "priv_as_shared")))
    print(f"  private class accuracy, nearest guide:  {'':7}" + "".join(f"{nm[:8]:>9}" for nm in names))
    for name, r in res.items():
        print(f"  {'':40}{name:7}" + "".join(f"{x:9.1f}" for x in r["nearest"]["per_class"]))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--config", required=True, help="config with adapt.fixed_shots")
    ap.add_argument("--runs", default="adapt_1shot_abl-none", help="run-name prefix before _seed<k>")
    ap.add_argument("--stage", choices=["start", "final", "both"], default="both")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args(argv)
    device, task = get_device("auto"), Path(args.task_dir)
    cfg = load_config(args.config)
    tgt = build_domain(cfg, cfg["tgt_domain"], cfg["tgt_train_split"], train_aug=False)
    ns, n = tgt.n_shared, tgt.n_shared + tgt.n_private
    names = tgt.class_names[ns:]
    runs = sorted(p for p in task.glob(f"{args.runs}_seed*") if (p / "final.pt").exists())
    cks = {p.name.rsplit("_", 1)[1]: load_ckpt(p / "final.pt") for p in runs}
    sets = {s: {int(c): v for c, v in ck["labelled"].items()} for s, ck in cks.items()}
    sets["fixed"] = fixed_labelled(tgt, range(ns, n), cfg["adapt"]["fixed_shots"])
    keep = torch.ones(len(tgt), dtype=torch.bool)
    keep[[i for lab in sets.values() for v in lab.values() for i in v]] = False
    print(f"===== {task}  fixed shots from {args.config}; scored on {int(keep.sum())} of {len(tgt)} images "
          f"(all sets' labelled images excluded)")
    src = load_ckpt(task / "source" / "final.pt")
    mu_s = GaussianPrototypes.from_state_dict(src["protos"], device).means
    model = CIDA(cfg, ns, n - ns, "none").to(device)
    loader = DataLoader(tgt, 64, num_workers=args.num_workers)
    if args.stage in ("start", "both"):
        model.ft.load_state_dict(src["fs"])
        if cfg["adapt"].get("adabn", True):
            adabn(model.ft, loader, device)
        V_all, Y_all = features(model.ft, tgt, device, args.num_workers)
        show("Start (fs + AdaBN, fe = id), same features for every set",
             score_sets(V_all, Y_all, mu_s.float().cpu(), sets, keep, ns, n), names)
    if args.stage in ("final", "both"):
        res = {}
        for s, ck in cks.items():
            model.load_state_dict(ck["model"])
            model.eval()
            V_all, Y_all = features(model.ft, tgt, device, args.num_workers)
            r = score_sets(V_all, Y_all, model.fe(mu_s).float().cpu(), {s: sets[s], "fixed": sets["fixed"]}, keep, ns, n)
            res[f"{s}"], res[f"{s}fx"] = r[s], r["fixed"]
        show("Final (each seed's adapted ft, fe): own shot (seedK) vs fixed shot (seedKfx)", res, names)


if __name__ == "__main__":
    main()
