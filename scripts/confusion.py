"""Where the private-class errors go: confusion matrix per finished adaptation run.

Re-evaluates each run's final.pt on the target eval split (no training) and writes
<run>/confusion.json and <run>/oneshot.png (the labelled one-shot images). Prints, per seed:
  PRIV          accuracy on private-class samples (as in final_metrics.json)
  PRIV|priv     the same samples, argmax over the private logits only: PRIV if no private
                sample were lost to a shared class (separates head competition from confusion)
  per class     accuracy of each private class, and where its errors mostly go
  guide         the guide nearest to that class's one-shot guide, and the distance ratio
                (nearest / median distance between guides; small = the two guides nearly coincide)

Needs a GPU node on the cluster (it runs the model):
  sbatch slurm/confusion.sbatch runs/digits_m2u runs/digits_u2m runs/digits_s2m
Locally:
  python scripts/confusion.py runs/digits_m2u [--runs adapt_1shot_abl-none]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import CIDA  # noqa: E402
from cida.common import get_device, load_ckpt  # noqa: E402
from cida.data import base_dataset, build_domain, resolve_classes  # noqa: E402
from cida.prototypes import GaussianPrototypes  # noqa: E402


@torch.no_grad()
def analyse(run_dir, device, num_workers):
    ck = load_ckpt(run_dir / "final.pt")
    cfg, labelled = ck["cfg"], ck["labelled"]
    dom = cfg["tgt_domain"]
    _, _, names = base_dataset(cfg, dom, cfg["tgt_eval_split"], False)
    cls_names = [str(names[c]) for c in resolve_classes(cfg["shared"], names) + resolve_classes(cfg["private"], names)]
    tgt_eval = build_domain(cfg, dom, cfg["tgt_eval_split"], train_aug=False)
    tgt_train = build_domain(cfg, dom, cfg["tgt_train_split"], train_aug=False)
    ns, n = tgt_eval.n_shared, tgt_eval.n_shared + tgt_eval.n_private
    private = list(range(ns, n))

    model = CIDA(cfg, ns, n - ns, cfg["adapt"].get("ablation", "none")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    lab_idx = {c: labelled[c] for c in private}
    excl = {i for v in lab_idx.values() for i in v} if cfg["tgt_eval_split"] == cfg["tgt_train_split"] else set()
    preds, preds_p, ys = [], [], []
    for x, y, i in DataLoader(tgt_eval, 256, num_workers=num_workers):
        keep = torch.tensor([j.item() not in excl for j in i])
        if not keep.any():
            continue
        logits = model(x[keep].to(device)).cpu()
        preds.append(logits.argmax(1))
        preds_p.append(logits[:, ns:].argmax(1) + ns)
        ys.append(y[keep])
    pred, pred_p, y = torch.cat(preds), torch.cat(preds_p), torch.cat(ys)
    conf = torch.zeros(n, n, dtype=torch.long)
    conf.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    is_p = y >= ns

    # Guides as in adaptation: fe(mu_s) for shared classes, mean ft(one-shot images) for private.
    src = load_ckpt(Path(cfg["out_dir"]) / "source" / "final.pt")
    mu_s = GaussianPrototypes.from_state_dict(src["protos"], device).means
    oneshot = {c: torch.stack([tgt_train[i][0] for i in lab_idx[c]]) for c in private}
    g_priv = torch.stack([model.ft(oneshot[c].to(device)).mean(0) for c in private])
    guides = torch.cat([model.fe(mu_s), g_priv]).cpu()
    dist = torch.cdist(guides, guides)
    median = dist[~torch.eye(n, dtype=torch.bool)].median()
    dist.fill_diagonal_(float("inf"))

    per_class = {}
    for c in private:
        row = conf[c].clone()
        total = row.sum().item()
        row[c] = -1
        worst = row.argmax().item()
        near = dist[c].argmin().item()
        per_class[cls_names[c]] = {
            "acc": 100 * conf[c, c].item() / max(total, 1),
            "top_error": cls_names[worst], "top_error_pct": 100 * conf[c, worst].item() / max(total, 1),
            "to_shared_pct": 100 * conf[c, :ns].sum().item() / max(total, 1),
            "nearest_guide": cls_names[near], "nearest_ratio": (dist[c, near] / median).item(),
            "oneshot_idx": [int(tgt_train.indices[i]) for i in lab_idx[c]],
        }
    out = {
        "run": run_dir.name, "classes": cls_names, "confusion": conf.tolist(),
        "priv": 100 * (pred[is_p] == y[is_p]).float().mean().item(),
        "priv_given_priv": 100 * (pred_p[is_p] == y[is_p]).float().mean().item(),
        "shared": 100 * (pred[~is_p] == y[~is_p]).float().mean().item(),
        "per_private_class": per_class,
    }
    with open(run_dir / "confusion.json", "w") as f:
        json.dump(out, f, indent=2)
    try:
        from torchvision.utils import save_image
        imgs = torch.cat([oneshot[c][:1] for c in private])
        lo, hi = imgs.amin(dim=(1, 2, 3), keepdim=True), imgs.amax(dim=(1, 2, 3), keepdim=True)
        save_image((imgs - lo) / (hi - lo + 1e-8), run_dir / "oneshot.png", nrow=len(private), padding=2)
    except Exception as e:  # the image is a convenience only
        print(f"  (oneshot.png not written: {e})")
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
        print(f"\n===== {task}  ({len(runs)} runs)")
        for r in runs:
            o = analyse(r, device, args.num_workers)
            print(f"{r.name[-5:]}  PRIV {o['priv']:5.1f}  PRIV|priv {o['priv_given_priv']:5.1f}  shared {o['shared']:5.1f}")
            for name, p in o["per_private_class"].items():
                print(f"   class {name:>14}: acc {p['acc']:5.1f} | errors mostly -> {p['top_error']:>14} "
                      f"({p['top_error_pct']:4.1f}%), to shared {p['to_shared_pct']:4.1f}% | "
                      f"nearest guide {p['nearest_guide']:>14} (ratio {p['nearest_ratio']:.2f})")


if __name__ == "__main__":
    main()
