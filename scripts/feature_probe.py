"""Can each feature space tell the target-private classes apart from one labelled image?

For a finished task (source model + adaptation seeds), classifies every target sample by its
nearest guide, as the method's pseudo-labelling does, in several feature spaces:
  imagenet      ResNet-50 with ImageNet weights only, avg-pool features (2048-d); ResNet tasks only
  fs backbone   the source model's ResNet-50 after source training (2048-d); ResNet tasks only
  fs            the full source feature extractor (feat_dim)
  ft            each seed's adapted feature extractor (feat_dim)
fs/imagenet use target BatchNorm statistics (AdaBN), as adaptation does.
Guides: shared class = mean feature of that class's source images; private class = the seed's
one-shot target image(s). Reported, averaged over seeds (Euclidean on L2-normalised features):
  priv|priv   private samples, nearest among private guides only: are the new classes separable?
  priv        private samples, nearest among all guides
  shared      shared samples, nearest among all guides
  all         all target samples (= nearest-guide pseudo-label accuracy)
If imagenet >> fs on priv|priv, source training discarded what separates the new classes.

  sbatch slurm/py.sbatch scripts/feature_probe.py runs/office_a2d
"""
import argparse
import copy
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapt import CIDA, adabn  # noqa: E402
from cida.common import get_device, load_ckpt  # noqa: E402
from cida.data import build_domain  # noqa: E402
from cida.models import build_feature_extractor  # noqa: E402


@torch.no_grad()
def feats(model, ds, device, nw):
    model.eval()
    out, ys = [], []
    for x, y, _ in DataLoader(ds, 128, num_workers=nw):
        out.append(F.normalize(model(x.to(device)).float(), dim=1).cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(out), torch.cat(ys)


def nearest_guide(Vt, Yt, Vs, Ys, labelled, ns, n):
    shared_g = torch.stack([Vs[Ys == c].mean(0) for c in range(ns)])
    priv_g = torch.stack([Vt[labelled[c]].mean(0) for c in range(ns, n)])
    lab = torch.tensor([i for c in range(ns, n) for i in labelled[c]])
    keep = torch.ones(len(Yt), dtype=torch.bool)
    keep[lab] = False  # the one-shot images themselves are not scored
    V, Y = Vt[keep], Yt[keep]
    pred = torch.cdist(V, torch.cat([shared_g, priv_g])).argmin(1)
    pred_pp = torch.cdist(V, priv_g).argmin(1) + ns
    p = Y >= ns
    return {"priv|priv": 100 * (pred_pp[p] == Y[p]).float().mean().item(),
            "priv": 100 * (pred[p] == Y[p]).float().mean().item(),
            "shared": 100 * (pred[~p] == Y[~p]).float().mean().item(),
            "all": 100 * (pred == Y).float().mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--runs", default="adapt_1shot_abl-none")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    device = get_device("auto")
    task = Path(args.task_dir)
    runs = sorted(p for p in task.glob(f"{args.runs}_seed*") if (p / "final.pt").exists())
    src = load_ckpt(task / "source" / "final.pt")
    cfg = load_ckpt(runs[0] / "final.pt")["cfg"] if runs else src["cfg"]
    tgt = build_domain(cfg, cfg["tgt_domain"], cfg["tgt_train_split"], train_aug=False)
    srcd = build_domain(cfg, cfg["src_domain"], cfg["src_train_split"], train_aug=False, source_only=True)
    ns, n = tgt.n_shared, tgt.n_shared + tgt.n_private
    tgt_loader = DataLoader(tgt, 128, num_workers=args.num_workers)

    fs = build_feature_extractor(src["cfg"]).to(device)
    fs.load_state_dict(src["fs"])
    spaces = {}
    if hasattr(fs, "backbone"):
        imnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        imnet.fc = torch.nn.Identity()
        spaces["imagenet"] = imnet.to(device)
        spaces["fs backbone"] = copy.deepcopy(fs.backbone)
    spaces["fs"] = fs
    for name, m in spaces.items():  # target BN statistics, as in adaptation; source features use them too
        adabn(m, tgt_loader, device)

    labelled = {r: load_ckpt(r / "final.pt")["labelled"] for r in runs}
    rows = {}
    for name, m in spaces.items():
        Vt, Yt = feats(m, tgt, device, args.num_workers)
        Vs, Ys = feats(m, srcd, device, args.num_workers)
        rows[name] = [nearest_guide(Vt, Yt, Vs, Ys, labelled[r], ns, n) for r in runs]
    ft_rows = []
    for r in runs:
        ck = load_ckpt(r / "final.pt")
        model = CIDA(ck["cfg"], ns, n - ns, ck["cfg"]["adapt"].get("ablation", "none")).to(device)
        model.load_state_dict(ck["model"])
        Vt, Yt = feats(model.ft, tgt, device, args.num_workers)
        Vs, Ys = feats(model.ft, srcd, device, args.num_workers)
        ft_rows.append(nearest_guide(Vt, Yt, Vs, Ys, ck["labelled"], ns, n))
    rows["ft (adapted)"] = ft_rows

    print(f"\n===== {task}: nearest-guide accuracy on the target, mean over {len(runs)} seeds (per seed in brackets)")
    print(f"{'feature space':14}" + "".join(f"{k:>12}" for k in ["priv|priv", "priv", "shared", "all"]) + "   priv|priv per seed")
    for name, rs in rows.items():
        print(f"{name:14}" + "".join(f"{st.mean(r[k] for r in rs):12.1f}" for k in ["priv|priv", "priv", "shared", "all"])
              + "   [" + " ".join(f"{r['priv|priv']:.0f}" for r in rs) + "]")


if __name__ == "__main__":
    main()
