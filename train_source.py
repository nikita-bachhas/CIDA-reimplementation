"""Stage 1: foresighted source-model training (paper §3.1, supp. Algo. 2).

  1 warm-up epoch of plain cross-entropy, then alternate (iter % 2) between
    Ls1 (class separability, softmax over prototype log-densities; trains fs) and
    Ls2 (CE on source + CE on negatives labelled |Cs|; trains fs, gs),
  each with its own Adam optimiser. Prototypes are refitted after every epoch.

Resumes automatically from <out_dir>/source/ckpt.pt. Writes <out_dir>/source/final.pt.

  python train_source.py --config configs/digits_m2u.yaml [--set src_train.epochs=5 ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cida.common import (Logger, Timer, extract, get_device, load_ckpt, load_config, rng_state, save_ckpt,
                         set_rng_state, set_seed)
from cida.data import build_domain
from cida.models import build_classifier, build_feature_extractor
from cida.prototypes import GaussianPrototypes


def fit_prototypes(fs, loader, n_classes, device, pcfg):
    feats, labels, _ = extract(fs, loader, device)
    return GaussianPrototypes.fit(feats, labels, n_classes, pcfg["cov_shrink"], pcfg["cov_eps"]).to(device)


@torch.no_grad()
def evaluate(fs, gs, loader, protos, device, n_shared, scfg):
    fs.eval()
    correct = unknown = n = 0
    for x, y, _ in loader:
        pred = gs(fs(x.to(device))).argmax(1).cpu()
        correct += (pred == y).sum().item()
        unknown += (pred == n_shared).sum().item()
        n += len(y)
    neg = protos.sample_negatives(1024, scfg["k_sigma"], scfg["neg_mode"], scfg.get("chi2_quantile", 0.997))
    neg_det = (gs(neg).argmax(1) == n_shared).float().mean().item()
    fs.train()
    return {"src_acc": 100 * correct / n, "src_pred_unknown": 100 * unknown / n,
            "neg_detected": 100 * neg_det, "neg_accept_rate": protos.last_accept_rate}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. src_train.epochs=5")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    scfg = cfg["src_train"]
    out = Path(cfg["out_dir"]) / "source"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("source_seed", 0))
    device = get_device(cfg.get("device", "auto"))
    log = Logger(out)

    # ---- data (source classes Cs only)
    dom = cfg["src_domain"]
    train_ds = build_domain(cfg, dom, cfg["src_train_split"], train_aug=True, source_only=True)
    feat_ds = build_domain(cfg, dom, cfg["src_train_split"], train_aug=False, source_only=True)
    eval_ds = build_domain(cfg, dom, cfg["src_eval_split"], train_aug=False, source_only=True)
    n_shared = train_ds.n_shared
    bs, nw = cfg["batch_size"], cfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=True,
                              persistent_workers=nw > 0)
    feat_loader = DataLoader(feat_ds, 2 * bs, num_workers=nw)
    eval_loader = DataLoader(eval_ds, 2 * bs, num_workers=nw)

    # ---- model + optimisers (one Adam per loss, supp. Algo. 2 lines 15-17)
    fs = build_feature_extractor(cfg).to(device)
    gs = build_classifier(cfg, n_shared + 1).to(device)  # +1 = OOD / negative class
    lr = cfg["lr"]
    opt_warm = torch.optim.Adam(list(fs.parameters()) + list(gs.parameters()), lr)
    opt_ls1 = torch.optim.Adam(fs.parameters(), lr)
    opt_ls2 = torch.optim.Adam(list(fs.parameters()) + list(gs.parameters()), lr)
    opts = {"warm": opt_warm, "ls1": opt_ls1, "ls2": opt_ls2}

    warm, epochs = scfg["warmup_epochs"], scfg["epochs"]
    start_epoch, it, protos = 0, 0, None
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = load_ckpt(ckpt_path)
        fs.load_state_dict(ck["fs"]), gs.load_state_dict(ck["gs"])
        for k, o in opts.items():
            o.load_state_dict(ck["opts"][k])
        start_epoch, it = ck["epoch"], ck["iter"]
        protos = GaussianPrototypes.from_state_dict(ck["protos"], device) if ck["protos"] else None
        set_rng_state(ck["rng"])
        log.log(event="resumed", epoch=start_epoch, iter=it)

    def checkpoint(epoch_done):
        save_ckpt({"fs": fs.state_dict(), "gs": gs.state_dict(), "opts": {k: o.state_dict() for k, o in opts.items()},
                   "epoch": epoch_done, "iter": it, "protos": protos.state_dict() if protos else None,
                   "rng": rng_state(), "cfg": cfg}, ckpt_path)

    timer = Timer(cfg.get("ckpt_every_minutes"))
    n_neg = max(1, round(scfg.get("neg_ratio", 1.0) * bs))  # Nsrc = Nneg by default
    use_ls1, use_neg = scfg.get("use_ls1", True), scfg.get("use_negatives", True)
    labels_neg = torch.full((n_neg,), n_shared, device=device, dtype=torch.long)

    for epoch in range(start_epoch, warm + epochs):
        if epoch >= warm and protos is None:  # only when warmup_epochs = 0
            protos = fit_prototypes(fs, feat_loader, n_shared, device, scfg)
        fs.train(), gs.train()
        sums = {"warm": 0.0, "ls1": 0.0, "ls2": 0.0}
        counts = {k: 0 for k in sums}
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if epoch < warm:  # Algo. 2 lines 2-5: vanilla CE to initialise the prototypes
                key = "warm"
                loss = F.cross_entropy(gs(fs(x)), y)
            else:
                key = "ls1" if (use_ls1 and it % 2 == 0) else "ls2"
                u = fs(x)
                if key == "ls1":  # Eq. 3
                    loss = F.cross_entropy(protos.log_prob(u), y)
                else:  # Eq. 4
                    loss = F.cross_entropy(gs(u), y)
                    if use_neg:
                        un = protos.sample_negatives(n_neg, scfg["k_sigma"], scfg["neg_mode"],
                                                     scfg.get("chi2_quantile", 0.997))
                        loss = loss + F.cross_entropy(gs(un), labels_neg)
                it += 1
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            opts[key].step()
            sums[key] += loss.item()
            counts[key] += 1
            if timer.due():
                checkpoint(epoch)  # mid-epoch save: a resume restarts this epoch
        protos = fit_prototypes(fs, feat_loader, n_shared, device, scfg)  # end-of-epoch refit
        metrics = evaluate(fs, gs, eval_loader, protos, device, n_shared, scfg)
        log.log(epoch=epoch + 1, iter=it, **{f"loss_{k}": sums[k] / counts[k] for k in sums if counts[k]}, **metrics)
        checkpoint(epoch + 1)

    save_ckpt({"fs": fs.state_dict(), "gs": gs.state_dict(), "protos": protos.state_dict(), "cfg": cfg,
               "n_shared": n_shared}, out / "final.pt")
    log.log(event="done", path=str(out / "final.pt"))


if __name__ == "__main__":
    main()
