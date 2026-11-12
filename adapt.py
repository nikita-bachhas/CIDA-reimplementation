"""Stage 2: Class-Incremental DA on the target domain (paper §3.2, supp. Algo. 3-4).

Source-free: only {fs, gs} and the source Gaussian prototypes are loaded.
  ft <- fs;  gs frozen;  fe, fd pre-trained to near-identity on proxy-source samples;
  gt new classifier for C't.
Round-robin (iter % 5) over [Lr1, Lr2, Lc, La1, La2], each with its own Adam:
  Lr1 (fe, fd, gt)  CE on proxy-source samples through [gs.fd.fe | gt.fe]   Eq. 5
  Lr2 (fe, fd)      ||fd(fe(u)) - u||^2                                      Eq. 5
  Lc  (ft, gt)      CE on confident target samples B_t through [gs.fd.ft | gt.ft]  Eq. 7
  La1 (ft)          class separability w.r.t. target prototypes P_t, nearest-guide labels  Eq. 6
  La2 (fe, ft)      ||ft(x) - guide_c||^2 for x in B_t^c                     Eq. 6
Guides: fe(mu_s^c) for shared classes, ft(one-shot sample) for private classes.
At the end of every epoch: pseudo-label all target samples, refit P_t, reselect B_t (top n%).

  python adapt.py --config configs/digits_m2u.yaml --seed 0 [--set adapt.ablation=b ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cida.common import (Logger, Timer, extract, get_device, infinite, load_ckpt, load_config, rng_state,
                         save_ckpt, set_rng_state, set_seed)
from cida.data import Relabeled, build_domain, sample_labelled
from cida.metrics import cida_metrics
from cida.models import build_classifier, build_feature_extractor, build_projection
from cida.prototypes import GaussianPrototypes

LOSSES = ["Lr1", "Lr2", "Lc", "La1", "La2"]  # supp. Algo. 4 line 8
ABLATIONS = {
    "none": LOSSES,
    "a": ["Lr1", "Lc", "La1", "La2"],  # Ours-a: no fe/fd (fixed shared guides), no Lr2
    "b": ["Lc", "La1", "La2"],  # Ours-b: no Lr
    "c": ["Lr1", "Lr2", "Lc", "La1"],  # Ours-c: no La2
}


def run_name(cfg, seed):
    a = cfg["adapt"]
    shot = f"few{int(100 * a['fewshot_frac'])}" if a.get("fewshot_frac") else f"{a.get('shots', 1)}shot"
    tag = a.get("tag") or ""
    return f"adapt_{shot}_abl-{a.get('ablation', 'none')}{('_' + tag) if tag else ''}_seed{seed}"


class CIDA(nn.Module):
    def __init__(self, cfg, n_shared, n_private, ablation):
        super().__init__()
        self.n_shared = n_shared
        self.fs = build_feature_extractor(cfg)  # only used to initialise ft
        self.ft = build_feature_extractor(cfg)
        self.gs = build_classifier(cfg, n_shared + 1)
        self.gt = build_classifier(cfg, n_private)
        no_proj = ablation == "a"
        self.fe = nn.Identity() if no_proj else build_projection(cfg)
        self.fd = nn.Identity() if no_proj else build_projection(cfg)

    def logits_v(self, v):
        """Concatenated logits [gs.fd(v)|_Cs , gt(v)] for V-space features."""
        return torch.cat([self.gs(self.fd(v))[:, : self.n_shared], self.gt(v)], 1)

    def forward(self, x):
        return self.logits_v(self.ft(x))


@torch.no_grad()
def adabn(model, loader, device):
    """Re-estimate BatchNorm running statistics on the target domain (AdaBN).

    ft starts as a copy of fs, so its BN layers hold *source* statistics. Pseudo-labels are
    computed in eval mode, and with source statistics a shared-class USPS digit lies closer
    to any one-shot USPS guide than to its MNIST class mean: on Digits M->U ~75% of shared
    samples were pseudo-labelled as private classes. Target statistics cut this to ~15-20%.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the whole pass
    model.train()
    for x, _, _ in loader:
        model(x.to(device))
    for m in bns:
        m.momentum = 0.1
    model.eval()


def params(*mods):
    return [p for m in mods for p in m.parameters() if p.requires_grad]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0, help="selects the one-shot samples and RNG")
    ap.add_argument("--source_ckpt", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true", help="rerun even if final_metrics.json exists")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    acfg = cfg["adapt"]
    ablation = acfg.get("ablation", "none")
    losses = ABLATIONS[ablation]
    out = Path(cfg["out_dir"]) / run_name(cfg, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final_metrics.json").exists() and not args.force:
        print(f"[skip] already finished: {out / 'final_metrics.json'} (use --force to rerun)", flush=True)
        return
    set_seed(1000 + args.seed)
    device = get_device(cfg.get("device", "auto"))
    log = Logger(out)

    # ---------------------------------------------------------------- data
    dom, bs, nw = cfg["tgt_domain"], cfg["batch_size"], cfg.get("num_workers", 4)
    tgt_aug = build_domain(cfg, dom, cfg["tgt_train_split"], train_aug=True)
    tgt_plain = build_domain(cfg, dom, cfg["tgt_train_split"], train_aug=False)
    tgt_eval = build_domain(cfg, dom, cfg["tgt_eval_split"], train_aug=False)
    n_shared, n_private = tgt_aug.n_shared, tgt_aug.n_private
    n_total = n_shared + n_private
    private_classes = list(range(n_shared, n_total))

    labelled = sample_labelled(tgt_plain.labels, private_classes, acfg.get("shots", 1), acfg.get("fewshot_frac"),
                               seed=args.seed)
    lab_idx = [i for c in private_classes for i in labelled[c]]
    lab_y = [c for c in private_classes for _ in labelled[c]]
    log.log(event="labelled", n=len(lab_idx), per_class={c: len(v) for c, v in labelled.items()})

    # Labelled images (eval transform) for computing private guides; cached if small.
    cache = None
    if len(lab_idx) <= 512:
        cache = torch.stack([tgt_plain[i][0] for i in lab_idx])
    g_per_class = acfg.get("max_guide_samples_per_class", 16)

    def labelled_batch():
        """Up to g_per_class labelled images per private class -> (x, y)."""
        sel = []
        for c in private_classes:
            pos = [k for k, yy in enumerate(lab_y) if yy == c]
            sel += random.sample(pos, min(g_per_class, len(pos)))
        x = cache[sel] if cache is not None else torch.stack([tgt_plain[lab_idx[k]][0] for k in sel])
        return x.to(device), torch.tensor([lab_y[k] for k in sel], device=device)

    tgt_loader = DataLoader(tgt_aug, bs, shuffle=True, drop_last=True, num_workers=nw, pin_memory=True,
                            persistent_workers=nw > 0)
    plain_loader = DataLoader(tgt_plain, 2 * bs, num_workers=nw)
    eval_loader = DataLoader(tgt_eval, 2 * bs, num_workers=nw)
    eval_excl = set(lab_idx) if cfg["tgt_eval_split"] == cfg["tgt_train_split"] else set()

    # --------------------------------------------------------------- model
    src_path = args.source_ckpt or Path(cfg["out_dir"]) / "source" / "final.pt"
    src = load_ckpt(src_path)
    model = CIDA(cfg, n_shared, n_private, ablation).to(device)
    model.fs.load_state_dict(src["fs"])
    model.gs.load_state_dict(src["gs"])
    protos_s = GaussianPrototypes.from_state_dict(src["protos"], device)
    for p in list(model.gs.parameters()) + list(model.fs.parameters()):
        p.requires_grad_(False)  # gs frozen throughout adaptation; fs kept only as reference
    model.fs.eval()
    mu_s = protos_s.means  # [Cs, D]

    lr = acfg.get("lr", cfg["lr"])
    groups = {
        "Lr1": (model.fe, model.fd, model.gt),
        "Lr2": (model.fe, model.fd),
        "Lc": (model.ft, model.gt),
        "La1": (model.ft,),
        "La2": (model.fe, model.ft),
    }
    opts = {k: torch.optim.Adam(params(*groups[k]), lr) for k in losses if params(*groups[k])}
    losses = [k for k in losses if k in opts]
    pre_opts = None if ablation == "a" else {
        "fe": torch.optim.Adam(model.fe.parameters(), lr), "fd": torch.optim.Adam(model.fd.parameters(), lr)}

    state = {"epoch": 0, "iter": 0, "pretrained": False}
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ck = load_ckpt(ckpt_path)
        model.load_state_dict(ck["model"])
        for k, o in opts.items():
            o.load_state_dict(ck["opts"][k])
        state = ck["state"]
        set_rng_state(ck["rng"])
        log.log(event="resumed", **state)
    else:
        model.ft.load_state_dict(model.fs.state_dict())  # Algo. 4 line 2
        if acfg.get("adabn", True):
            adabn(model.ft, plain_loader, device)
            log.log(event="adabn", note="ft BatchNorm statistics re-estimated on the target domain")

    def checkpoint():
        save_ckpt({"model": model.state_dict(), "opts": {k: o.state_dict() for k, o in opts.items()},
                   "state": state, "rng": rng_state(), "labelled": labelled, "cfg": cfg}, ckpt_path)

    # ------------------------------------- fe, fd near-identity pre-training
    proxy_per_class = max(1, math.ceil(bs / n_shared))
    if not state["pretrained"] and pre_opts:
        for i in range(acfg.get("proj_pretrain_iters", 2000)):
            u, _ = protos_s.sample_class(proxy_per_class)
            le = (model.fe(u) - u).pow(2).sum(1).mean()
            ld = (model.fd(u) - u).pow(2).sum(1).mean()
            for o in pre_opts.values():
                o.zero_grad(set_to_none=True)
            (le + ld).backward()
            pre_opts["fe"].step(), pre_opts["fd"].step()
        rel = ((model.fd(model.fe(u)) - u).norm(dim=1) / u.norm(dim=1)).mean().item()
        log.log(event="proj_pretrained", loss_fe=le.item(), loss_fd=ld.item(), rel_recon_err=rel)
        state["pretrained"] = True
    elif ablation == "a":
        state["pretrained"] = True

    # ------------------------------------------------------------- helpers
    def shared_guides():
        return model.fe(mu_s)  # trainable through fe (fixed mu_s under ablation a)

    def private_guides(v_lab, y_lab):
        return torch.stack([v_lab[y_lab == c].mean(0) for c in private_classes])

    def ft_with_guides(x):
        """ft on a batch plus labelled samples in the same forward pass (shared BN batch)."""
        xl, yl = labelled_batch()
        v = model.ft(torch.cat([x, xl]))
        v, vl = v[: len(x)], v[len(x):]
        return v, torch.cat([shared_guides(), private_guides(vl, yl)])

    @torch.no_grad()
    def refresh():
        """Pseudo-label D_t, fit P_t, pick confident sets B_t (Algo. 3, Algo. 4 lines 27-30)."""
        V, Y, _ = extract(model.ft, plain_loader, device)
        model.ft.eval(), model.fe.eval()
        vl = []
        for k in range(0, len(lab_idx), 256):
            xb = cache[k:k + 256] if cache is not None else torch.stack(
                [tgt_plain[i][0] for i in lab_idx[k:k + 256]])
            vl.append(model.ft(xb.to(device)).cpu())
        guides = torch.cat([shared_guides().cpu(), private_guides(torch.cat(vl), torch.tensor(lab_y))])
        model.ft.train(), model.fe.train()
        dist = torch.cdist(V, guides)
        d, k = dist.min(1)
        protos_t = GaussianPrototypes.fit(V, k, n_total, acfg["cov_shrink"], acfg["cov_eps"]).to(device)
        b_idx, b_lab = [], []
        for c in range(n_total):
            members = (k == c).nonzero().squeeze(1)
            if len(members) == 0:
                continue
            nb = max(1, int(acfg["n_percent"] / 100 * len(members)))
            keep = members[d[members].argsort()[:nb]]
            b_idx += keep.tolist()
            b_lab += [c] * len(keep)
        b_idx_t, b_lab_t = torch.tensor(b_idx), torch.tensor(b_lab)
        diag = {
            "pl_acc_all": 100 * (k == Y).float().mean().item(),
            "B_size": len(b_idx),
            "B_precision": 100 * (Y[b_idx_t] == b_lab_t).float().mean().item(),
            "B_precision_priv": 100 * (Y[b_idx_t] == b_lab_t)[b_lab_t >= n_shared].float().mean().item(),
            "B_min_class": min((b_lab_t == c).sum().item() for c in range(n_total)),
        }
        b_ds = Relabeled(tgt_aug, b_idx, b_lab)
        b_loader = DataLoader(b_ds, bs, shuffle=True, drop_last=len(b_ds) >= bs, num_workers=nw)
        return protos_t, infinite(b_loader), diag

    @torch.no_grad()
    def evaluate():
        model.eval()
        preds, ys = [], []
        for x, y, i in eval_loader:
            keep = torch.tensor([j.item() not in eval_excl for j in i])
            if keep.any():
                preds.append(model(x[keep].to(device)).argmax(1).cpu())
                ys.append(y[keep])
        model.train()
        model.fs.eval()
        return cida_metrics(torch.cat(preds), torch.cat(ys), n_shared, n_total)

    # ------------------------------------------------------------ training
    model.train()
    model.fs.eval()
    protos_t, b_iter, diag = refresh()
    if state["epoch"] == 0:
        log.log(epoch=0, **diag, **evaluate())  # before adaptation (source model + fe/fd)
    t_iter = infinite(tgt_loader)
    iters_per_epoch = acfg.get("iters_per_epoch") or len(tgt_loader) * len(losses)
    timer = Timer(cfg.get("ckpt_every_minutes"))

    for epoch in range(state["epoch"], acfg["epochs"]):
        sums = {k: 0.0 for k in losses}
        counts = {k: 0 for k in losses}
        for _ in range(iters_per_epoch):
            key = losses[state["iter"] % len(losses)]
            if key == "Lr1":
                u, y = protos_s.sample_class(proxy_per_class)
                loss = F.cross_entropy(model.logits_v(model.fe(u)), y)
            elif key == "Lr2":
                u, _ = protos_s.sample_class(proxy_per_class)
                loss = (model.fd(model.fe(u)) - u).pow(2).sum(1).mean()
            elif key == "Lc":
                x, y, _ = next(b_iter)
                loss = F.cross_entropy(model(x.to(device)), y.to(device))
            elif key == "La1":
                x, _, _ = next(t_iter)
                v, guides = ft_with_guides(x.to(device))
                k = torch.cdist(v.detach(), guides.detach()).argmin(1)  # nearest-guide pseudo-label
                loss = F.cross_entropy(protos_t.log_prob(v), k)
            else:  # La2
                x, y, _ = next(b_iter)
                v, guides = ft_with_guides(x.to(device))
                loss = (v - guides[y.to(device)]).pow(2).sum(1).mean()
            for o in opts.values():
                o.zero_grad(set_to_none=True)
            loss.backward()
            opts[key].step()
            sums[key] += loss.item()
            counts[key] += 1
            state["iter"] += 1
            if timer.due():
                checkpoint()
        protos_t, b_iter, diag = refresh()
        metrics = evaluate()
        state["epoch"] = epoch + 1
        log.log(epoch=epoch + 1, **{f"loss_{k}": sums[k] / max(counts[k], 1) for k in losses}, **diag, **metrics)
        checkpoint()

    final = {"run": out.name, "config": args.config, "seed": args.seed, "ablation": ablation, **evaluate()}
    with open(out / "final_metrics.json", "w") as f:
        json.dump(final, f, indent=2)
    save_ckpt({"model": model.state_dict(), "labelled": labelled, "cfg": cfg}, out / "final.pt")
    log.log(event="done", **final)


if __name__ == "__main__":
    main()
