"""Guide rectification (after Liu et al., "Prototype Rectification for Few-Shot Learning", ECCV 2020).

A private guide is the feature of one labelled image, a random point of its class cloud. A target
sample is then on average 2 tr(Sigma) (squared) from its own private guide, against tr(Sigma) for a
mean-type guide such as fe(mu_c), so samples drift to the shared classes. Rectification replaces a
guide by a weighted mean of itself and the target samples it confidently claims:

    w_i  = softmax_i(eps * cos(x_i, g_c))      over {g_c} + members, g_c = the basic guide
    w_0 >= anchor                               (the labelled image keeps at least this weight)
    g'_c = sum_i w_i x_i

eps -> inf recovers the basic guide; eps -> 0 is a plain mean. Repeated `iters` times, each time
re-assigning samples to the current guides (k-means started at, and anchored to, the basic guide).
No target labels are used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def rectify_guides(V: torch.Tensor, base: torch.Tensor, rows, n_percent: float = 20.0, eps: float = 10.0,
                   anchor: float = 0.3, iters: int = 3, select: str = "distance"):
    """Rectify the guides in `rows` with the target features V.

    V      [N, D] unlabelled target features
    base   [C, D] basic guides (shared fe(mu_c), private one-shot features)
    rows   guide indices to rectify (e.g. the private classes)
    select "distance": members nearest to the guide (as B_t); "margin": smallest d1/d2 ratio, which
           drops samples nearly as close to another guide
    Returns (guides [C, D], w0 [C], corr [C, D]) with guides = w0[:, None] * base + corr, so the
    caller can keep the basic guide differentiable: live = w0 * live_base + corr.
    """
    V, base = V.float(), base.float()
    C = base.shape[0]
    w0 = torch.ones(C)
    cur = base.clone()
    rows = list(rows)
    if iters <= 0 or not rows:
        return cur, w0, torch.zeros_like(base)
    for _ in range(iters):
        dist = torch.cdist(V, cur)
        d2, idx = dist.topk(min(2, C), dim=1, largest=False)
        k = idx[:, 0]
        score = d2[:, 0] / d2[:, 1].clamp_min(1e-12) if (select == "margin" and C > 1) else d2[:, 0]
        new = cur.clone()
        for c in rows:
            members = (k == c).nonzero().squeeze(1)
            if len(members) == 0:
                new[c], w0[c] = base[c], 1.0
                continue
            nb = max(1, int(n_percent / 100 * len(members)))
            keep = members[score[members].argsort()[:nb]]
            X = torch.cat([base[c:c + 1], V[keep]])
            w = torch.softmax(eps * F.cosine_similarity(X, base[c:c + 1], dim=1), 0)
            if w[0] < anchor:
                w[1:] = w[1:] / w[1:].sum() * (1 - anchor)
                w[0] = anchor
            new[c], w0[c] = (w[:, None] * X).sum(0), w[0]
        cur = new
    corr = cur - w0[:, None] * base
    return cur, w0, corr
