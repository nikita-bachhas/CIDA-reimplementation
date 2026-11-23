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

kmeans_guides is the unrestricted version: all members, plain means, optional anchor, optionally
balanced cluster sizes (sinkhorn_assign). overcluster and label_propagation are two further
label-free ways to place the class boundaries (see scripts/guide_diag.py).
"""
from __future__ import annotations

import math

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


@torch.no_grad()
def sinkhorn_assign(D2, tau=0.1, n_iter=100):
    """Hard assignment of N samples to C clusters with (roughly) equal cluster sizes.

    Entropic optimal transport (log-domain Sinkhorn) from uniform sample mass to uniform cluster
    mass with cost D2 / mean(D2); tau is the entropy weight. Returns argmax_c of the transport plan.
    """
    N, C = D2.shape
    logK = -(D2 / D2.mean().clamp_min(1e-12)) / tau
    log_r, log_c = -math.log(N), -math.log(C)
    u, v = torch.zeros(N), torch.zeros(C)
    for _ in range(n_iter):
        u = log_r - torch.logsumexp(logK + v[None], 1)
        v = log_c - torch.logsumexp(logK + u[:, None], 0)
    return (logK + u[:, None] + v[None]).argmax(1)


@torch.no_grad()
def kmeans_guides(V, base, rows, iters=30, anchor=0.0, balanced=False, tau=0.1):
    """Lloyd's k-means over all unlabelled samples, started at the guides; only `rows` move.

    Unlike rectify_guides there is no top-n% member selection and no cosine weighting: each listed
    guide becomes the plain mean of every sample assigned to it, mixed with its starting guide by
    `anchor` (0 = free k-means). balanced=True assigns samples by sinkhorn_assign (equal cluster
    sizes, i.e. a uniform class prior) instead of nearest guide. Cluster c keeps the label of the
    guide it started from. Stops early once the guides no longer move; an empty cluster keeps its
    current guide. Label-free.
    """
    V, base = V.float(), base.float()
    cur = base.clone()
    rows = list(rows)
    for _ in range(iters if rows else 0):
        D2 = torch.cdist(V, cur).pow(2)
        k = sinkhorn_assign(D2, tau) if balanced else D2.argmin(1)
        new = cur.clone()
        for c in rows:
            m = k == c
            if m.any():
                new[c] = anchor * base[c] + (1 - anchor) * V[m].mean(0)
        done = torch.allclose(new, cur)
        cur = new
        if done:
            break
    return cur


@torch.no_grad()
def overcluster(V, init_labels, n_classes, per_class=3, iters=30, seed=0):
    """K-means with per_class * n_classes free clusters; each cluster takes the majority label.

    Centres start at random samples (seeded). After convergence each cluster is labelled by the
    majority of `init_labels` among its members (e.g. the labels from kmeans_guides), so a class
    may own several clusters and its boundary can bend. Returns per-sample labels. Label-free.
    """
    V = V.float()
    g = torch.Generator().manual_seed(seed)
    K = min(len(V), per_class * n_classes)
    cur = V[torch.randperm(len(V), generator=g)[:K]].clone()
    for _ in range(iters):
        k = torch.cdist(V, cur).argmin(1)
        new = torch.stack([V[k == j].mean(0) if (k == j).any() else cur[j] for j in range(K)])
        done = torch.allclose(new, cur)
        cur = new
        if done:
            break
    k = torch.cdist(V, cur).argmin(1)
    lab = torch.stack([torch.bincount(init_labels[k == j], minlength=n_classes).argmax() if (k == j).any()
                       else torch.tensor(0) for j in range(K)])
    return lab[k]


@torch.no_grad()
def label_propagation(V, G, knn=10, alpha=0.99, iters=50, balance=True):
    """Label spreading (Zhou et al., NeurIPS 2004) from the guides over a kNN graph of the samples.

    Nodes are the samples plus one node per guide (guide c carries label c). Edges: cosine kNN,
    symmetrised, weights = cosine similarity clipped at 0, normalised D^-1/2 W D^-1/2. Iterates
    F <- alpha S F + (1 - alpha) Y0. balance=True divides each class column by its total mass
    (a uniform class prior). Returns (labels, confidence) for the samples. Label-free. Dense N x N,
    meant for Office-sized targets.
    """
    X = F.normalize(torch.cat([V.float(), G.float()]), dim=1)
    N, C = len(X), len(G)
    sim = X @ X.T
    sim.fill_diagonal_(-1)
    val, idx = sim.topk(min(knn, N - 1), dim=1)
    W = torch.zeros(N, N).scatter_(1, idx, val.clamp_min(0))
    W = torch.maximum(W, W.T)
    dinv = W.sum(1).clamp_min(1e-12).rsqrt()
    S = dinv[:, None] * W * dinv[None]
    Y0 = torch.zeros(N, C)
    Y0[len(V) + torch.arange(C), torch.arange(C)] = 1
    Fm = Y0.clone()
    for _ in range(iters):
        Fm = alpha * (S @ Fm) + (1 - alpha) * Y0
    Fm = Fm[:len(V)]
    if balance:
        Fm = Fm / Fm.sum(0, keepdim=True).clamp_min(1e-12)
    P = Fm / Fm.sum(1, keepdim=True).clamp_min(1e-12)
    conf, lab = P.max(1)
    return lab, conf
