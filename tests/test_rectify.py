import torch

from cida.rectify import rectify_guides


def _toy(seed=0, n=400, d=20, sep=8.0):
    """Shared class 0 with a mean-type guide, private class 1 with a one-shot guide at its edge."""
    g = torch.Generator().manual_seed(seed)
    m0, m1 = torch.zeros(d), torch.zeros(d)
    m1[0] = sep
    x0 = m0 + torch.randn(n, d, generator=g)
    x1 = m1 + torch.randn(n, d, generator=g)
    j = (x1 - m1).norm(dim=1).argmax()  # least typical private sample is the labelled one
    shot = x1[j]
    x1 = torch.cat([x1[:j], x1[j + 1:]])  # the labelled image is not among the unlabelled samples
    V = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(len(x0)), torch.ones(len(x1))]).long()
    return V, y, torch.stack([m0, shot]), m1


def test_toy_reproduces_the_private_to_shared_drift():
    V, y, base, _ = _toy()
    pred = torch.cdist(V, base).argmin(1)
    assert (pred[y == 1] == 0).float().mean() > 0.1  # some private samples are nearer the shared mean


def test_rectified_guide_moves_towards_class_mean_and_fixes_labels():
    V, y, base, m1 = _toy()
    g, w0, corr = rectify_guides(V, base, rows=[1], n_percent=50, eps=1.0, anchor=0.2, iters=5)
    assert (g[1] - m1).norm() < 0.5 * (base[1] - m1).norm()
    acc = lambda G: (torch.cdist(V, G).argmin(1) == y)[y == 1].float().mean()  # noqa: E731
    assert acc(g) > acc(base) + 0.1
    assert torch.equal(g[0], base[0])  # rows not listed are untouched
    assert torch.allclose(g, w0[:, None] * base + corr, atol=1e-5)
    assert w0[1] >= 0.2 and w0[0] == 1


def test_large_eps_or_no_iters_returns_basic_guides():
    V, _, base, _ = _toy()
    g, w0, _ = rectify_guides(V, base, rows=[1], eps=1e4, anchor=0.0)
    assert torch.allclose(g, base, atol=1e-4) and torch.isclose(w0[1], torch.tensor(1.0))
    g, w0, corr = rectify_guides(V, base, rows=[1], iters=0)
    assert torch.equal(g, base) and torch.equal(corr, torch.zeros_like(base))


def test_margin_selection_runs():
    V, y, base, m1 = _toy(seed=1)
    g, _, _ = rectify_guides(V, base, rows=[1], select="margin", eps=1.0, anchor=0.2)
    assert (g[1] - m1).norm() < (base[1] - m1).norm()
