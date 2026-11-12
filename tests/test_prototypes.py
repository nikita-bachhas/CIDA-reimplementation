import math

import torch

from cida.prototypes import GaussianPrototypes


def _toy(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = torch.cat([means[c] + torch.randn(n, 2, generator=g) for c in range(3)])
    y = torch.arange(3).repeat_interleave(n)
    return x, y, means


def test_fit_recovers_means():
    x, y, means = _toy()
    p = GaussianPrototypes.fit(x, y, 3, shrink=0.0, eps=1e-6)
    assert torch.allclose(p.means, means, atol=0.2)


def test_log_prob_matches_torch_mvn():
    x, y, _ = _toy()
    p = GaussianPrototypes.fit(x, y, 3, shrink=0.05, eps=1e-4)
    u = torch.randn(7, 2) * 5
    ref = torch.stack(
        [torch.distributions.MultivariateNormal(p.means[c], p.covs[c]).log_prob(u) for c in range(3)], 1
    )
    assert torch.allclose(p.log_prob(u), ref, atol=1e-3)


def test_log_prob_is_differentiable_wrt_input():
    x, y, _ = _toy()
    p = GaussianPrototypes.fit(x, y, 3)
    u = torch.randn(4, 2, requires_grad=True)
    torch.nn.functional.cross_entropy(p.log_prob(u), torch.tensor([0, 1, 2, 0])).backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_paper_threshold_equals_density_at_3_sigma_point():
    """Algo. 1 defines T_c = P_c(mu + 3 sqrt(lambda) nu); check maha there is 9."""
    x, y, _ = _toy()
    p = GaussianPrototypes.fit(x, y, 3)
    evals, evecs = torch.linalg.eigh(p.covs[0].double())
    point = p.means[0] + 3 * evals[-1].sqrt().float() * evecs[:, -1].float()
    assert math.isclose(p.mahalanobis(point[None])[0, 0].item(), 9.0, rel_tol=1e-3)


def test_negatives_lie_outside_3_sigma_of_nearest_class():
    torch.manual_seed(0)
    x, y, _ = _toy()
    p = GaussianPrototypes.fit(x, y, 3)
    neg = p.sample_negatives(256)
    assert neg.shape == (256, 2)
    yhat = p.log_prob(neg).argmax(1)
    maha = p.mahalanobis(neg).gather(1, yhat[:, None]).squeeze(1)
    assert (maha > 9).all()
    assert 0 < p.last_accept_rate < 1


def test_sample_class_shapes_and_state_roundtrip():
    x, y, _ = _toy()
    p = GaussianPrototypes.fit(x, y, 3)
    u, yy = p.sample_class(5)
    assert u.shape == (15, 2) and yy.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    q = GaussianPrototypes.from_state_dict(p.state_dict())
    assert torch.allclose(q.log_prob(u), p.log_prob(u))


def test_empty_and_singleton_classes_fall_back():
    x, y, _ = _toy()
    y = y.clone()
    y[y == 2] = 0
    y[0] = 2  # class 2 has one sample, class 3 none
    p = GaussianPrototypes.fit(x, y, 4)
    assert torch.isfinite(p.log_prob(torch.zeros(1, 2))).all()
