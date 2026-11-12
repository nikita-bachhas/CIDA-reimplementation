"""Gaussian Prototypes (paper §3.1a, supp. Algo. 1).

A set of class-conditional multivariate Gaussians N(mu_c, Sigma_c) plus one global
Gaussian N(mu, Sigma) over all features. Used for:
  * the class-separability losses Ls1 / La1 (softmax over class log-densities),
  * proxy-source sampling during adaptation (Lr1, Lr2),
  * negative (out-of-distribution) sampling for Ls2.

All statistics are treated as constants (no gradient); `log_prob` is differentiable
with respect to its input features only.
"""
from __future__ import annotations

import math

import torch


def _regularise(cov: torch.Tensor, shrink: float, eps: float) -> torch.Tensor:
    """Shrink towards a scaled identity so the covariance is well conditioned.

    With 256-d features and ~100 images per Office class, the empirical covariance
    is rank deficient, so some regularisation is unavoidable (not specified in the paper).
    """
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
    scale = torch.clamp(scale, min=1e-8)[..., None, None]
    return (1.0 - shrink) * cov + (shrink + eps) * scale * eye


def _empirical_cov(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(0, keepdim=True)
    return xc.T @ xc / max(len(x) - 1, 1)


class GaussianPrototypes:
    def __init__(self, means, covs, gmean, gcov):
        self.means = means.float()  # [C, D]
        self.covs = covs.float()  # [C, D, D]
        self.gmean = gmean.float()  # [D]
        self.gcov = gcov.float()  # [D, D]
        self.last_accept_rate = float("nan")
        self._prepare()

    # ------------------------------------------------------------------ setup
    def _prepare(self):
        # Decompose in float64 on the CPU (MPS has no float64), then move back.
        dev = self.means.device
        chol = torch.linalg.cholesky(self.covs.cpu().double())
        eye = torch.eye(self.dim, dtype=torch.float64).expand_as(chol)
        self.chol = chol.float().to(dev)
        self.linv = torch.linalg.solve_triangular(chol, eye, upper=False).float().to(dev)
        self.logdet = (2 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)).float().to(dev)
        self.gchol = torch.linalg.cholesky(self.gcov.cpu().double()).float().to(dev)

    @property
    def num_classes(self) -> int:
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @classmethod
    @torch.no_grad()
    def fit(cls, feats, labels, num_classes, shrink=0.1, eps=1e-4):
        """Fit per-class and global Gaussians. Classes with < 2 samples fall back to
        the global covariance (and the global mean if empty)."""
        feats = feats.double()
        labels = labels.long()
        gmean = feats.mean(0)
        gcov = _regularise(_empirical_cov(feats), shrink, eps)
        means, covs = [], []
        for c in range(num_classes):
            fc = feats[labels == c]
            if len(fc) >= 2:
                means.append(fc.mean(0))
                covs.append(_regularise(_empirical_cov(fc), shrink, eps))
            else:
                means.append(fc[0] if len(fc) == 1 else gmean)
                covs.append(gcov)
        return cls(torch.stack(means), torch.stack(covs), gmean, gcov)

    # ---------------------------------------------------------------- density
    def mahalanobis(self, u: torch.Tensor) -> torch.Tensor:
        """Squared Mahalanobis distance of each row of u to each class: [N, C]."""
        diff = u[:, None, :] - self.means[None]  # [N, C, D]
        z = torch.einsum("ced,ncd->nce", self.linv, diff)
        return z.pow(2).sum(-1)

    def log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """Class log-densities log P^c(u): [N, C]. Used as logits in Ls1 / La1."""
        return -0.5 * (self.mahalanobis(u) + self.logdet[None] + self.dim * math.log(2 * math.pi))

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_class(self, n_per_class: int):
        """Proxy-source samples: n_per_class from every class prototype."""
        c, d = self.num_classes, self.dim
        eps = torch.randn(c, n_per_class, d, device=self.means.device)
        u = self.means[:, None, :] + torch.einsum("cde,cne->cnd", self.chol, eps)
        y = torch.arange(c, device=u.device).repeat_interleave(n_per_class)
        return u.reshape(c * n_per_class, d), y

    @torch.no_grad()
    def sample_global(self, n: int):
        eps = torch.randn(n, self.dim, device=self.gmean.device)
        return self.gmean[None] + eps @ self.gchol.T

    def maha_threshold(self, k_sigma=3.0, mode="paper", quantile=0.997) -> float:
        """Squared-Mahalanobis threshold that defines 'outside the class prototype'.

        paper: Algo. 1 uses the density at mu_c + k*sqrt(lambda_max)*nu_max. For a
               Gaussian that point has squared Mahalanobis distance exactly k^2, so
               P_c(u) < T_c  <=>  maha_c(u) > k^2.
        chi2:  dimension-aware alternative (chi-square quantile with D dof). In high
               dimensions almost every sample is > 3 'sigma' away under the paper rule,
               so this is offered as a documented deviation.
        """
        if mode == "paper":
            return float(k_sigma) ** 2
        if mode == "chi2":
            from scipy.stats import chi2

            return float(chi2.ppf(quantile, self.dim))
        raise ValueError(mode)

    @torch.no_grad()
    def sample_negatives(self, n: int, k_sigma=3.0, mode="paper", quantile=0.997, max_rounds=50):
        """Negative features u_n ~ P_s (global) that fall outside the k-sigma region of
        their most likely class prototype (supp. Algo. 1, lines 7-14)."""
        thr = self.maha_threshold(k_sigma, mode, quantile)
        kept, drawn, accepted = [], 0, 0
        for _ in range(max_rounds):
            cand = self.sample_global(4 * n)
            drawn += len(cand)
            yhat = self.log_prob(cand).argmax(1)
            maha = self.mahalanobis(cand).gather(1, yhat[:, None]).squeeze(1)
            ok = cand[maha > thr]
            accepted += len(ok)
            kept.append(ok)
            if sum(len(k) for k in kept) >= n:
                break
        out = torch.cat(kept)
        self.last_accept_rate = accepted / max(drawn, 1)
        if len(out) < n:  # rare: top up with the most outlying candidates
            cand = self.sample_global(4 * n)
            yhat = self.log_prob(cand).argmax(1)
            maha = self.mahalanobis(cand).gather(1, yhat[:, None]).squeeze(1)
            out = torch.cat([out, cand[maha.argsort(descending=True)[: n - len(out)]]])
        return out[:n]

    # ------------------------------------------------------------- persistence
    def to(self, device):
        return GaussianPrototypes(
            self.means.to(device), self.covs.to(device), self.gmean.to(device), self.gcov.to(device)
        )

    def state_dict(self):
        return {k: getattr(self, k).cpu() for k in ("means", "covs", "gmean", "gcov")}

    @classmethod
    def from_state_dict(cls, sd, device="cpu"):
        return cls(sd["means"], sd["covs"], sd["gmean"], sd["gcov"]).to(device)
