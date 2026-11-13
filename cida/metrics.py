"""CIDA evaluation: ALL (over Ct) and PRIV (over C't) accuracy, per-sample and class-averaged.

The paper does not say which averaging it uses, so both are reported.
"""
import torch


def cida_metrics(pred: torch.Tensor, y: torch.Tensor, n_shared: int, n_total: int) -> dict:
    pred, y = pred.cpu(), y.cpu()
    correct = (pred == y).float()
    priv = y >= n_shared
    out = {
        "all": correct.mean().item(),
        "priv": correct[priv].mean().item() if priv.any() else float("nan"),
        "shared": correct[~priv].mean().item() if (~priv).any() else float("nan"),
    }
    # Where the errors go: shared samples predicted as a private class, and vice versa.
    out["shared_as_priv"] = (pred[~priv] >= n_shared).float().mean().item() if (~priv).any() else float("nan")
    out["priv_as_shared"] = (pred[priv] < n_shared).float().mean().item() if priv.any() else float("nan")
    per_class = torch.tensor([correct[y == c].mean().item() if (y == c).any() else float("nan") for c in range(n_total)])
    out["all_cls"] = per_class.nanmean().item()
    out["priv_cls"] = per_class[n_shared:].nanmean().item()
    out["shared_cls"] = per_class[:n_shared].nanmean().item()
    return {k: 100 * v for k, v in out.items()}
