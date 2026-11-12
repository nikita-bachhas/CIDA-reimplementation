"""Config loading, seeding, checkpointing and logging helpers."""
from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# ------------------------------------------------------------------ config
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(_load_yaml(path.parent / base), cfg)
    return cfg


def load_config(path, overrides=()) -> dict:
    """YAML config with optional `base:` inheritance and `a.b=value` CLI overrides."""
    cfg = _load_yaml(Path(path))
    for ov in overrides or ():
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    cfg.setdefault("out_dir", f"runs/{cfg['name']}")
    return cfg


# ------------------------------------------------------------------ runtime
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rng_state() -> dict:
    s = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_ckpt(obj, path):
    """Atomic save, so a job killed mid-write never leaves a corrupt checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def extract(model, loader, device):
    """Run `model` over a loader yielding (x, y, idx); returns features, labels, idx on CPU."""
    was_training = model.training
    model.eval()
    feats, labels, idxs = [], [], []
    for x, y, i in loader:
        feats.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(torch.as_tensor(y))
        idxs.append(torch.as_tensor(i))
    model.train(was_training)
    return torch.cat(feats), torch.cat(labels), torch.cat(idxs)


class Logger:
    """Prints and appends JSON lines to <dir>/log.jsonl."""

    def __init__(self, out_dir):
        self.path = Path(out_dir) / "log.jsonl"
        self.t0 = time.time()

    def log(self, **kv):
        kv = {"time_min": round((time.time() - self.t0) / 60, 2), **kv}
        line = json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kv.items()})
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


class Timer:
    """Signals when it is time for an intermediate checkpoint."""

    def __init__(self, minutes):
        self.every = minutes * 60 if minutes else None
        self.last = time.time()

    def due(self):
        if self.every and time.time() - self.last > self.every:
            self.last = time.time()
            return True
        return False
