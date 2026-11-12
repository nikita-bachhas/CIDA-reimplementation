"""Datasets, class splits and labelled-sample selection.

Every dataset returned here yields (image, label, index) with labels remapped so that
shared classes are 0..|Cs|-1 and target-private classes are |Cs|..|Ct|-1.
Samples from classes outside Cs and C't are dropped.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets as tvd
from torchvision import transforms as T

DIGITS = ("mnist", "usps", "svhn")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------- transforms
class ToRGB:
    def __call__(self, im):
        return im.convert("RGB")


class RandomChannelSwap:
    """RGB channel permutation (supp. Fig. 2)."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return x[torch.randperm(3)]
        return x


def digits_transform(name, size=32):
    tf = [T.Resize((size, size))]
    if name != "svhn":
        tf.append(ToRGB())
    return T.Compose(tf + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


def image_transform(train: bool, size=224):
    if not train:
        return T.Compose([T.Resize(256), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    # Augmentations listed in supp. Fig. 2: flip, +-15 deg rotation, crop, colour jitter, channel swap.
    return T.Compose(
        [
            T.Resize(256),
            T.RandomRotation(15),
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(),
            RandomChannelSwap(0.2),
            T.Normalize(MEAN, STD),
        ]
    )


# --------------------------------------------------------------- base datasets
class FakeDomain(Dataset):
    """Tiny synthetic 10-class 'digits' domain for smoke tests (no download)."""

    def __init__(self, domain: str, split: str, n_per_class=40, num_classes=10, size=32):
        g = torch.Generator().manual_seed(0)
        templates = torch.randn(num_classes, 3, size, size, generator=g)
        dseed = sum(map(ord, domain))
        gd = torch.Generator().manual_seed(dseed)
        shift = 0.6 * torch.randn(3, size, size, generator=gd)
        gs = torch.Generator().manual_seed(dseed * 7 + (split == "test"))
        n = n_per_class * num_classes
        self.targets = torch.arange(num_classes).repeat_interleave(n_per_class).tolist()
        noise = 0.8 * torch.randn(n, 3, size, size, generator=gs)
        self.x = templates[self.targets] + shift + noise
        self.classes = [str(i) for i in range(num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _digits(name, split, root):
    tf = digits_transform(name)
    train = split == "train"
    if name == "mnist":
        ds = tvd.MNIST(root, train=train, download=True, transform=tf)
        return ds, ds.targets.tolist()
    if name == "usps":
        ds = tvd.USPS(root, train=train, download=True, transform=tf)
        return ds, list(ds.targets)
    if name == "svhn":
        ds = tvd.SVHN(root, split=split, download=True, transform=tf)
        return ds, ds.labels.tolist()
    raise ValueError(name)


def base_dataset(cfg: dict, domain: str, split: str, train_aug: bool):
    """Returns (dataset yielding (x, y), list of original targets, list of class names)."""
    kind = cfg["dataset"]
    root = Path(cfg["data_root"])
    if kind == "fake":
        ds = FakeDomain(domain, split)
        return ds, ds.targets, ds.classes
    if kind == "digits":
        ds, t = _digits(domain, split, str(root))
        return ds, t, [str(i) for i in range(10)]
    if kind in ("office31", "visda"):
        rel = cfg["domain_paths"][domain]  # e.g. office31/amazon/images or visda/train
        ds = tvd.ImageFolder(str(root / rel), transform=image_transform(train_aug, cfg.get("image_size", 224)))
        return ds, list(ds.targets), list(ds.classes)
    raise ValueError(kind)


# ---------------------------------------------------------------- class splits
def resolve_classes(spec, class_names):
    """Class list given as ints, names, or {range: [a, b]} (inclusive, 0-based)."""
    if isinstance(spec, dict):
        a, b = spec["range"]
        return list(range(a, b + 1))
    return [class_names.index(c) if isinstance(c, str) else int(c) for c in spec]


class Remapped(Dataset):
    """Keeps only shared + private classes and remaps labels to 0..|Ct|-1."""

    def __init__(self, base, targets, shared, private):
        mapping = {c: i for i, c in enumerate(list(shared) + list(private))}
        self.base = base
        self.n_shared, self.n_private = len(shared), len(private)
        self.indices = [i for i, t in enumerate(targets) if t in mapping]
        self.labels = [mapping[targets[i]] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        x, _ = self.base[self.indices[j]]
        return x, self.labels[j], j


class Relabeled(Dataset):
    """A subset of a Remapped dataset with substitute labels (pseudo-labels)."""

    def __init__(self, ds, indices, labels):
        self.ds, self.indices, self.labels = ds, list(indices), list(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        x, _, i = self.ds[self.indices[j]]
        return x, self.labels[j], i


def build_domain(cfg, domain, split, train_aug, source_only=False):
    """Domain dataset restricted to Cs (source_only) or Ct = Cs + C't."""
    base, targets, names = base_dataset(cfg, domain, split, train_aug)
    shared = resolve_classes(cfg["shared"], names)
    private = [] if source_only else resolve_classes(cfg["private"], names)
    return Remapped(base, targets, shared, private)


# ----------------------------------------------------------- labelled samples
def sample_labelled(labels, classes, shots=1, frac=None, seed=0):
    """Pick the one-shot (or few-shot fraction) labelled target-private samples.

    Returns {class: [dataset indices]}; deterministic in `seed`.
    """
    rng = random.Random(seed)
    out = {}
    for c in classes:
        idx = [i for i, lab in enumerate(labels) if lab == c]
        k = max(1, round(frac * len(idx))) if frac else int(shots)
        out[c] = sorted(rng.sample(idx, min(k, len(idx))))
    return out
