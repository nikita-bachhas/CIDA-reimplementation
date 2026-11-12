"""Networks from supp. Table 2.

Office / VisDA: fs = ResNet-50 (to avg-pool, 2048) -> FC1024-ELU-BN -> FC256-ELU-BN
                gs = 256 -> 64 (ELU) -> |Cs|+1      gt = 256 -> 64 (ELU) -> |C't|
                fe, fd = 3 x FC256 (ELU)
Digits:         fs = modified LeNet: conv5(64) conv5(64) conv3/s2(128) conv3/s2(128)
                     -> flatten 3200 -> FC100, LeakyReLU
                gs = 100 -> 32 -> |Cs|+1   gt = 100 -> 32 -> |C't|   fe, fd = 2 x FC100
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm


def act(name):
    return {"elu": nn.ELU(), "lrelu": nn.LeakyReLU(0.2), "relu": nn.ReLU()}[name]


def mlp(dims, activation, final_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_act:
            layers.append(act(activation))
    return nn.Sequential(*layers)


class LeNetFeat(nn.Module):
    def __init__(self, feat_dim=100, bn=True):
        super().__init__()

        def block(cin, cout, k, s):
            return [nn.Conv2d(cin, cout, k, s)] + ([nn.BatchNorm2d(cout)] if bn else []) + [nn.LeakyReLU(0.2)]

        self.conv = nn.Sequential(
            *block(3, 64, 5, 1),  # 32 -> 28
            *block(64, 64, 5, 1),  # 28 -> 24
            *block(64, 128, 3, 2),  # 24 -> 11
            *block(128, 128, 3, 2),  # 11 -> 5
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(5 * 5 * 128, feat_dim), nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.fc(self.conv(x))


class ResNetFeat(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(2048, 1024), nn.ELU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, feat_dim), nn.ELU(), nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_feature_extractor(cfg):
    m = cfg["model"]
    if m["backbone"] == "lenet":
        return LeNetFeat(m["feat_dim"], bn=m.get("bn", True))
    if m["backbone"] == "resnet50":
        return ResNetFeat(m["feat_dim"], pretrained=m.get("pretrained", True))
    raise ValueError(m["backbone"])


def build_classifier(cfg, n_out):
    m = cfg["model"]
    return mlp([m["feat_dim"], m["cls_hidden"], n_out], m["act"])


def build_projection(cfg):
    """fe / fd. The paper's final layer has an activation; that cannot represent the
    identity on BN outputs (negative values), so `proj_final_act` defaults to False."""
    m = cfg["model"]
    d = m["feat_dim"]
    return mlp([d] * (m["proj_layers"] + 1), m["act"], final_act=m.get("proj_final_act", False))
