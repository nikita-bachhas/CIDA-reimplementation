"""Safe time-limit checkpoints, stage resuming and random-support presets."""
import json
import signal
from pathlib import Path

import pytest
import torch
from torch import nn

import adapt
import train_source
from cida.common import CheckpointOnSignal, load_ckpt, load_config
from cida.data import sample_labelled


REPO = Path(__file__).resolve().parents[1]
CFG = str(REPO / "configs/smoke.yaml")


def test_installed_signal_handler_requests_checkpoint(monkeypatch):
    # Windows has no USR1; raise_signal(TERM) exercises the same Python handler
    # without using os.kill, which forcibly terminates processes on Windows.
    signum = getattr(signal, "SIGUSR1", signal.SIGTERM)
    monkeypatch.setattr(signal, "SIGUSR1", signum, raising=False)
    previous = signal.getsignal(signum)
    stop = CheckpointOnSignal()
    try:
        signal.raise_signal(signum)
        assert stop.requested
    finally:
        stop.close()
    assert signal.getsignal(signum) == previous


def test_signal_defers_save_until_safe_point():
    stop = CheckpointOnSignal()
    saved = []
    try:
        stop._request(None, None)
        assert saved == []
        with pytest.raises(SystemExit) as exc:
            stop.exit_if_requested(lambda: saved.append("saved"))
        assert exc.value.code == 99
        assert saved == ["saved"]
    finally:
        stop.close()


def interrupt_after(checks):
    class Interrupt(CheckpointOnSignal):
        def __init__(self):
            super().__init__()
            self.checks = 0

        def exit_if_requested(self, checkpoint):
            self.checks += 1
            if self.checks == checks:
                self._request(None, None)
            super().exit_if_requested(checkpoint)

    return Interrupt


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    # Real loaders, optimizers, Gaussian fitting, losses and checkpoint IO; a small
    # feature extractor keeps these restart tests quick on CPU.
    def tiny_features(cfg):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, cfg["model"]["feat_dim"]), nn.ELU())

    monkeypatch.setattr(train_source, "build_feature_extractor", tiny_features)
    monkeypatch.setattr(adapt, "build_feature_extractor", tiny_features)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    overrides = [f"out_dir={tmp_path}", "keep_ckpt=true", "model.feat_dim=8", "model.cls_hidden=4",
                 "src_train.epochs=1", "adapt.epochs=1", "adapt.iters_per_epoch=5",
                 "adapt.proj_pretrain_iters=2", "adapt.seed_offset=0", "adapt.la1_samples=confident"]
    yield tmp_path, overrides
    torch.set_num_threads(old_threads)


def test_source_signal_checkpoint_then_resume(tiny_training, monkeypatch):
    out, overrides = tiny_training
    args = ["--config", CFG, "--set", *overrides]
    monkeypatch.setattr(train_source, "CheckpointOnSignal", interrupt_after(1))
    with pytest.raises(SystemExit) as exc:
        train_source.main(args)
    assert exc.value.code == 99
    ck = load_ckpt(out / "source/ckpt.pt")
    assert ck["opts"]["warm"]["state"]
    assert not (out / "source/final.pt").exists()
    monkeypatch.setattr(train_source, "CheckpointOnSignal", CheckpointOnSignal)
    train_source.main(args)
    assert (out / "source/final.pt").exists()
    rows = [json.loads(line) for line in (out / "source/log.jsonl").read_text().splitlines()]
    assert any(row.get("event") == "resumed" for row in rows)


def test_adaptation_signal_restores_same_random_supports_and_optimizer(tiny_training, monkeypatch):
    out, overrides = tiny_training
    train_source.main(["--config", CFG, "--set", *overrides])
    args = ["--config", CFG, "--seed", "3", "--set", *overrides]
    # Two projection pretraining updates, then Lr1, Lr2 and Lc.
    monkeypatch.setattr(adapt, "CheckpointOnSignal", interrupt_after(5))
    with pytest.raises(SystemExit) as exc:
        adapt.main(args)
    assert exc.value.code == 99
    run = out / "adapt_1shot_abl-none_seed3"
    ck = load_ckpt(run / "ckpt.pt")
    assert ck["state"]["iter"] == 3
    assert ck["opts"]["Lc"]["state"]
    assert not (run / "final_metrics.json").exists()
    monkeypatch.setattr(adapt, "CheckpointOnSignal", CheckpointOnSignal)
    adapt.main(args)
    final = load_ckpt(run / "final.pt")
    assert final["labelled"] == ck["labelled"]
    assert (run / "final_metrics.json").exists()
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert any(row.get("event") == "resumed" and row["iter"] == 3 for row in rows)


@pytest.mark.parametrize("transfer", ["m2u", "u2m", "s2m"])
def test_random_confident_preset(transfer):
    cfg = load_config(REPO / f"configs/digits_{transfer}.yaml", ["preset=digits_random_confident"])
    a = cfg["adapt"]
    assert cfg["out_dir"] == f"runs_digits_random_confident/digits_{transfer}"
    assert cfg["ckpt_every_minutes"] == 5 and cfg["keep_ckpt"]
    assert a["seed_offset"] == 0 and a["shots"] == 1
    assert a["fixed_shots"] is None and a["fewshot_frac"] is None
    assert a["la1_samples"] == "confident" and a["pseudo_label"] == "nearest"
    assert not a["rectify"]["enabled"]
    labels = [c for c in range(10) for _ in range(40)]
    draws = [sample_labelled(labels, list(range(5, 10)), shots=1, seed=s) for s in range(5)]
    assert all(all(len(images) == 1 for images in draw.values()) for draw in draws)
    assert len({tuple(draw[c][0] for c in range(5, 10)) for draw in draws}) == 5
