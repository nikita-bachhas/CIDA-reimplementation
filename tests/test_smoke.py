"""End-to-end run on synthetic data: source training, resume, adaptation, ablations."""
import json
from pathlib import Path

import pytest
import torch

import adapt
import train_source

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "smoke"
    train_source.main(["--config", CFG, "--set", f"out_dir={out}", "keep_ckpt=true"])  # kept for the resume test
    return out


def test_source_outputs(out_dir):
    assert (out_dir / "source" / "final.pt").exists()
    logs = [json.loads(l) for l in open(out_dir / "source" / "log.jsonl")]
    epochs = [l for l in logs if "src_acc" in l]
    assert len(epochs) == 3  # 1 warm-up + 2 epochs
    assert epochs[-1]["src_acc"] > 50


def test_finished_source_is_skipped(out_dir, capsys):
    train_source.main(["--config", CFG, "--set", f"out_dir={out_dir}"])
    assert "[skip]" in capsys.readouterr().out


def test_source_resumes_from_checkpoint(out_dir):
    train_source.main(["--config", CFG, "--force", "--set", f"out_dir={out_dir}"])
    logs = [json.loads(l) for l in open(out_dir / "source" / "log.jsonl")]
    assert any(l.get("event") == "resumed" for l in logs)


def test_finished_adaptation_is_skipped(out_dir, capsys):
    args = ["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}"]
    adapt.main(args)
    capsys.readouterr()
    adapt.main(args)
    assert "[skip]" in capsys.readouterr().out


@pytest.mark.parametrize("ablation", ["none", "a", "b", "c"])
def test_adapt(out_dir, ablation):
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", f"adapt.ablation={ablation}"])
    res = json.load(open(out_dir / f"adapt_1shot_abl-{ablation}_seed0" / "final_metrics.json"))
    for k in ("all", "priv", "shared", "all_cls", "priv_cls"):
        assert 0 <= res[k] <= 100


def test_adapt_fewshot(out_dir):
    adapt.main(["--config", CFG, "--seed", "1", "--set", f"out_dir={out_dir}", "adapt.fewshot_frac=0.05"])
    assert (out_dir / "adapt_few5_abl-none_seed1" / "final_metrics.json").exists()


def test_adapt_la1_variants(out_dir):
    adapt.main(["--config", CFG, "--seed", "2", "--set", f"out_dir={out_dir}", "adapt.la1_samples=all",
                "adapt.la1_temp=100", "adapt.tag=la1var"])
    assert (out_dir / "adapt_1shot_abl-none_la1var_seed2" / "final_metrics.json").exists()


def test_adapt_share_peak_stop(out_dir):
    """adapt.stop=share_peak keeps the first epoch with the highest predicted private share, stops `patience`
    epochs later, and a resumed job neither trains further nor changes the result."""
    sets = ["--set", f"out_dir={out_dir}", "keep_ckpt=true", "adapt.epochs=6", "adapt.stop=share_peak",
            "adapt.stop_patience=1", "adapt.tag=sharestop"]
    adapt.main(["--config", CFG, "--seed", "0", *sets])
    run = out_dir / "adapt_1shot_abl-none_sharestop_seed0"
    res = json.load(open(run / "final_metrics.json"))
    ep = {l["epoch"]: l for l in map(json.loads, open(run / "log.jsonl")) if "epoch" in l and l["epoch"] > 0}
    shares = [ep[e]["pred_priv_share"] for e in sorted(ep)]
    assert sorted(ep) == list(range(1, res["epochs_trained"] + 1))
    assert res["stop_epoch"] == 1 + shares.index(max(shares))  # first epoch at the maximum: strict rises only
    assert res["epochs_trained"] == min(6, res["stop_epoch"] + 1)
    assert abs(res["all"] - ep[res["stop_epoch"]]["all"]) < 1e-3  # final.pt is the stop-epoch model
    assert abs(res["pred_priv_share"] - res["stop_share"]) < 1e-3
    assert [f.name for f in run.glob("best_e*.pt")] == [f"best_e{res['stop_epoch']}.pt"]
    adapt.main(["--config", CFG, "--seed", "0", "--force", *sets])
    logs = [json.loads(l) for l in open(run / "log.jsonl")]
    after = logs[max(i for i, l in enumerate(logs) if l.get("event") == "resumed"):]
    assert not any("epoch" in l and "event" not in l for l in after)  # the "resumed" line carries state["epoch"]
    res2 = json.load(open(run / "final_metrics.json"))
    assert res2["stop_epoch"] == res["stop_epoch"] and abs(res2["all"] - res["all"]) < 1e-6


def test_adapt_share_peak_cleans_up(out_dir):
    adapt.main(["--config", CFG, "--seed", "1", "--set", f"out_dir={out_dir}", "adapt.stop=share_peak",
                "adapt.tag=sharestop"])
    run = out_dir / "adapt_1shot_abl-none_sharestop_seed1"
    res = json.load(open(run / "final_metrics.json"))
    assert 1 <= res["stop_epoch"] <= 2 and 0 <= res["pred_priv_share"] <= 100
    assert (run / "final.pt").exists() and not list(run.glob("best_e*.pt")) and not (run / "ckpt.pt").exists()


def test_adapt_balanced_b_sampler(out_dir):
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", "adapt.b_sampler=balanced",
                "adapt.tag=bbal"])
    res = json.load(open(out_dir / "adapt_1shot_abl-none_bbal_seed0" / "final_metrics.json"))
    assert 0 <= res["all"] <= 100 and "stop_epoch" not in res


@pytest.mark.parametrize("key", ["stop", "b_sampler"])
def test_adapt_rejects_unknown_options(out_dir, key):
    with pytest.raises(ValueError, match=f"adapt.{key}"):
        adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", f"adapt.{key}=bogus",
                    f"adapt.tag=bad{key}"])


def test_confusion_script(out_dir):
    from scripts.confusion import analyse
    adapt.main(["--config", CFG, "--seed", "3", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed3"
    o = analyse(run, torch.device("cpu"), 0)
    final = json.load(open(run / "final_metrics.json"))
    assert abs(o["priv"] - final["priv"]) < 1e-6 and o["priv_given_priv"] >= o["priv"]
    assert (run / "confusion.json").exists()


def test_posthoc_script(out_dir):
    from scripts.posthoc import analyse
    adapt.main(["--config", CFG, "--seed", "4", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed4"
    o = analyse(run, torch.device("cpu"), 0)
    final = json.load(open(run / "final_metrics.json"))
    assert abs(o["paper"]["all"] - final["all"]) < 1e-6  # the paper's rule reproduces final_metrics
    assert 0 <= o["pi_private"] <= 1 and set(o) >= {"gate", "bias", "both"}


def test_feature_probe_script(out_dir, monkeypatch, capsys):
    import sys as _sys
    from scripts import feature_probe
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}"])
    monkeypatch.setattr(_sys, "argv", ["feature_probe.py", str(out_dir), "--num_workers", "0"])
    feature_probe.main()
    out = capsys.readouterr().out
    assert "priv|priv" in out and "ft (adapted)" in out


def test_resnet_head_matches_paper():
    from cida.models import ResNetFeat
    head = ResNetFeat(256, pretrained=False, n_fc_out=2).head  # supp. Table 2A
    kinds = [type(m).__name__ for m in head]
    assert kinds == ["Linear", "ELU", "BatchNorm1d", "Linear", "ELU", "Linear", "ELU", "BatchNorm1d"]
    old = ResNetFeat(256, pretrained=False, n_fc_out=1).head  # runs saved before the fix still load
    assert [type(m).__name__ for m in old] == ["Linear", "ELU", "BatchNorm1d", "Linear", "ELU", "BatchNorm1d"]


def test_feature_probe_source_only(tmp_path, monkeypatch, capsys):
    import sys as _sys
    from scripts import feature_probe
    out = tmp_path / "src_only"
    train_source.main(["--config", CFG, "--set", f"out_dir={out}"])
    monkeypatch.setattr(_sys, "argv", ["feature_probe.py", str(out), "--num_workers", "0"])
    feature_probe.main()
    text = capsys.readouterr().out
    assert "seeds 0-4" in text and "ft (adapted)" not in text


def test_backbone_param_groups():
    from cida.models import ResNetFeat, freeze_backbone, param_groups
    fs = ResNetFeat(256, pretrained=False)
    g = param_groups([fs], 1e-4, 0.1)
    assert [x["lr"] for x in g] == [1e-4, 1e-5]
    assert sum(len(x["params"]) for x in g) == len(list(fs.parameters()))
    freeze_backbone(fs)
    g = param_groups([fs], 1e-4, 0.0)
    assert len(g) == 1 and len(g[0]["params"]) == len(list(fs.head.parameters()))


def test_vanilla_preset():
    from cida.common import load_config
    cfgs = Path(CFG).parent
    v = load_config(cfgs / "office_a2d.yaml", ["preset=vanilla"])
    assert v["out_dir"] == "runs_vanilla/office_a2d" and v["src_train"]["epochs"] == 50
    assert v["src_train"]["backbone_lr_mult"] == 1.0 and v["adapt"]["backbone_lr_mult"] == 1.0
    assert v["adapt"]["la1_samples"] == "all" and v["adapt"]["adabn"] is False
    assert v["model"]["proj_residual"] is False and v["model"]["proj_final_act"] is True
    assert v["model"]["resnet_fc_out"] == 2  # paper head, not one of our fixes
    d = load_config(cfgs / "digits_m2u.yaml", ["preset=vanilla", "adapt.epochs=3"])
    assert d["out_dir"] == "runs_vanilla/digits_m2u" and d["adapt"]["epochs"] == 3  # overrides beat the preset
    assert load_config(cfgs / "office_a2d.yaml")["out_dir"] == "runs/office_a2d"  # default unchanged


def test_adapt_vanilla(out_dir):
    adapt.main(["--config", CFG, "--seed", "3", "--set", "preset=vanilla", f"out_dir={out_dir}", "adapt.tag=vanilla"])
    assert (out_dir / "adapt_1shot_abl-none_vanilla_seed3" / "final_metrics.json").exists()


def test_finished_run_deletes_resume_checkpoint(out_dir):
    adapt.main(["--config", CFG, "--seed", "4", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed4"
    assert (run / "final.pt").exists() and not (run / "ckpt.pt").exists()


def test_adabn_skips_a_final_batch_of_one():
    """A target set of size 2*bs*k + 1 leaves a last batch of one image, which BatchNorm cannot train on
    (Office-31 Amazon with 31 classes: 2817 images, batches of 64)."""
    from adapt import adabn

    x = torch.randn(5, 4)
    ds = torch.utils.data.TensorDataset(x, torch.zeros(5), torch.arange(5))
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    adabn(model, torch.utils.data.DataLoader(ds, batch_size=2), torch.device("cpu"))  # batches 2, 2, 1
    assert not model.training
    assert model[1].num_batches_tracked.item() == 2
