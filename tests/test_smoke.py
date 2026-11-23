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


@pytest.mark.parametrize("select,shared", [("distance", False), ("margin", True)])
def test_adapt_rectify(out_dir, select, shared):
    tag = f"rect{select[0]}{int(shared)}"
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", "adapt.rectify.enabled=true",
                f"adapt.rectify.select={select}", f"adapt.rectify.shared={shared}", f"adapt.tag={tag}"])
    run = out_dir / f"adapt_1shot_abl-none_{tag}_seed0"
    logs = [json.loads(l) for l in open(run / "log.jsonl")]
    ep = [l for l in logs if "rect_w0" in l]
    assert ep and all(0.3 - 1e-6 <= l["rect_w0"] <= 1 for l in ep) and "pl_acc_basic" in ep[0]
    assert (run / "final_metrics.json").exists()


def test_guide_diag_script(out_dir):
    from scripts.guide_diag import VARIANTS, analyse
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed0"
    for stage in ("final", "start"):
        o = analyse(run, torch.device("cpu"), 0, stage=stage)
        assert set(VARIANTS) <= set(o) and 0 <= o["rect"]["priv"] <= 100
        assert (run / f"guide_diag_{stage}.json").exists()
    o = analyse(run, torch.device("cpu"), 0, label_frac=0.5)
    assert min(o["labelled_per_class"]) > 1 and (run / "guide_diag_final_few50.json").exists()
    assert 0 <= o["lp"]["priv"] <= 100 and 0 <= o["km-over"]["priv"] <= 100


def test_confusion_script(out_dir):
    from scripts.confusion import analyse
    adapt.main(["--config", CFG, "--seed", "3", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed3"
    o = analyse(run, torch.device("cpu"), 0)
    final = json.load(open(run / "final_metrics.json"))
    assert abs(o["priv"] - final["priv"]) < 1e-6 and o["priv_given_priv"] >= o["priv"]
    assert (run / "confusion.json").exists()
    from scripts.confusion import print_class_table
    pc = o["per_class"]
    assert len(pc) == len(o["classes"]) and all(0 <= p["acc"] <= 100 for p in pc.values())
    priv = [p for p in pc.values() if p["group"] == "private"]
    assert abs(sum(p["acc"] for p in priv) / len(priv) - final["priv_cls"]) < 1e-3
    print_class_table([o, o])


def test_posthoc_script(out_dir, monkeypatch, capsys):
    from scripts.posthoc import analyse
    adapt.main(["--config", CFG, "--seed", "4", "--set", f"out_dir={out_dir}"])
    run = out_dir / "adapt_1shot_abl-none_seed4"
    o = analyse(run, torch.device("cpu"), 0)
    final = json.load(open(run / "final_metrics.json"))
    assert abs(o["paper"]["all"] - final["all"]) < 1e-6  # the paper's rule reproduces final_metrics
    assert 0 <= o["pi_private"] <= 1 and set(o) >= {"gate", "bias", "both"}
    assert abs(o["paper"]["priv_cls"] - final["priv_cls"]) < 1e-4 and len(o["classes"]) == len(o["paper"]["per_class"])
    # Guide relabelling: nearest guide is what guide_diag calls "basic"; lp runs on this small target.
    from scripts.posthoc import GUIDE_METHODS, guide_labels, print_class_table
    assert all(o[m] is not None and 0 <= o[m]["priv"] <= 100 for m in GUIDE_METHODS)
    V, G = torch.randn(40, 8), torch.randn(6, 8)
    assert torch.equal(guide_labels(V, G, 3, "nearest"), torch.cdist(V, G).argmin(1))
    print_class_table([o, o], ["paper", *GUIDE_METHODS], "lp")
    # Raising the gt offset only moves predictions from shared to private classes.
    sw = o["sweep"]
    assert len(sw) > 10 and all(a["offset"] <= b["offset"] for a, b in zip(sw, sw[1:]))
    assert all(a["priv"] <= b["priv"] + 1e-6 and a["shared"] >= b["shared"] - 1e-6 for a, b in zip(sw, sw[1:]))
    from scripts.posthoc import print_sweep
    print_sweep([o, o])
    import sys as _sys
    from scripts import posthoc
    monkeypatch.setattr(_sys, "argv", ["posthoc.py", str(out_dir), "--num_workers", "0"])
    posthoc.main()
    assert "Class-wise accuracy" in capsys.readouterr().out


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


def test_adapt_fixed_shots(out_dir):
    """adapt.fixed_shots replaces the random one-shot draw; the seed then only sets the training RNG."""
    from cida.common import load_config
    from cida.data import build_domain, fixed_labelled
    spec = "{'5': 3, '6': 0, '7': 1, '8': 2, '9': 4}"
    sets = [f"out_dir={out_dir}", f"adapt.fixed_shots={spec}", "adapt.tag=fixed"]
    for seed in ("0", "1"):
        adapt.main(["--config", CFG, "--seed", seed, "--set", *sets])
    cfg = load_config(CFG, sets)
    ds = build_domain(cfg, cfg["tgt_domain"], cfg["tgt_train_split"], train_aug=False)
    want = fixed_labelled(ds, range(5, 10), cfg["adapt"]["fixed_shots"])
    assert want[5] == [[j for j, lab in enumerate(ds.labels) if lab == 5][3]]
    for seed in (0, 1):
        ck = torch.load(out_dir / f"adapt_1shot_abl-none_fixed_seed{seed}" / "final.pt", weights_only=False)
        assert {int(c): v for c, v in ck["labelled"].items()} == want
    with pytest.raises(ValueError, match="missing"):
        fixed_labelled(ds, range(5, 10), {"5": 0})


@pytest.mark.parametrize("la1", ["confident", "all"])
def test_adapt_lp_pseudo_labels(out_dir, la1):
    """adapt.pseudo_label=lp: label propagation labels the refresh (B_t, P_t, La1), with either La1 source."""
    sets = [f"out_dir={out_dir}", "adapt.pseudo_label=lp", f"adapt.la1_samples={la1}", "adapt.tag=lp"]
    adapt.main(["--config", CFG, "--seed", "0", "--set", *sets])
    run = out_dir / "adapt_1shot_abl-none_lp_seed0"
    rows = [json.loads(line) for line in open(run / "log.jsonl")]
    refresh = [r for r in rows if "B_size" in r]
    assert refresh and all("pl_acc_nearest" in r for r in refresh)
    assert (run / "final.pt").exists()
    with pytest.raises(ValueError, match="pseudo_label"):
        adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", "adapt.pseudo_label=knn",
                    "adapt.tag=bad"])


def test_contact_sheet(tmp_path):
    from PIL import Image
    from scripts.contact_sheet import main
    root = tmp_path / "data" / "office31" / "dslr" / "images"
    for c in range(31):
        (root / f"c{c:02d}").mkdir(parents=True)
        for k in range(3):
            Image.new("RGB", (40, 30), (8 * c, 0, 0)).save(root / f"c{c:02d}" / f"frame_{k:04d}.jpg")
    main([str(Path(CFG).parent / "office_a2d.yaml"), "--data_root", str(tmp_path / "data"), "--out", str(tmp_path / "o")])
    assert len(list((tmp_path / "o" / "dslr").glob("*.png"))) == 10  # one sheet per private class (20-29)


def test_shot_check_script(out_dir, tmp_path, capsys):
    from scripts.shot_check import main
    for seed in ("0", "1"):
        adapt.main(["--config", CFG, "--seed", seed, "--set", f"out_dir={out_dir}"])
    cfg = tmp_path / "fixed.yaml"
    cfg.write_text(f"base: {Path(CFG).as_posix()}\nadapt:\n  fixed_shots: {{'5': 3, '6': 0, '7': 1, '8': 2, '9': 4}}\n")
    main([str(out_dir), "--config", str(cfg), "--num_workers", "0"])
    out = capsys.readouterr().out
    assert "fix avg" in out and "rnd avg" in out and "seed0fx" in out


OFFICE31_CLASSES = [
    "back_pack", "bike", "bike_helmet", "bookcase", "bottle", "calculator", "desk_chair", "desk_lamp",
    "desktop_computer", "file_cabinet", "headphones", "keyboard", "laptop_computer", "letter_tray",
    "mobile_phone", "monitor", "mouse", "mug", "paper_notebook", "pen", "phone", "printer", "projector",
    "punchers", "ring_binder", "ruler", "scissors", "speaker", "stapler", "tape_dispenser", "trash_can"]


@pytest.mark.parametrize("name, n_shared, n_private", [
    ("office_a2d", 10, 10), ("office_a2d_fixedshot", 10, 10),
    ("office20_a2d", 20, 11), ("office20_a2d_fixedshot", 20, 11), ("office20_w2d_fixedshot", 20, 11),
    ("office_a2d_lpref", 10, 10), ("office20_a2d_lpref", 20, 11), ("office20_a2d_ft", 20, 11),
    ("office20_a2w_lpref", 20, 11), ("office20_d2a_lpref", 20, 11), ("office20_d2w_lpref", 20, 11),
    ("office20_w2a_lpref", 20, 11), ("office20_w2d_lpref", 20, 11),
    ("office20_a2d_fixedshot_lp20", 20, 11), ("office20_w2d_fixedshot_lp20", 20, 11),
    ("office20_w2d_srcft100", 20, 11), ("office20_w2d_src180", 20, 11)])
def test_office_configs(tmp_path, name, n_shared, n_private):
    """Class splits of the Office configs, and fixed_shots resolving to one file per private class."""
    from PIL import Image
    from cida.common import load_config
    from cida.data import build_domain, fixed_labelled
    cfg = load_config(Path(CFG).parent / f"{name}.yaml", [f"data_root={tmp_path}"])
    root = tmp_path / cfg["domain_paths"][cfg["tgt_domain"]]
    for c in OFFICE31_CLASSES:
        (root / c).mkdir(parents=True)
        for k in range(1, 26):
            Image.new("RGB", (8, 8)).save(root / c / f"frame_{k:04d}.jpg")
    ds = build_domain(cfg, cfg["tgt_domain"], "all", False)
    assert (ds.n_shared, ds.n_private) == (n_shared, n_private)
    if cfg["adapt"].get("fixed_shots"):
        lab = fixed_labelled(ds, range(n_shared, n_shared + n_private), cfg["adapt"]["fixed_shots"])
        assert all(len(v) == 1 for v in lab.values()) and len(lab) == n_private
