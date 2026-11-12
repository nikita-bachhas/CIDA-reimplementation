"""End-to-end run on synthetic data: source training, resume, adaptation, ablations."""
import json
from pathlib import Path

import pytest

import adapt
import train_source

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "smoke"
    train_source.main(["--config", CFG, "--set", f"out_dir={out}"])
    return out


def test_source_outputs(out_dir):
    assert (out_dir / "source" / "final.pt").exists()
    logs = [json.loads(l) for l in open(out_dir / "source" / "log.jsonl")]
    epochs = [l for l in logs if "src_acc" in l]
    assert len(epochs) == 3  # 1 warm-up + 2 epochs
    assert epochs[-1]["src_acc"] > 50


def test_source_resume_is_noop_when_finished(out_dir):
    train_source.main(["--config", CFG, "--set", f"out_dir={out_dir}"])
    logs = [json.loads(l) for l in open(out_dir / "source" / "log.jsonl")]
    assert any(l.get("event") == "resumed" for l in logs)


@pytest.mark.parametrize("ablation", ["none", "a", "b", "c"])
def test_adapt(out_dir, ablation):
    adapt.main(["--config", CFG, "--seed", "0", "--set", f"out_dir={out_dir}", f"adapt.ablation={ablation}"])
    res = json.load(open(out_dir / f"adapt_1shot_abl-{ablation}_seed0" / "final_metrics.json"))
    for k in ("all", "priv", "shared", "all_cls", "priv_cls"):
        assert 0 <= res[k] <= 100


def test_adapt_fewshot(out_dir):
    adapt.main(["--config", CFG, "--seed", "1", "--set", f"out_dir={out_dir}", "adapt.fewshot_frac=0.05"])
    assert (out_dir / "adapt_few5_abl-none_seed1" / "final_metrics.json").exists()
