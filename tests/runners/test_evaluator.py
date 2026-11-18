"""Tests for ``mon.runners.evaluator`` and ``mon evaluate`` (CPU, tiny folders; no downloads)."""

import csv
from pathlib import Path as StdPath

import pytest
import torch
import yaml

from mon.core.path import datasets_dir
from mon.data.io import write_image
from mon.metrics import psnr, ssim
from mon.runners.cli import main
from mon.runners.evaluator import EvaluateConfig, _gt_mean, evaluate

FR = ["psnr", "ssim"]


def _write(root: StdPath, images: dict[str, torch.Tensor]) -> StdPath:
    for name, img in images.items():
        write_image((img * 255).round().to(torch.uint8), root / name)
    return root


@pytest.fixture
def pair(tmp_path: StdPath) -> tuple[StdPath, StdPath, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    targets = {"a.png": torch.rand(3, 24, 40, generator=gen), "night/b.png": torch.rand(3, 32, 32, generator=gen)}
    preds   = {k: (v * 0.7 + 0.1).clamp(0, 1) for k, v in targets.items()}
    preds["a_illumination.png"] = torch.rand(3, 24, 40, generator=gen)  # a save_extra output: no target
    return _write(tmp_path / "pred", preds), _write(tmp_path / "gt", targets), preds, targets


def _quant(t: torch.Tensor) -> torch.Tensor:
    return ((t * 255).round() / 255)[None]


def test_scores_match_metrics_and_files(pair):
    pred_dir, gt_dir, preds, targets = pair
    means = evaluate(EvaluateConfig(pred=str(pred_dir), data=str(gt_dir), metrics=FR, device="cpu"))
    want_psnr = [psnr(_quant(preds[k]), _quant(targets[k])).item() for k in ("a.png", "night/b.png")]
    want_ssim = [ssim(_quant(preds[k]), _quant(targets[k])).item() for k in ("a.png", "night/b.png")]
    assert means["psnr"] == pytest.approx(sum(want_psnr) / 2) and means["ssim"] == pytest.approx(sum(want_ssim) / 2)
    rows = list(csv.reader((pred_dir / "metrics.csv").open()))
    assert rows[0] == ["stem", "psnr", "ssim"] and [r[0] for r in rows[1:]] == ["a", "night/b"]
    assert float(rows[1][1]) == pytest.approx(want_psnr[0])
    summary = yaml.safe_load((pred_dir / "metrics.yaml").read_text())
    assert summary["count"] == 2 and summary["higher_better"] == {"psnr": True, "ssim": True}
    assert summary["config"]["data"] == str(gt_dir)


def test_existing_files_raise_unless_exist_ok(pair):
    pred_dir, gt_dir, *_ = pair
    cfg = {"pred": str(pred_dir), "data": str(gt_dir), "metrics": FR, "device": "cpu"}
    evaluate(cfg)
    with pytest.raises(FileExistsError, match="exist_ok"):
        evaluate(cfg)
    evaluate(cfg | {"exist_ok": True})
    evaluate(cfg | {"gt_mean": True})  # separate files
    assert (pred_dir / "metrics_gt_mean.csv").is_file()


def test_gt_mean_matches_target_brightness():
    pred = torch.full((1, 3, 4, 4), 0.2)
    target = torch.full((1, 3, 4, 4), 0.5)
    assert torch.allclose(_gt_mean(pred, target), target)
    assert _gt_mean(torch.full((1, 3, 2, 2), 0.6), torch.ones(1, 3, 2, 2)).max() == 1.0  # clipped


def test_gt_mean_raises_psnr_for_a_brightness_error(tmp_path: StdPath):
    gen = torch.Generator().manual_seed(1)
    target = torch.rand(3, 32, 32, generator=gen) * 0.5 + 0.25
    pred_dir = _write(tmp_path / "p", {"x.png": target * 0.6})
    gt_dir   = _write(tmp_path / "g", {"x.png": target})
    plain = evaluate({"pred": str(pred_dir), "data": str(gt_dir), "metrics": ["psnr"], "device": "cpu"})
    fixed = evaluate({"pred": str(pred_dir), "data": str(gt_dir), "metrics": ["psnr"], "device": "cpu", "gt_mean": True})
    assert fixed["psnr"] > plain["psnr"] + 10


def test_no_targets_runs_no_reference_only(pair, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    pred_dir, *_ = pair

    class Fake(torch.nn.Module):
        mode, higher_better = "nr", False

        def forward(self, pred: torch.Tensor) -> torch.Tensor:
            return pred.mean(dim=(1, 2, 3)).double()

    from mon.metrics.registry import METRICS
    monkeypatch.setattr(METRICS, "build", lambda name, **kw: Fake() if name == "niqe" else METRICS.get(name)(**kw))
    means = evaluate({"pred": str(pred_dir), "metrics": ["psnr", "niqe"], "device": "cpu"})
    assert list(means) == ["niqe"] and "skipping full-reference" in caplog.text
    rows = list(csv.reader((pred_dir / "metrics.csv").open()))
    assert len(rows) == 4  # header + every prediction, extras included
    with pytest.raises(ValueError, match="nothing to compute"):
        evaluate({"pred": str(pred_dir), "metrics": FR, "device": "cpu", "exist_ok": True})


def test_missing_prediction_and_size_mismatch(pair, tmp_path: StdPath):
    pred_dir, gt_dir, *_ = pair
    extra_gt = _write(tmp_path / "gt2", {"a.png": torch.rand(3, 24, 40), "zzz.png": torch.rand(3, 8, 8)})
    with pytest.raises(ValueError, match="1 target.* no prediction.*zzz"):
        evaluate({"pred": str(pred_dir), "data": str(extra_gt), "metrics": FR, "device": "cpu"})
    wrong = _write(tmp_path / "gt3", {"a.png": torch.rand(3, 40, 24), "night/b.png": torch.rand(3, 32, 32)})
    with pytest.raises(ValueError, match="differ in size"):
        evaluate({"pred": str(pred_dir), "data": str(wrong), "metrics": FR, "device": "cpu"})


@pytest.mark.parametrize(("overrides", "error", "match"), [
    ({"metrics": ["nope"]}, KeyError, "nope"),
    ({"metrics": []}, ValueError, "empty"),
    ({"metric_args": {"lpips": {"net": "vgg"}}}, ValueError, "not in metrics"),
    ({"data": "no_such_dataset"}, FileNotFoundError, "neither a folder"),
])
def test_invalid_configs(pair, overrides: dict, error: type, match: str):
    pred_dir, gt_dir, *_ = pair
    with pytest.raises(error, match=match):
        evaluate({"pred": str(pred_dir), "data": str(gt_dir), "metrics": FR, "device": "cpu"} | overrides)


def test_cli_evaluate(pair):
    pred_dir, gt_dir, *_ = pair
    assert main(["evaluate", f"pred={pred_dir}", f"data={gt_dir}", "metrics=[psnr]", "device=cpu"]) == 0
    assert (pred_dir / "metrics.yaml").is_file()


@pytest.mark.slow
def test_dataset_targets_by_stem(tmp_path: StdPath):
    """Predicting LOL-v1 with the identity writes the inputs back; PSNR vs the targets is the input PSNR."""
    try:
        root = datasets_dir() / "lol_v1"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    means = evaluate({"pred": str(root / "test" / "images"), "data": "lol_v1", "metrics": ["psnr"],
                      "device": "cpu", "output_dir": str(tmp_path)})
    assert 7.0 < means["psnr"] < 8.5  # LOL-v1 test inputs vs targets: ~7.8 dB
