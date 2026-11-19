"""Tests for ``mon.runners.trainer`` and ``mon train`` (tiny synthetic data; CPU unless marked ``gpu``)."""

import csv
from pathlib import Path as StdPath

import pytest
import torch
import yaml

from mon.data.io import write_image
from mon.models.registry import MODELS, load
from mon.nn.training import TrainRecipe
from mon.runners import trainer
from mon.runners.cli import main
from mon.runners.trainer import TrainConfig, resolve_train_config, train

FAST = {"resize": [32, 32], "batch_size": 2, "num_workers": 0, "device": "cpu", "seed": 0}


def _images(root: StdPath, n: int = 4, size: tuple[int, int] = (40, 48), *, seed: int = 0) -> StdPath:
    gen = torch.Generator().manual_seed(seed)
    for i in range(n):
        write_image(torch.randint(0, 120, (3, *size), dtype=torch.uint8, generator=gen), root / f"{i:02d}.png")
    return root


@pytest.fixture
def folder(tmp_path: StdPath) -> StdPath:
    return _images(tmp_path / "train")


def _cfg(folder: StdPath, tmp_path: StdPath, **kw: object) -> dict[str, object]:
    return {"model": "zero_dce", "data": str(folder), "epochs": 1, "run_dir": str(tmp_path / "run"), **FAST, **kw}


# --- Config ---

def test_recipe_defaults_sit_under_user_values():
    cfg = resolve_train_config({"model": "zero_dce", "epochs": 3})
    assert (cfg.data, cfg.resize, cfg.batch_size, cfg.grad_clip) == ("sice_me", [256, 256], 8, 0.1)
    assert (cfg.optimizer.name, cfg.optimizer.lr, cfg.optimizer.weight_decay, cfg.epochs) == ("adam", 1e-4, 1e-4, 3)
    pp = resolve_train_config({"model": "zero_dce++", "model_args": {"num_iters": 8}})
    assert pp.resize == [512, 512] and pp.model_args == {"scale_factor": 1.0, "num_iters": 8}
    assert resolve_train_config({"model": "zero_ig", "data": "x"}).shuffle is False


@pytest.mark.parametrize(("cfg", "match"), [
    ({"data": "x"}, "model is required"),
    ({"model": "dccnet", "data": "x"}, "no training recipe"),
    ({"model": "sci"}, "names no training set"),
    ({"model": "zero_dce", "optimizer": {"name": "lion"}}, "optimizer.name"),
    ({"model": "zero_dce", "monitor": "psnr"}, "needs val_data"),
    ({"model": "zero_dce", "resize": [32]}, r"resize must be \[H, W\]"),
])
def test_invalid_configs(cfg: dict, match: str):
    with pytest.raises(ValueError, match=match):
        resolve_train_config(cfg)


def test_registry_recipes_resolve():
    trainable = [e.name for e in MODELS.entries() if e.meta.get("train")]
    assert trainable == ["sci", "zero_dce", "zero_dce++", "zero_ig"]
    for name in trainable:
        assert isinstance(trainer._recipe(name), TrainRecipe)


# --- Runs ---

def test_run_writes_log_checkpoints_and_loadable_weights(folder: StdPath, tmp_path: StdPath):
    run = train(_cfg(folder, tmp_path, epochs=2, save_every=1))
    assert {p.name for p in run.iterdir()} == {"config.yaml", "log.csv", "last.ckpt", "best.pt",
                                              "epoch_0001.pt", "epoch_0002.pt"}
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"] and rows[1]["step"] == "4"
    assert {"train/loss", "train/spatial", "train/exposure", "train/color", "train/smoothness"} <= set(rows[0])
    saved = yaml.safe_load((run / "config.yaml").read_text())
    assert saved["seed"] == 0 and saved["optimizer"]["lr"] == 1e-4  # recipe value recorded
    model = load("zero_dce", weights=str(run / "best.pt"))
    assert model(torch.rand(1, 3, 16, 16))["enhanced"].shape == (1, 3, 16, 16)


def test_resume_continues_exactly(folder: StdPath, tmp_path: StdPath):
    straight = train(_cfg(folder, tmp_path, epochs=3, hflip=True, run_dir=str(tmp_path / "a")))
    part     = train(_cfg(folder, tmp_path, epochs=1, hflip=True, run_dir=str(tmp_path / "b")))
    resumed  = train({"resume": str(part), "epochs": 3})
    a = torch.load(straight / "last.ckpt", weights_only=True)
    b = torch.load(resumed / "last.ckpt", weights_only=True)
    assert a["step"] == b["step"] == 6
    for k, v in a["model"].items():
        assert torch.equal(v, b["model"][k]), k
    rows = list(csv.DictReader((resumed / "log.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert train({"resume": str(resumed / "last.ckpt"), "epochs": 3}) == resumed  # done: no-op
    with pytest.raises(ValueError, match="only"):
        train({"resume": str(resumed), "batch_size": 4})


def test_existing_run_dir_raises(folder: StdPath, tmp_path: StdPath):
    train(_cfg(folder, tmp_path))
    with pytest.raises(FileExistsError, match="exist_ok"):
        train(_cfg(folder, tmp_path))


def test_validation_with_targets_and_best_by_metric(tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "datasets" / "lol_v1"
    for split in ("train", "test"):
        _images(root / split / "images", n=2, seed=1)
        _images(root / split / "targets", n=2, seed=2)
    monkeypatch.setenv("DATASETS_DIR", str(tmp_path / "datasets"))
    run = train({"model": "zero_dce", "data": "lol_v1", "val_data": "lol_v1", "epochs": 2, "val_every": 2,
                 "run_dir": str(tmp_path / "run"), **FAST})
    rows = list(csv.DictReader((run / "log.csv").open()))
    assert rows[0]["val/psnr"] == "" and float(rows[1]["val/psnr"]) > 0  # validated on epoch 2 only
    best = torch.load(run / "last.ckpt", weights_only=True)["best"]
    assert best["name"] == "psnr" and best["epoch"] == 2


def test_fr_metrics_without_targets_raise(folder: StdPath, tmp_path: StdPath):
    with pytest.raises(ValueError, match="no targets"):
        train(_cfg(folder, tmp_path, val_data=str(folder)))


@pytest.mark.parametrize("model", ["sci", "zero_dce++", "zero_ig"])
def test_each_recipe_trains_one_epoch(model: str, folder: StdPath, tmp_path: StdPath):
    run = train(_cfg(folder, tmp_path, model=model, batch_size=1))
    row = next(csv.DictReader((run / "log.csv").open()))
    assert float(row["train/loss"]) > 0 and row["step"] == "4"


def test_non_finite_loss_raises(folder: StdPath, tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    real = trainer._recipe("zero_dce")
    bad  = TrainRecipe(loss=real.loss, step=lambda m, l, b: {"loss": m(b["image"])["enhanced"].sum() * float("nan")})
    monkeypatch.setattr(trainer, "_recipe", lambda name: bad)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(_cfg(folder, tmp_path))


@pytest.mark.gpu
def test_amp_on_cuda(folder: StdPath, tmp_path: StdPath):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    run = train(_cfg(folder, tmp_path, device="cuda", amp=True))
    assert torch.load(run / "last.ckpt", weights_only=True)["scaler"]["scale"] > 0


def test_cli_train(folder: StdPath, tmp_path: StdPath, capsys: pytest.CaptureFixture[str]):
    args = ["train", "model=zero_dce", f"data={folder}", "epochs=1", f"run_dir={tmp_path / 'cli'}",
            "resize=[32,32]", "batch_size=2", "num_workers=0", "device=cpu", "seed=0"]
    assert main(args) == 0 and capsys.readouterr().out.strip().endswith("cli")


def test_full_config_instance_overrides_recipe(folder: StdPath, tmp_path: StdPath):
    cfg = resolve_train_config(TrainConfig(model="zero_dce", data=str(folder)))
    assert cfg.optimizer.lr == 1e-3 and cfg.resize is None  # schema defaults, not the recipe's


def test_cache_gives_identical_training(folder: StdPath, tmp_path: StdPath):
    plain  = train(_cfg(folder, tmp_path, epochs=2, run_dir=str(tmp_path / "plain")))
    cached = train(_cfg(folder, tmp_path, epochs=2, cache=True, num_workers=2, run_dir=str(tmp_path / "cached")))
    a = torch.load(plain / "last.ckpt", weights_only=True)["model"]
    b = torch.load(cached / "last.ckpt", weights_only=True)["model"]
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    with pytest.raises(ValueError, match="deterministic"):
        resolve_train_config({"model": "zero_dce", "cache": True, "hflip": True})


@pytest.mark.gpu
def test_device_cache_matches_ram_cache(folder: StdPath, tmp_path: StdPath):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    ram = train(_cfg(folder, tmp_path, device="cuda", cache=True, run_dir=str(tmp_path / "ram")))
    gpu = train(_cfg(folder, tmp_path, device="cuda", cache=True, cache_device="device", run_dir=str(tmp_path / "gpu")))
    a = torch.load(ram / "last.ckpt", weights_only=True)["model"]
    b = torch.load(gpu / "last.ckpt", weights_only=True)["model"]
    assert all(torch.allclose(v, b[k], atol=1e-6) for k, v in a.items())


def test_auto_batch_size_is_opt_in(folder: StdPath, tmp_path: StdPath):
    assert resolve_train_config({"model": "zero_dce"}).batch_size == 8  # the recipe's, unless asked
    assert resolve_train_config({"model": "zero_dce", "batch_size": "auto"}).batch_size == "auto"
    with pytest.raises(ValueError, match="batch_size must be"):
        resolve_train_config({"model": "zero_dce", "batch_size": 0})
    with pytest.raises(ValueError, match="needs a CUDA device"):
        train(_cfg(folder, tmp_path, batch_size="auto"))


@pytest.mark.gpu
def test_auto_batch_size_on_cuda(folder: StdPath, tmp_path: StdPath, caplog: pytest.LogCaptureFixture):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    run = train(_cfg(folder, tmp_path, device="cuda", batch_size="auto"))
    chosen = yaml.safe_load((run / "config.yaml").read_text())["batch_size"]
    assert chosen == 4  # capped by the 4 training images; a power of two
    assert "does not reproduce the paper" in caplog.text  # recipe batch 8 != 4


@pytest.mark.gpu
def test_resume_on_cuda(folder: StdPath, tmp_path: StdPath):
    """The checkpoint loads onto the GPU; its RNG states must still restore (they need CPU tensors)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    part = train(_cfg(folder, tmp_path, epochs=1, device="cuda", run_dir=str(tmp_path / "gpu")))
    resumed = train({"resume": str(part), "epochs": 2})
    assert torch.load(resumed / "last.ckpt", weights_only=True)["epoch"] == 2
