"""Tests for ``mon.runners.predictor`` and the ``mon predict`` CLI (CPU, tiny images, random-init models)."""

from pathlib import Path as StdPath

import pytest
import torch
import yaml

from mon.data.io import read_image, write_image
from mon.models.registry import MODELS, load
from mon.runners.cli import main
from mon.runners.predictor import PredictConfig, predict


def _images(root: StdPath, sizes: dict[str, tuple[int, int]]) -> StdPath:
    gen = torch.Generator().manual_seed(0)
    for name, (h, w) in sizes.items():
        write_image(torch.randint(0, 80, (3, h, w), dtype=torch.uint8, generator=gen), root / name)
    return root


@pytest.fixture
def images(tmp_path: StdPath) -> StdPath:
    return _images(tmp_path / "night", {"a.png": (30, 45), "sub/b.png": (24, 32)})


def _cfg(images: StdPath, tmp_path: StdPath, **kwargs: object) -> PredictConfig:
    fields: dict[str, object] = {"data": str(images), "run_dir": str(tmp_path / "run"), "device": "cpu", "seed": 0}
    return PredictConfig(**(fields | kwargs))  # type: ignore[arg-type]


# --- Weighted models ---

def test_mirrors_input_tree_and_matches_direct_forward(images: StdPath, tmp_path: StdPath):
    run = predict(_cfg(images, tmp_path, model="zero_dce", weights=None))
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*.png")) == ["a.png", "sub/b.png"]
    torch.manual_seed(0)
    model = load("zero_dce", weights=None)
    x = read_image(images / "a.png").float()[None] / 255
    with torch.no_grad():
        expected = (model(x)["enhanced"].clamp(0, 1)[0] * 255).round().to(torch.uint8)
    assert (read_image(run / "a.png").int() - expected.int()).abs().max() <= 1
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    assert cfg["seed"] == 0 and cfg["zero_shot"] is False and cfg["run_dir"] == str(run)
    assert set(yaml.safe_load((run / "timing.yaml").read_text())) == {"infer", "save"}


def test_pads_to_input_multiple_and_crops_back(images: StdPath, tmp_path: StdPath):
    assert MODELS.get("dccnet").input_multiple == 8  # type: ignore[attr-defined]
    run = predict(_cfg(images, tmp_path, model="dccnet", weights=None))  # 30x45 is not a multiple of 8
    assert read_image(run / "a.png").shape == (3, 30, 45)
    assert read_image(run / "sub/b.png").shape == (3, 24, 32)


def test_save_extra_writes_image_shaped_outputs(images: StdPath, tmp_path: StdPath):
    run = predict(_cfg(images, tmp_path, model="retinexnet", weights=None, save_extra=True))
    names = sorted(p.name for p in run.glob("*.png"))
    assert "a.png" in names and "a_illumination.png" in names and "a_reflectance.png" in names
    assert read_image(run / "a_illumination.png").shape[-2:] == (30, 45)


def test_tiled_matches_whole_when_one_tile(images: StdPath, tmp_path: StdPath):
    whole = predict(_cfg(images, tmp_path, model="zero_dce", weights=None))
    tiled = predict(PredictConfig(model="zero_dce", weights=None, data=str(images), run_dir=str(tmp_path / "t"),
                                  device="cpu", seed=0, tile=64, tile_overlap=16))
    assert torch.equal(read_image(whole / "a.png"), read_image(tiled / "a.png"))


def test_output_key_and_batching(tmp_path: StdPath):
    same = _images(tmp_path / "same", {"a.png": (16, 16), "b.png": (16, 16)})
    run = predict(_cfg(same, tmp_path, model="retinexnet", weights=None, batch_size=2, output_key="reflectance"))
    assert sorted(p.name for p in run.glob("*.png")) == ["a.png", "b.png"]
    with pytest.raises(KeyError, match="not in the model's outputs"):
        predict(_cfg(same, tmp_path, model="zero_dce", weights=None, output_key="nope", run_dir=str(tmp_path / "x")))


# --- Zero-shot models ---

def test_zero_shot_fits_per_image_with_seed(tmp_path: StdPath):
    imgs = _images(tmp_path / "in", {"odd.png": (15, 21)})  # odd: padded to even for Zero-IG, cropped back
    run = predict(_cfg(imgs, tmp_path, model="zero_ig", fit_args={"iterations": 1}))
    first = read_image(run / "odd.png")
    assert first.shape == (3, 15, 21)
    again = predict(_cfg(imgs, tmp_path, model="zero_ig", fit_args={"iterations": 1}, run_dir=str(tmp_path / "r2")))
    assert torch.equal(first, read_image(again / "odd.png"))
    assert yaml.safe_load((run / "config.yaml").read_text())["zero_shot"] is True


def test_zero_shot_false_runs_released_weights_path(tmp_path: StdPath):
    imgs = _images(tmp_path / "in", {"a.png": (16, 16)})
    run = predict(_cfg(imgs, tmp_path, model="zero_ig", zero_shot=False, weights=None))
    assert read_image(run / "a.png").shape == (3, 16, 16)


def test_zero_shot_rejects_tiling(images: StdPath, tmp_path: StdPath):
    with pytest.raises(ValueError, match="tile is not supported"):
        predict(_cfg(images, tmp_path, model="zs_n2n", tile=32))


# --- Run directory and validation ---

def test_existing_run_dir_raises_and_is_kept(images: StdPath, tmp_path: StdPath):
    run = predict(_cfg(images, tmp_path, model="zero_dce", weights=None))
    marker = run / "keep.txt"
    marker.write_text("x")
    with pytest.raises(FileExistsError, match="exist_ok"):
        predict(_cfg(images, tmp_path, model="zero_dce", weights=None))
    predict(_cfg(images, tmp_path, model="zero_dce", weights=None, exist_ok=True))
    assert marker.read_text() == "x"


def test_default_run_dir(images: StdPath, tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    run = predict(PredictConfig(model="zero_dce", weights=None, data=str(images), device="cpu", seed=0))
    assert run == tmp_path / "runs" / "predict" / "zero_dce-random" / "night"
    run = predict(PredictConfig(model="zero_dce", weights=None, data=str(images / "a.png"), device="cpu", seed=0))
    assert run.name == "a"
    run = predict(PredictConfig(model="zs_n2n", data=[str(images / "a.png")], device="cpu", seed=0,
                                weights="ignored", fit_args={"iterations": 1}))
    assert run == tmp_path / "runs" / "predict" / "zs_n2n" / "files"


@pytest.mark.parametrize(("overrides", "error", "match"), [
    ({"model": "mertens"}, ValueError, "exposure stacks"),
    ({"model": "dccnet", "pad_multiple": 12}, ValueError, "multiple of dccnet's input_multiple=8"),
    ({"model": "zero_dce", "padding_mode": "circular"}, ValueError, "padding_mode"),
    ({"model": "zero_dce", "data": "no_such_dataset_or_path"}, FileNotFoundError, "neither an existing path"),
])
def test_invalid_configs(images: StdPath, tmp_path: StdPath, overrides: dict, error: type, match: str):
    cfg = {"data": str(images), "run_dir": str(tmp_path / "run"), "device": "cpu", "weights": None, **overrides}
    with pytest.raises(error, match=match):
        predict(cfg)
    assert not (tmp_path / "run").exists()


def test_requires_model_and_data():
    with pytest.raises(ValueError, match="model is required"):
        predict({"data": "x"})


# --- CLI ---

def test_cli_predict(images: StdPath, tmp_path: StdPath, capsys: pytest.CaptureFixture[str]):
    cfg_file = tmp_path / "predict.yaml"
    cfg_file.write_text("model: zero_dce\nweights: null\ndevice: cpu\nseed: 0\n")
    assert main(["predict", str(cfg_file), f"data={images}", f"run_dir={tmp_path / 'cli'}"]) == 0
    assert capsys.readouterr().out.strip().endswith("cli")
    assert (tmp_path / "cli" / "sub" / "b.png").is_file()


def test_cli_usage(capsys: pytest.CaptureFixture[str]):
    assert main(["--help"]) == 0 and "usage: mon predict" in capsys.readouterr().out
    assert main([]) == 2
    assert main(["train"]) == 2 and "unknown command" in capsys.readouterr().err
