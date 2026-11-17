"""Tests for ``mon.core.config``."""

from dataclasses import dataclass, field
from pathlib import Path as StdPath
from typing import Any

import pytest
from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError, MissingMandatoryValue, ValidationError

from mon.core.config import BASE_KEY, load_config, save_config


def write(path: StdPath, text: str) -> StdPath:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@dataclass
class OptimizerConfig:
    name: str   = "adam"
    lr  : float = 1e-4


@dataclass
class TrainConfig:
    exp_name : str = MISSING
    epochs   : int = 100
    seed     : int | None = 0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model    : dict[str, Any] = field(default_factory=dict)  # open: model-specific kwargs


# --- Loading and layering ---

def test_plain_file_and_overrides(tmp_path: StdPath):
    f = write(tmp_path / "cfg.yaml", "epochs: 10\noptimizer:\n  lr: 0.1\n")
    cfg = load_config(f, ["optimizer.lr=0.5", "new.key=3"])
    assert isinstance(cfg, DictConfig)
    assert cfg.epochs == 10 and cfg.optimizer.lr == 0.5 and cfg.new.key == 3


def test_no_path_overrides_only():
    assert OmegaConf.to_container(load_config(overrides=["a=1"])) == {"a": 1}
    assert OmegaConf.to_container(load_config()) == {}


def test_override_can_restore_default_and_set_null(tmp_path: StdPath):
    f = write(tmp_path / "cfg.yaml", "exp_name: x\nepochs: 300\nseed: 7\n")
    cfg = load_config(f, ["epochs=100", "seed=null"], schema=TrainConfig)
    assert cfg.epochs == 100  # equal to the schema default, still applied
    assert cfg.seed is None


def test_base_inheritance_order_and_nesting(tmp_path: StdPath):
    write(tmp_path / "base" / "root.yaml", "a: root\nb: root\nc: root\nlist: [1, 2]\n")
    write(tmp_path / "base" / "mid.yaml", f"{BASE_KEY}: root.yaml\nb: mid\n")
    write(tmp_path / "other.yaml", "c: other\n")
    f = write(tmp_path / "exp" / "cfg.yaml", f"{BASE_KEY}: [../base/mid.yaml, ../other.yaml]\nlist: [9]\n")
    cfg = load_config(f, ["a=cli"])
    assert OmegaConf.to_container(cfg) == {"a": "cli", "b": "mid", "c": "other", "list": [9]}
    assert BASE_KEY not in cfg


def test_base_absolute_and_home(tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path / "home_base.yaml", "x: 1\n")
    absolute = write(tmp_path / "abs_base.yaml", "y: 2\n")
    f = write(tmp_path / "d" / "cfg.yaml", f"{BASE_KEY}: ['~/home_base.yaml', '{absolute}']\n")
    assert OmegaConf.to_container(load_config(f)) == {"x": 1, "y": 2}


def test_base_cycle_raises(tmp_path: StdPath):
    write(tmp_path / "a.yaml", f"{BASE_KEY}: b.yaml\n")
    write(tmp_path / "b.yaml", f"{BASE_KEY}: a.yaml\n")
    with pytest.raises(ValueError, match="circular _base_"):
        load_config(tmp_path / "a.yaml")


def test_missing_base_and_file(tmp_path: StdPath):
    f = write(tmp_path / "cfg.yaml", f"{BASE_KEY}: nope.yaml\n")
    with pytest.raises(FileNotFoundError):
        load_config(f)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


@pytest.mark.parametrize(("text", "match"), [("- 1\n- 2\n", "mapping"), (f"{BASE_KEY}: 3\n", "path or a list")])
def test_malformed_files(tmp_path: StdPath, text: str, match: str):
    with pytest.raises(TypeError, match=match):
        load_config(write(tmp_path / "cfg.yaml", text))


def test_interpolation_and_env(tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_DIR", "/data")
    f = write(tmp_path / "cfg.yaml", "root: ${oc.env:DATA_DIR}/lol\nout: ${root}/runs\n")
    cfg = load_config(f)
    assert cfg.out == "/data/lol/runs"


# --- Schema ---

def test_schema_defaults_and_types(tmp_path: StdPath):
    f = write(tmp_path / "cfg.yaml", "exp_name: run1\noptimizer:\n  lr: '0.01'\nmodel:\n  depth: 18\n")
    cfg = load_config(f, schema=TrainConfig)
    assert cfg.epochs == 100 and cfg.optimizer.name == "adam"
    assert cfg.optimizer.lr == 0.01 and isinstance(cfg.optimizer.lr, float)  # coerced
    assert cfg.model.depth == 18                                             # open dict


def test_schema_rejects_bad_type_and_unknown_key(tmp_path: StdPath):
    with pytest.raises(ValidationError):
        load_config(write(tmp_path / "a.yaml", "epochs: many\n"), schema=TrainConfig)
    with pytest.raises(ConfigKeyError):
        load_config(write(tmp_path / "b.yaml", "epoch: 5\n"), schema=TrainConfig)  # typo
    with pytest.raises(ConfigKeyError):
        load_config(overrides=["optimizer.lrr=1"], schema=TrainConfig)
    cfg = load_config(overrides=["exp_name=x"], schema=TrainConfig)
    with pytest.raises(ConfigAttributeError):
        _ = cfg.nope


def test_schema_missing_value_raises_on_access():
    cfg = load_config(schema=TrainConfig)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.exp_name
    assert OmegaConf.is_missing(cfg, "exp_name")


def test_schema_instance_as_defaults():
    cfg = load_config(schema=TrainConfig(exp_name="from_instance", epochs=5))
    assert cfg.exp_name == "from_instance" and cfg.epochs == 5


# --- Saving ---

def test_save_resolves_and_round_trips(tmp_path: StdPath, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATA_DIR", "/data")
    f = write(tmp_path / "cfg.yaml", "exp_name: e\nroot: ${oc.env:DATA_DIR}\nout: ${root}/runs\n")
    cfg = load_config(f)
    out = save_config(cfg, tmp_path / "runs" / "e" / "config.yaml")
    text = out.read_text()
    assert "/data/runs" in text and "${" not in text
    assert OmegaConf.to_container(load_config(out)) == OmegaConf.to_container(cfg, resolve=True)


def test_save_unresolved_keeps_interpolations(tmp_path: StdPath):
    cfg = load_config(overrides=["a=1", "b=${a}"])
    assert "${a}" in save_config(cfg, tmp_path / "c.yaml", resolve=False).read_text()


def test_save_dataclass_dict_and_missing(tmp_path: StdPath):
    out = save_config(TrainConfig(), tmp_path / "schema.yaml")
    assert "exp_name: ???" in out.read_text()
    assert load_config(save_config({"x": 1}, tmp_path / "d.yaml")).x == 1


def test_save_overwrite_false(tmp_path: StdPath):
    f = save_config({"x": 1}, tmp_path / "c.yaml")
    with pytest.raises(FileExistsError):
        save_config({"x": 2}, f, overwrite=False)
    assert load_config(f).x == 1
