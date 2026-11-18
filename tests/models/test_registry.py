"""Tests for ``mon.models.registry``: the lazy model index and ``load``."""

import importlib
import inspect
import pkgutil
import subprocess
import sys

import pytest
import torch
from torch import nn

import mon.models
from mon.core.enums import Task
from mon.core.weights import Weights, WeightsEnum
from mon.models.registry import MODELS, load


def _model_modules() -> list[str]:
    return [m.name for m in pkgutil.walk_packages(mon.models.__path__, "mon.models.") if m.name != "mon.models.registry"]


def _public_model_classes() -> dict[str, type]:
    """``"module:Class"`` → class for every public ``nn.Module`` defined in a ``mon.models`` module, except losses (``*Loss``)."""
    found = {}
    for name in _model_modules():
        module = importlib.import_module(name)
        for attr in getattr(module, "__all__", []):
            obj = getattr(module, attr)
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj.__module__ == name \
                    and not attr.endswith("Loss"):
                found[f"{name}:{obj.__qualname__}"] = obj
    return found


# --- Index ---

def test_listing_imports_neither_models_nor_torch():
    code = ("import sys; from mon.models import MODELS; from mon.core.enums import Task; "
            "names = MODELS.names(tasks=Task.LLE, arch='zero_dce'); "
            "print(names, 'torch' in sys.modules, any(m.startswith('mon.models.lle') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "['zero_dce', 'zero_dce++'] False False"


def test_every_entry_resolves_to_a_model_with_weights_enum_and_tasks():
    for entry in MODELS.entries():
        assert inspect.isclass(entry.obj) and issubclass(entry.obj, nn.Module), entry.name
        assert entry.target == f"{entry.obj.__module__}:{entry.obj.__qualname__}"
        enum = getattr(entry.obj, "weights_enum", None)
        assert enum is None or issubclass(enum, WeightsEnum)
        assert entry.meta["tasks"] and all(isinstance(t, Task) for t in entry.meta["tasks"])


def test_every_public_model_class_is_indexed_once():
    targets = [MODELS.entry(n).target for n in MODELS]
    assert len(targets) == len(set(targets))
    assert set(_public_model_classes()) == set(targets)


def test_entries_by_task_and_arch():
    assert MODELS.names(tasks=Task.LLE) == ["clode", "colie", "dccnet", "flol", "hvi_cidnet", "pairlie", "sci", "sci++", "zero_dce", "zero_dce++"]
    assert MODELS.names(arch="zero_dce") == ["zero_dce", "zero_dce++"]
    assert MODELS.get("Zero_DCE++").__name__ == "ZeroDCEPP"


# --- load ---

def test_load_random_init_in_eval_mode_with_kwargs():
    model = load("zero_dce++", weights=None, scale_factor=4)
    assert type(model).__name__ == "ZeroDCEPP" and not model.training and model.scale_factor == 4


def test_load_default_released_weights():
    from mon.models.lle.zero_dce import ZeroDCE_Weights
    try:
        expected = ZeroDCE_Weights.DEFAULT.weights.load_state_dict(download=False)
    except FileNotFoundError:
        pytest.skip("released weights not available")
    for spec in ("default", "SICE_ME", ZeroDCE_Weights.SICE_ME):
        model = load("zero_dce", spec)
        assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())


def test_load_checkpoint_path_and_device(tmp_path):
    source = load("zero_dce", weights=None, hidden_dim=8)
    path = tmp_path / "net.pt"
    torch.save(source.state_dict(), path)
    for spec in (path, str(path), Weights(str(path))):
        model = load("zero_dce", spec, device="cpu", hidden_dim=8)
        assert torch.equal(model.e_conv1.weight, source.e_conv1.weight)
        assert model.e_conv1.weight.device.type == "cpu"


def test_load_mismatched_checkpoint_raises_unless_not_strict(tmp_path):
    path = tmp_path / "small.pt"
    torch.save({"e_conv1.bias": torch.zeros(32)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load("zero_dce", path)
    assert torch.equal(load("zero_dce", path, strict=False).e_conv1.bias, torch.zeros(32))


def test_load_errors(tmp_path):
    with pytest.raises(KeyError, match="did you mean 'zero_dce'"):
        load("zerodce")
    with pytest.raises(FileNotFoundError):
        load("zero_dce", tmp_path / "missing.pt")

    class NoWeights(nn.Module):  # learned, but no released weights
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

    class Classical(nn.Module): ...  # no parameters
    MODELS.add(NoWeights, "no_weights_test")
    MODELS.add(Classical, "classical_test")
    try:
        with pytest.raises(ValueError, match="has no pretrained weights"):
            load("no_weights_test")
        assert isinstance(load("no_weights_test", weights=None), NoWeights)
        assert isinstance(load("classical_test"), Classical)  # "default" = nothing to load
    finally:
        MODELS.remove("no_weights_test")
        MODELS.remove("classical_test")
