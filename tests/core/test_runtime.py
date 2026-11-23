"""Tests for ``mon.core.runtime``."""

import logging
import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from mon.core import runtime
from mon.core.runtime import available_devices, seed_everything, select_device


# --- available_devices ---

def test_available_devices_real_system():
    devices = available_devices()
    assert devices[0] == "cpu"
    assert ("mps" in devices) == torch.backends.mps.is_available()
    cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert sum(d.startswith("cuda:") for d in devices) == cuda


# --- select_device ---

@pytest.fixture
def fake_devices(monkeypatch: pytest.MonkeyPatch):
    def use(devices: list[str]) -> None:
        monkeypatch.setattr(runtime, "available_devices", lambda: list(devices))
    return use


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        (["cpu"], "cpu"),
        (["cpu", "mps"], "mps"),
        (["cpu", "cuda:0", "cuda:1"], "cuda:0"),
        (["cpu", "mps", "cuda:0"], "cuda:0"),
    ],
)
@pytest.mark.parametrize("spec", ["auto", None, " AUTO "])
def test_select_auto(fake_devices, devices: list[str], expected: str, spec: str | None):
    fake_devices(devices)
    assert select_device(spec) == torch.device(expected)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("cuda", "cuda:0"),
        ("CUDA:1", "cuda:1"),
        (1, "cuda:1"),
        (torch.device("cuda"), "cuda:0"),
        (torch.device("cuda", 1), "cuda:1"),
        ("mps", "mps"),
        ("mps:0", "mps"),
        ("cpu", "cpu"),
        (torch.device("cpu"), "cpu"),
    ],
)
def test_select_explicit(fake_devices, spec, expected: str):
    fake_devices(["cpu", "mps", "cuda:0", "cuda:1"])
    assert select_device(spec) == torch.device(expected)


@pytest.mark.parametrize("spec", ["cuda", "cuda:5", 5, "cuad", "mps", torch.device("cuda")])
def test_select_unavailable_raises_never_falls_back(fake_devices, spec):
    fake_devices(["cpu"])
    with pytest.raises(ValueError, match="not available; available devices: cpu"):
        select_device(spec)


def test_select_bad_types(fake_devices):
    fake_devices(["cpu", "cuda:0"])
    with pytest.raises(ValueError, match=">= 0"):
        select_device(-1)
    for bad in (True, 1.0, ["cpu"]):
        with pytest.raises(TypeError):
            select_device(bad)  # type: ignore[arg-type]


def test_select_real_cpu():
    assert select_device("cpu") == torch.device("cpu")


# --- seed_everything ---

@pytest.fixture
def restore_determinism(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(runtime._CUBLAS_ENV, raising=False)
    state = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.use_deterministic_algorithms(state[0], warn_only=state[1])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[2], state[3]


def _draws() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_is_reproducible():
    assert seed_everything(123) == 123
    first = _draws()
    seed_everything(123)
    assert _draws() == first
    seed_everything(124)
    assert _draws() != first


def test_seed_none_draws_logs_and_returns(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO, logger="mon.core.runtime"):
        seed = seed_everything()
    assert 0 <= seed <= runtime._MAX_SEED
    assert f"seed={seed}" in caplog.text
    first = _draws()
    seed_everything(seed)
    assert _draws() == first


@pytest.mark.parametrize(
    ("seed", "exc"),
    [(-1, ValueError), (2**32, ValueError), (1.5, TypeError), (True, TypeError), ("1", TypeError)],
)
def test_seed_validation(seed, exc: type[Exception]):
    with pytest.raises(exc):
        seed_everything(seed)


def test_deterministic_flags(restore_determinism: None):
    seed_everything(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert os.environ[runtime._CUBLAS_ENV] == ":4096:8"


def test_deterministic_keeps_user_cublas_and_warn_only(restore_determinism: None, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(runtime._CUBLAS_ENV, ":16:8")
    seed_everything(0, deterministic=True, warn_only=True)
    assert os.environ[runtime._CUBLAS_ENV] == ":16:8"
    assert torch.is_deterministic_algorithms_warn_only_enabled()
