"""Tests for ``mon.core.weights``."""

import functools
import hashlib
import pathlib
from collections.abc import Callable

import pytest
import torch

from mon.core.constants import DATA_DIR_ENV, ZOO_DIR_ENV
from mon.core.weights import Weights, WeightsEnum, resolve_weights


@pytest.fixture
def zoo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    z = tmp_path / "zoo"
    (z / "lle" / "net").mkdir(parents=True)
    torch.save({"w": torch.ones(2)}, z / "lle" / "net" / "lol.pt")
    monkeypatch.setenv(ZOO_DIR_ENV, str(z))
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    return z


class Net_Weights(WeightsEnum):
    LOL  = Weights("lle/net/lol.pt", name="lol", meta={"psnr": 21.3})
    SICE = Weights("lle/net/sice.pt", name="sice", url=None)
    DEFAULT = LOL


# --- Weights ---

def test_path_kept_as_given_and_frozen(zoo: pathlib.Path):
    w = Weights(pathlib.Path("lle/net/lol.pt"))
    assert w.path == "lle/net/lol.pt"  # not normalized against cwd
    with pytest.raises(AttributeError):
        w.path = "x"  # type: ignore[misc]
    with pytest.raises(ValueError):
        Weights("  ")


def test_resolve_and_load_from_zoo(zoo: pathlib.Path):
    w = Net_Weights.LOL.weights
    assert w.resolve() == zoo / "lle" / "net" / "lol.pt"
    state = w.load_state_dict()
    assert torch.equal(state["w"], torch.ones(2))


def test_missing_file_raises_never_skips(zoo: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="no url to download from"):
        Net_Weights.SICE.weights.resolve()
    with pytest.raises(FileNotFoundError):
        Net_Weights.SICE.weights.load_state_dict()


def test_weights_only_refuses_arbitrary_objects(zoo: pathlib.Path):
    torch.save({"fn": functools.partial(print)}, zoo / "evil.pt")
    with pytest.raises(Exception, match="(?i)weights.only|unsupported|global"):
        Weights("evil.pt").load_state_dict()
    assert "fn" in Weights("evil.pt").load_state_dict(weights_only=False)


def test_resolve_downloads_to_zoo_with_hash(zoo: pathlib.Path, tmp_path: pathlib.Path):
    remote = tmp_path / "remote.pt"
    torch.save({"w": torch.zeros(1)}, remote)
    digest = hashlib.sha256(remote.read_bytes()).hexdigest()
    w = Weights("lle/net/remote.pt", url=remote.as_uri(), sha256=digest)
    path = w.resolve()
    assert path == zoo / "lle" / "net" / "remote.pt" and path.is_file()
    with pytest.raises(FileNotFoundError):
        Weights("lle/net/other.pt", url=remote.as_uri()).resolve(download=False)


def test_equality_ignores_transforms_and_meta():
    a = Weights("x.pt", transforms=functools.partial(int), meta={"a": 1})
    b = Weights("x.pt", transforms=functools.partial(int), meta={"b": 2})
    assert a == b and hash(a) == hash(b)
    assert Weights("x.pt") != Weights("y.pt")


# --- WeightsEnum ---

def test_enum_alias_and_type_check():
    assert Net_Weights.DEFAULT is Net_Weights.LOL
    assert [m.name for m in Net_Weights] == ["LOL", "SICE"]
    with pytest.raises(TypeError, match="must be a Weights"):
        class Bad(WeightsEnum):
            X = "path.pt"


# --- resolve_weights ---

@pytest.mark.parametrize("spec", [None, "", "none", "None", "null"])
def test_resolve_none_like(spec):
    assert resolve_weights(spec, Net_Weights) is None


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("default", Net_Weights.LOL.weights),
        ("DEFAULT", Net_Weights.LOL.weights),
        ("sice", Net_Weights.SICE.weights),
        (" Lol ", Net_Weights.LOL.weights),
        (Net_Weights.SICE, Net_Weights.SICE.weights),
        (Net_Weights.LOL.weights, Net_Weights.LOL.weights),
    ],
)
def test_resolve_from_enum(spec, expected: Weights):
    assert resolve_weights(spec, Net_Weights) is expected


def test_resolve_paths():
    assert resolve_weights("some/other.pt", Net_Weights) == Weights("some/other.pt")
    assert resolve_weights(pathlib.Path("/abs/w.pt")) == Weights("/abs/w.pt")
    assert resolve_weights("lol") == Weights("lol")  # no enum: treated as a path


def test_resolve_default_errors():
    with pytest.raises(ValueError, match="need the model's WeightsEnum"):
        resolve_weights("default")

    class NoDefault(WeightsEnum):
        A = Weights("a.pt")

    with pytest.raises(ValueError, match="no DEFAULT member; choose one of: A"):
        resolve_weights("default", NoDefault)


def test_resolve_bad_type():
    with pytest.raises(TypeError):
        resolve_weights(3)  # type: ignore[arg-type]
