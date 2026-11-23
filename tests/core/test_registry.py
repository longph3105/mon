"""Tests for ``mon.core.registry``."""

import functools
import sys
import textwrap

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.registry import Entry, Registry, RegistryKeyError


@pytest.fixture
def models() -> Registry[type]:
    reg = Registry[type]("models")

    @reg.register("zero_dce", arch="zero_dce", tasks={Task.LLE})
    class ZeroDCE:
        ...

    @reg.register("zero_dce++", arch="zero_dce", tasks={Task.LLE})
    class ZeroDCEpp:
        ...

    @reg.register(arch="unet", tasks=[Task.SEGMENT, Task.DENOISE])
    class UNet:
        def __init__(self, depth: int = 4) -> None:
            self.depth = depth

    return reg


# --- Registration ---

def test_register_returns_object_unchanged_and_names_default_to_lowercase(models: Registry[type]):
    unet = models.get("unet")
    assert unet.__name__ == "UNet"
    assert not hasattr(unet, "name") and not hasattr(unet, "arch")  # registry doesn't mutate classes
    assert list(models) == ["unet", "zero_dce", "zero_dce++"]
    assert len(models) == 3 and repr(models) == "Registry('models', 3 entries)"


def test_add_direct_and_explicit_name():
    reg = Registry[object]("optimizers")
    entry = reg.add(torch.optim.AdamW, "AdamW", family="adam")
    assert isinstance(entry, Entry) and entry.name == "adamw"
    assert reg.get("ADAMW") is torch.optim.AdamW


def test_duplicate_raises_unless_replace(models: Registry[type]):
    class Other:
        ...
    with pytest.raises(RegistryKeyError, match=r"'unet' is already registered in 'models'.*replace=True"):
        models.add(Other, "UNet")
    models.add(Other, "unet", replace=True)
    assert models.get("unet") is Other


def test_reregistering_same_object_is_noop(models: Registry[type]):
    unet = models.get("unet")
    entry = models.add(unet, arch="unet", tasks=[Task.SEGMENT, Task.DENOISE])
    assert entry is models.entry("unet")
    with pytest.raises(RegistryKeyError):
        models.add(unet, arch="different")  # same object, different metadata


def _define(module: str, qualname: str = "Net") -> type:
    """A fresh class object, as if ``module`` defining ``qualname`` were imported again."""
    namespace: dict = {"__name__": module}
    exec(f"class {qualname}: ...", namespace)
    return namespace[qualname]


def test_redefinition_from_reimported_module_replaces():
    reg = Registry[type]("models")
    old, new = _define("pkg.zoo"), _define("pkg.zoo")
    assert old is not new
    reg.add(old, "net", arch="a")
    assert reg.add(new, "net", arch="b").obj is new  # reload: newest definition wins
    assert reg.entry("net").meta["arch"] == "b"
    with pytest.raises(RegistryKeyError):
        reg.add(_define("pkg.other"), "net")          # same class name, different module
    with pytest.raises(RegistryKeyError):
        reg.add(_define("pkg.zoo", "Other"), "net")   # same module, different class


def test_name_required_without_dunder_name():
    reg = Registry[object]("fns")
    partial = functools.partial(max, 0)
    with pytest.raises(TypeError, match="pass name="):
        reg.add(partial)
    assert reg.add(partial, "max0").obj is partial


@pytest.mark.parametrize(("name", "exc"), [("", ValueError), ("   ", ValueError), (3, TypeError)])
def test_bad_names(name, exc: type[Exception]):
    with pytest.raises(exc):
        Registry[object]("x").add(object(), name)  # type: ignore[arg-type]


def test_meta_is_read_only(models: Registry[type]):
    with pytest.raises(TypeError):
        models.entry("unet").meta["arch"] = "x"  # type: ignore[index]


def test_remove(models: Registry[type]):
    assert models.remove("UNet").name == "unet"
    assert "unet" not in models


# --- Lookup ---

def test_lookup_is_case_insensitive(models: Registry[type]):
    assert "Zero_DCE++" in models and " unet " in models
    assert 3 not in models
    assert models.get("ZERO_DCE") is models.get("zero_dce")


def test_unknown_name_suggests_close_matches(models: Registry[type]):
    with pytest.raises(RegistryKeyError) as exc:
        models.get("zerodce")
    assert str(exc.value) == "'zerodce' is not registered in 'models'; did you mean 'zero_dce', 'zero_dce++'?"
    with pytest.raises(KeyError, match=r"'transformer' is not registered in 'models'$"):
        models.get("transformer")  # nothing close: no hint


def test_build(models: Registry[type]):
    net = models.build("UNet", depth=2)
    assert net.depth == 2
    reg = Registry[object]("values")
    reg.add(42, "answer")
    with pytest.raises(TypeError, match="not callable"):
        reg.build("answer")


def test_build_optimizer_with_params():
    reg = Registry[object]("optimizers")
    reg.add(torch.optim.SGD)
    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = reg.build("sgd", params, lr=0.1)
    assert isinstance(opt, torch.optim.SGD) and opt.defaults["lr"] == 0.1


# --- Filtering ---

def test_names_filter_by_equality_and_membership(models: Registry[type]):
    assert models.names() == ["unet", "zero_dce", "zero_dce++"]
    assert models.names(arch="zero_dce") == ["zero_dce", "zero_dce++"]
    assert models.names(tasks=Task.LLE) == ["zero_dce", "zero_dce++"]
    assert models.names(tasks="denoise") == ["unet"]  # StrEnum members match plain strings
    assert models.names(arch="unet", tasks=Task.SEGMENT) == ["unet"]
    assert models.names(arch="unet", tasks=Task.LLE) == []
    assert models.names(missing_key=1) == []


def test_string_meta_is_not_treated_as_collection():
    reg = Registry[object]("x")
    reg.add(object(), "a", arch="yolo12")
    assert reg.names(arch="yolo") == []  # no substring matching


def test_dataset_split_filter():
    datasets = Registry[object]("datasets")
    datasets.add(object(), "lol", tasks={Task.LLE}, splits={Split.TRAIN, Split.TEST})
    datasets.add(object(), "dicm", tasks={Task.LLE}, splits={Split.TEST})
    datasets.add(object(), "sid", tasks={Task.LLE}, splits={Split.TRAIN})
    assert datasets.names(tasks=Task.LLE) == ["dicm", "lol", "sid"]  # no hidden split requirement
    assert datasets.names(tasks=Task.LLE, splits=Split.TEST) == ["dicm", "lol"]


# --- Autoload ---

@pytest.fixture
def plugin_pkg(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway package that registers into ``REG`` when imported."""
    import mon.core.registry as regmod
    reg = regmod.Registry[type]("plugins", autoload="mon_test_plugins")
    monkeypatch.setattr(regmod, "_TEST_REG", reg, raising=False)
    pkg = tmp_path / "mon_test_plugins"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        from mon.core.registry import _TEST_REG
        from . import a  # noqa: F401
    """))
    (pkg / "a.py").write_text(textwrap.dedent("""
        from mon.core.registry import _TEST_REG

        @_TEST_REG.register("alpha", kind="x")
        class Alpha: ...
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield reg
    sys.modules.pop("mon_test_plugins", None)
    sys.modules.pop("mon_test_plugins.a", None)


def test_autoload_on_first_lookup_only(plugin_pkg: Registry[type]):
    reg = plugin_pkg
    assert "mon_test_plugins" not in sys.modules
    assert "pending" in repr(reg) and "mon_test_plugins" not in sys.modules  # repr doesn't load
    assert reg.names(kind="x") == ["alpha"]
    assert "mon_test_plugins" in sys.modules
    assert "alpha" in reg and len(reg) == 1 and list(reg) == ["alpha"] and "pending" not in repr(reg)


@pytest.mark.parametrize("lookup", [len, list, lambda r: "alpha" in r, lambda r: r.get("alpha"), lambda r: r.entries()])
def test_every_lookup_triggers_autoload(plugin_pkg: Registry[type], lookup):
    lookup(plugin_pkg)
    assert "mon_test_plugins" in sys.modules


def test_autoload_failure_is_retried():
    reg = Registry[type]("broken", autoload="mon_no_such_module_xyz")
    with pytest.raises(ModuleNotFoundError):
        reg.names()
    with pytest.raises(ModuleNotFoundError):  # not silently marked as loaded
        len(reg)


# --- Lazy entries ---

@pytest.fixture
def lazy_mod(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A throwaway module ``mon_test_lazy`` defining ``Net`` and ``Outer.Inner``; tracks whether it was imported."""
    (tmp_path / "mon_test_lazy.py").write_text(textwrap.dedent("""
        class Net:
            def __init__(self, width: int = 1) -> None:
                self.width = width

        class Outer:
            class Inner: ...
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "mon_test_lazy"
    sys.modules.pop("mon_test_lazy", None)


def test_add_lazy_filters_without_importing(lazy_mod: str):
    reg = Registry[type]("models")
    reg.add_lazy("Net", f"{lazy_mod}:Net", tasks={Task.LLE})
    assert reg.names(tasks=Task.LLE) == ["net"] and "net" in reg and len(reg) == 1 and list(reg) == ["net"]
    assert repr(reg) == "Registry('models', 1 entries)"
    assert lazy_mod not in sys.modules


@pytest.mark.parametrize("lookup", [
    lambda r: r.get("net"), lambda r: r.entry("NET").obj, lambda r: r.build("net", width=3), lambda r: r.entries(),
])
def test_lazy_lookups_import_and_cache(lazy_mod: str, lookup):
    reg = Registry[type]("models")
    reg.add_lazy("net", f"{lazy_mod}:Net", arch="x")
    lookup(reg)
    assert lazy_mod in sys.modules
    entry = reg.entry("net")
    assert entry.obj is sys.modules[lazy_mod].Net and entry.target == f"{lazy_mod}:Net" and entry.meta["arch"] == "x"
    assert reg.entry("net") is entry  # resolved once, then cached


def test_lazy_build_and_nested_attribute(lazy_mod: str):
    reg = Registry[type]("models")
    reg.add_lazy("net", f"{lazy_mod}:Net")
    reg.add_lazy("inner", f"{lazy_mod}:Outer.Inner")
    assert reg.build("net", width=3).width == 3
    assert reg.get("inner").__qualname__ == "Outer.Inner"


@pytest.mark.parametrize("target", ["mon_test_lazy", ":Net", "mon_test_lazy:", "", 3])
def test_add_lazy_rejects_malformed_targets(target):
    with pytest.raises(ValueError, match="package.module:Attr"):
        Registry[type]("models").add_lazy("net", target)  # type: ignore[arg-type]


@pytest.mark.parametrize(("target", "match"), [
    ("mon_no_such_module_xyz:Net", "No module named"),
    ("collections:NoSuchThing", "has no attribute"),
])
def test_bad_lazy_target_fails_on_lookup_with_context(target: str, match: str):
    reg = Registry[type]("models")
    reg.add_lazy("net", target)
    assert reg.names() == ["net"]  # registering and filtering don't import
    with pytest.raises(ImportError, match=rf"cannot load 'net' in 'models' from {target!r}: .*{match}"):
        reg.get("net")
    with pytest.raises(ImportError):  # not cached as broken or as resolved
        reg.get("net")


def test_add_lazy_duplicates(lazy_mod: str):
    reg = Registry[type]("models")
    reg.add_lazy("net", f"{lazy_mod}:Net", arch="x")
    reg.add_lazy("net", f"{lazy_mod}:Net", arch="x")  # same call: no-op
    with pytest.raises(RegistryKeyError, match=r"already registered .*mon_test_lazy:Net"):
        reg.add_lazy("net", f"{lazy_mod}:Outer.Inner", arch="x")
    with pytest.raises(RegistryKeyError):
        reg.add_lazy("net", f"{lazy_mod}:Net", arch="y")
    reg.get("net")
    reg.add_lazy("net", f"{lazy_mod}:Net", arch="x")  # still a no-op once resolved
    reg.add_lazy("net", f"{lazy_mod}:Outer.Inner", replace=True)
    assert reg.get("net").__qualname__ == "Outer.Inner"
    reg.add(dict, "eager")
    with pytest.raises(RegistryKeyError, match="already registered"):
        reg.add_lazy("eager", f"{lazy_mod}:Net")


def test_eager_add_resolves_matching_pending_entry(lazy_mod: str):
    reg = Registry[type]("models")
    reg.add_lazy("net", f"{lazy_mod}:Net", arch="x")
    import importlib
    net = importlib.import_module(lazy_mod).Net
    assert reg.add(net, "net", arch="x").target == f"{lazy_mod}:Net"  # e.g. a decorator in the target module
    assert reg.get("net") is net
    reg.add_lazy("other", f"{lazy_mod}:Net")
    with pytest.raises(RegistryKeyError, match="lazily"):
        reg.add(dict, "other")  # a different object than the pending target
