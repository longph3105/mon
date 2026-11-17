"""Tests specific to the lazy ``mon`` / ``mon.core`` namespaces (generic checks: ``tests/test_namespaces.py``)."""

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path as StdPath

import pytest

import mon
import mon.core as mc


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_submodules_and_dir():
    assert mc.path is importlib.import_module("mon.core.path")
    assert {"Path", "load_config", "select_device", "config", "runtime"} <= set(dir(mc))
    assert mc.get_console() is importlib.import_module("mon.core.console").console


def test_top_level_unknown_attribute():
    with pytest.raises(AttributeError):
        _ = mon.nope


# --- Laziness (fresh interpreters) ---

def test_core_import_and_light_names_do_not_load_torch():
    out = _run("""
        import sys
        import mon.core as mc
        a = "torch" in sys.modules
        mc.Path, mc.load_config, mc.Registry, mc.Size, mc.Weights, mc.get_logger
        b = "torch" in sys.modules
        mc.select_device
        c = "torch" in sys.modules
        print(a, b, c)
    """)
    assert out == "False False True"


def test_top_level_mon_is_lazy():
    out = _run("""
        import sys
        import mon
        before = any(m in sys.modules for m in ("mon.core", "mon.data"))
        size = mon.core.Size(2, 3)
        data_loaded = "mon.data" in sys.modules
        print(before, size.hw, data_loaded, mon.__version__ == __import__("importlib.metadata").metadata.version("mon"))
    """)
    assert out == "False (2, 3) False True"


@pytest.mark.slow
def test_static_type_checkers_see_reexports(tmp_path: StdPath):
    snippet = tmp_path / "use_core.py"
    snippet.write_text(textwrap.dedent("""
        import mon
        import mon.core as mc
        from mon.core import Path, Size, load_config, select_device

        p: Path = mc.Path("x") / "y"
        s: Size = Size(1, 2)
        n: int = mon.core.Size(3, 4).area
        reveal_type(load_config)
        reveal_type(select_device)
    """))
    out = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(snippet)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout
    revealed = [line for line in out.stdout.splitlines() if "Revealed type" in line]
    assert len(revealed) == 2
    assert "def (path:" in revealed[0]   # real signature, not Any
    assert "device" in revealed[1]
