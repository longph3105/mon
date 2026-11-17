"""Tests for the lazy ``mon`` / ``mon.core`` namespaces."""

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path as StdPath

import pytest

import mon
import mon.core as mc

_EXCLUDED = {("console", "console")}  # (submodule, name) not re-exported: collides with the submodule name


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


# --- Export table ---

def test_exports_match_submodule_all():
    expected = {
        name: sub
        for sub in mc._SUBMODULES
        for name in importlib.import_module(f"mon.core.{sub}").__all__
        if (sub, name) not in _EXCLUDED
    }
    assert mc._EXPORTS == expected


def test_all_is_sorted_literal_of_exports():
    assert mc.__all__ == sorted(mc._EXPORTS)


def test_no_export_shadows_a_submodule():
    assert not set(mc._EXPORTS) & set(mc._SUBMODULES)


@pytest.mark.parametrize("name", sorted(mc._EXPORTS))
def test_every_export_resolves_to_the_submodule_object(name: str):
    module = importlib.import_module(f"mon.core.{mc._EXPORTS[name]}")
    assert getattr(mc, name) is getattr(module, name)


def test_submodules_and_dir():
    assert mc.path is importlib.import_module("mon.core.path")
    assert {"Path", "load_config", "select_device", "config", "runtime"} <= set(dir(mc))
    assert mc.get_console() is importlib.import_module("mon.core.console").console


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = mc.nope
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
        before = "mon.core" in sys.modules
        size = mon.core.Size(2, 3)
        print(before, size.hw, mon.__version__ == __import__("importlib.metadata").metadata.version("mon"))
    """)
    assert out == "False (2, 3) True"


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


# --- Overview table in the package docstring ---

import re  # noqa: E402


def _overview_rows() -> list[tuple[str, str, list[str]]]:
    """Parse the rST simple table in ``mon.core.__doc__`` by its ``=`` rule columns."""
    lines = [line for line in (mc.__doc__ or "").splitlines()]
    rule_idx = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*=+(\s+=+){2}\s*", line)]
    assert len(rule_idx) == 3, "expected header rule, header separator, and closing rule"
    spans = [m.span() for m in re.finditer(r"=+", lines[rule_idx[0]])]
    rows = []
    for line in lines[rule_idx[1] + 1 : rule_idx[2]]:
        for (_, end), (nxt, _) in zip(spans, spans[1:]):
            assert not line[end:nxt].strip(), f"text in a column gap (misaligned row): {line!r}"
        assert len(line) <= spans[-1][1], f"row wider than the table: {line!r}"
        module, purpose, names = (line[a:b].strip() for a, b in spans)
        rows.append((module, purpose, [n.strip() for n in names.split(",")]))
    return rows


def test_overview_lists_every_submodule_once():
    modules = [m for m, _, _ in _overview_rows()]
    assert sorted(modules) == sorted(mc._SUBMODULES) and len(modules) == len(set(modules))


@pytest.mark.parametrize(("module", "purpose", "names"), _overview_rows())
def test_overview_row_matches_code(module: str, purpose: str, names: list[str]):
    sub = importlib.import_module(f"mon.core.{module}")
    assert purpose == (sub.__doc__ or "").strip().splitlines()[0]
    for name in names:
        assert mc._EXPORTS.get(name) == module, f"{name!r} is not exported from mon.core.{module}"
