"""Checks shared by every lazy ``mon`` package namespace (``mon.core``, ``mon.data``, …)."""

import ast
import importlib
import re
from pathlib import Path as StdPath
from types import ModuleType

import pytest

PACKAGES = ["mon.core", "mon.data", "mon.metrics", "mon.models", "mon.nn", "mon.nn.losses", "mon.runners"]

# (package, submodule, name) deliberately not re-exported.
EXCLUDED = {("mon.core", "console", "console")}  # collides with the ``console`` submodule name

# Which ``mon`` packages each package may import (layering).
ALLOWED_IMPORTS = {
    "mon.core": {"mon.core"},
    "mon.data": {"mon.core", "mon.data"},
    "mon.metrics": {"mon.core", "mon.metrics"},
    "mon.models": {"mon.core", "mon.models", "mon.nn"},
    "mon.nn"  : {"mon.core", "mon.nn"},
    "mon.nn.losses": {"mon.core", "mon.nn"},
    "mon.runners": {"mon.core", "mon.data", "mon.metrics", "mon.models", "mon.nn", "mon.runners"},
}

SRC = StdPath(__file__).resolve().parents[1] / "src"


def _pkg(name: str) -> ModuleType:
    return importlib.import_module(name)


# --- Export table ---

@pytest.mark.parametrize("package", PACKAGES)
def test_exports_match_submodule_all(package: str):
    pkg = _pkg(package)
    expected = {
        name: sub
        for sub in pkg._SUBMODULES
        for name in importlib.import_module(f"{package}.{sub}").__all__
        if (package, sub, name) not in EXCLUDED
    }
    assert pkg._EXPORTS == expected


@pytest.mark.parametrize("package", PACKAGES)
def test_all_is_sorted_literal_of_exports(package: str):
    pkg = _pkg(package)
    assert pkg.__all__ == sorted(pkg._EXPORTS)
    tree = ast.parse((SRC / package.replace(".", "/") / "__init__.py").read_text())
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets)
    ]
    assert assigns and isinstance(assigns[0].value, ast.List), "__all__ must be a literal list for static tools"


@pytest.mark.parametrize("package", PACKAGES)
def test_no_export_shadows_a_submodule(package: str):
    pkg = _pkg(package)
    assert not set(pkg._EXPORTS) & set(pkg._SUBMODULES)


@pytest.mark.parametrize(("package", "name"), [(p, n) for p in PACKAGES for n in sorted(_pkg(p)._EXPORTS)])
def test_every_export_resolves(package: str, name: str):
    pkg = _pkg(package)
    module = importlib.import_module(f"{package}.{pkg._EXPORTS[name]}")
    assert getattr(pkg, name) is getattr(module, name)


@pytest.mark.parametrize("package", PACKAGES)
def test_dir_and_unknown_attribute(package: str):
    pkg = _pkg(package)
    assert set(pkg._EXPORTS) | set(pkg._SUBMODULES) <= set(dir(pkg))
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        _ = pkg.nope


# --- Layering ---

@pytest.mark.parametrize("package", PACKAGES)
def test_package_only_imports_allowed_mon_packages(package: str):
    allowed = ALLOWED_IMPORTS[package]
    for file in (SRC / package.replace(".", "/")).rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else [])
            for name in names:
                if name == "mon" or name.startswith("mon."):
                    top = ".".join(name.split(".")[:2])
                    assert top in allowed, (
                        f"{file.relative_to(SRC)} imports {name}; {package} may import only {sorted(allowed)}"
                    )


# --- Overview table in the package docstring ---

def _overview_rows(package: str) -> list[tuple[str, str, list[str]]]:
    """Parse the rST simple table in the package docstring by its ``=`` rule columns.

    A row with an empty first column continues the row above (a wrapped purpose or name list).
    """
    lines = (_pkg(package).__doc__ or "").splitlines()
    rule_idx = [i for i, line in enumerate(lines) if re.fullmatch(r"\s*=+(\s+=+){2}\s*", line)]
    assert len(rule_idx) == 3, "expected header rule, header separator, and closing rule"
    spans = [m.span() for m in re.finditer(r"=+", lines[rule_idx[0]])]
    cells: list[list[str]] = []
    for line in lines[rule_idx[1] + 1 : rule_idx[2]]:
        for (_, end), (nxt, _) in zip(spans, spans[1:]):
            assert not line[end:nxt].strip(), f"text in a column gap (misaligned row): {line!r}"
        assert len(line) <= spans[-1][1], f"row wider than the table: {line!r}"
        module, purpose, names = (line[a:b].strip() for a, b in spans)
        if module:
            cells.append([module, purpose, names])
        else:
            assert cells, f"continuation row before the first row: {line!r}"
            cells[-1][1] = f"{cells[-1][1]} {purpose}".strip()
            cells[-1][2] = f"{cells[-1][2]} {names}".strip()
    return [(m, p, [n.strip() for n in names.split(",")]) for m, p, names in cells]


@pytest.mark.parametrize("package", PACKAGES)
def test_overview_lists_every_submodule_once(package: str):
    modules = [m for m, _, _ in _overview_rows(package)]
    assert sorted(modules) == sorted(_pkg(package)._SUBMODULES) and len(modules) == len(set(modules))


@pytest.mark.parametrize(
    ("package", "module", "purpose", "names"),
    [(p, *row) for p in PACKAGES for row in _overview_rows(p)],
)
def test_overview_row_matches_code(package: str, module: str, purpose: str, names: list[str]):
    sub = importlib.import_module(f"{package}.{module}")
    assert purpose == (sub.__doc__ or "").strip().splitlines()[0]
    for name in names:
        assert _pkg(package)._EXPORTS.get(name) == module, f"{name!r} is not exported from {package}.{module}"
