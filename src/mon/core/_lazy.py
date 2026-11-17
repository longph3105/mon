"""Lazy re-export of submodule names from a package namespace (PEP 562).

Private helper shared by ``mon.core``, ``mon.data``, …; each package keeps a literal
``__all__`` and ``TYPE_CHECKING`` imports so static tools still see the names.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any


def make_lazy(
    package   : str,
    exports   : Mapping[str, str],
    submodules: tuple[str, ...],
    namespace : dict[str, Any],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build a package's module-level ``__getattr__`` and ``__dir__``.

    Args:
        package: The package's ``__name__``.
        exports: Public name → submodule that defines it.
        submodules: Submodule names reachable as attributes.
        namespace: The package's ``globals()``; resolved names are cached here.

    Returns:
        ``(__getattr__, __dir__)`` to assign at module level.
    """
    def __getattr__(name: str) -> Any:
        if name in exports:
            value = getattr(importlib.import_module(f"{package}.{exports[name]}"), name)
            namespace[name] = value
            return value
        if name in submodules:
            return importlib.import_module(f"{package}.{name}")
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted({*namespace, *exports, *submodules})

    return __getattr__, __dir__
