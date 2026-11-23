"""Lazy re-export of submodule names from a package namespace (PEP 562).

Private helper shared by ``mon.core``, ``mon.data``, …; each package keeps a literal
``__all__`` and ``TYPE_CHECKING`` imports so static tools still see the names.
"""

import importlib
import warnings
from collections.abc import Callable, Mapping
from typing import Any


def make_lazy(
    package   : str,
    exports   : Mapping[str, str],
    submodules: tuple[str, ...],
    namespace : dict[str, Any],
    deprecated: Mapping[str, str] | None = None,
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build a package's module-level ``__getattr__`` and ``__dir__``.

    Args:
        package: The package's ``__name__``.
        exports: Public name → submodule that defines it.
        submodules: Submodule names reachable as attributes.
        namespace: The package's ``globals()``; resolved names are cached here.
        deprecated: Deprecated name → submodule. Resolved on every access (never cached, not in ``dir()``),
            so the submodule's own ``__getattr__`` (see ``deprecated_getattr``) warns each time.

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
        if deprecated and name in deprecated:
            return getattr(importlib.import_module(f"{package}.{deprecated[name]}"), name)
        raise AttributeError(f"module {package!r} has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted({*namespace, *exports, *submodules})

    return __getattr__, __dir__


def deprecated_getattr(module: str, values: Mapping[str, Any], removal: str) -> Callable[[str], Any]:
    """Build a module-level ``__getattr__`` serving ``values`` with a ``DeprecationWarning``.

    The names stay out of the module's globals and ``__all__``, so every access goes through here.

    Args:
        module: The module's ``__name__``.
        values: Deprecated name → value.
        removal: The version that removes them, e.g. ``"3.1"``.
    """
    def __getattr__(name: str) -> Any:
        if name in values:
            warnings.warn(f"{module}.{name} is deprecated (unused in mon) and will be removed in {removal}",
                          DeprecationWarning, stacklevel=2)
            return values[name]
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__
