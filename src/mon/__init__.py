"""mon: a Python + PyTorch library for computer-vision research.

Subpackages load lazily on first attribute access, so ``import mon`` stays cheap and
``mon.core.Path`` works without a separate ``import mon.core``.
"""

__all__ = ["__version__", "core", "data", "models", "nn"]

import importlib
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any

__version__ = _version("mon")

_SUBPACKAGES: tuple[str, ...] = ("core", "data", "models", "nn")


# Self-contained (not ``mon.core._lazy``) so ``import mon`` doesn't import any subpackage.
def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBPACKAGES})


if TYPE_CHECKING:
    from mon import core, data, models, nn
