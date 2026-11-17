"""Loss functions shared by several models, one module per family.

A loss used by one model only stays in that model's module; it moves here when a second
model needs it. Losses PyTorch already has (``nn.L1Loss``, ``nn.MSELoss``) are used
directly. Every loss computes per-sample values and takes ``reduction="mean"|"sum"|"none"``.
Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ==============  ==============================================================  ============================================
    Module          Purpose                                                         Key names
    ==============  ==============================================================  ============================================
    zero_reference  Zero-reference low-light enhancement losses (Zero-DCE family).  ExposureControlLoss, SpatialConsistencyLoss
    ==============  ==============================================================  ============================================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("zero_reference",)

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # zero_reference
    "ColorConstancyLoss"        : "zero_reference",
    "ExposureControlLoss"       : "zero_reference",
    "IlluminationSmoothnessLoss": "zero_reference",
    "SpatialConsistencyLoss"    : "zero_reference",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "ColorConstancyLoss",
    "ExposureControlLoss",
    "IlluminationSmoothnessLoss",
    "SpatialConsistencyLoss",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.nn.losses.zero_reference import (
        ColorConstancyLoss, ExposureControlLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss,
    )
