"""Neural-network building blocks: losses, model helpers, tiled inference.

Layers and losses are added when a ported model needs them; plain ``torch.nn`` and
``torchvision`` cover the rest. Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ======  ===============================================================  ===========================================
    Module  Purpose                                                          Key names
    ======  ===============================================================  ===========================================
    losses  Loss functions shared by several models, one module per family.  ExposureControlLoss, SpatialConsistencyLoss
    tiling  Tiled inference for large images, blended with a Hann window.    tiled_forward, tile_starts
    utils   Model helpers: parameter counts, freezing, device lookup.        count_parameters, freeze, module_device
    ======  ===============================================================  ===========================================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("losses", "tiling", "utils")

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # losses
    "ColorConstancyLoss"        : "losses",
    "ExposureControlLoss"       : "losses",
    "IlluminationSmoothnessLoss": "losses",
    "SpatialConsistencyLoss"    : "losses",
    # tiling
    "hann_window_2d"            : "tiling",
    "tile_starts"               : "tiling",
    "tiled_forward"             : "tiling",
    # utils
    "count_parameters"          : "utils",
    "freeze"                    : "utils",
    "module_device"             : "utils",
    "unfreeze"                  : "utils",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "ColorConstancyLoss",
    "ExposureControlLoss",
    "IlluminationSmoothnessLoss",
    "SpatialConsistencyLoss",
    "count_parameters",
    "freeze",
    "hann_window_2d",
    "module_device",
    "tile_starts",
    "tiled_forward",
    "unfreeze",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.nn.losses.zero_reference import (
        ColorConstancyLoss, ExposureControlLoss, IlluminationSmoothnessLoss, SpatialConsistencyLoss,
    )
    from mon.nn.tiling import hann_window_2d, tile_starts, tiled_forward
    from mon.nn.utils import count_parameters, freeze, module_device, unfreeze
