"""Image quality metrics: full-reference (PSNR, SSIM, LPIPS) and no-reference (NIQE).

Each metric is an ``nn.Module`` scoring every image of a batch (``[B]`` float64; callers reduce).
Defaults follow pyiqa, the toolbox ``mon_old`` used, so earlier numbers stay comparable; pyiqa
itself isn't a dependency (non-commercial license). Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ========  ============================================================================  ===================
    Module    Purpose                                                                       Key names
    ========  ============================================================================  ===================
    base      Metric base class and shared preprocessing: luma, 8-bit levels, border crop.  Metric, rgb_to_luma
    fidelity  Full-reference fidelity metrics: PSNR and SSIM.                               PSNR, SSIM, psnr
    lpips     LPIPS: learned perceptual image patch similarity (lower is better).           LPIPS
    niqe      NIQE: natural image quality evaluator (no-reference; lower is better).        NIQE
    registry  Metric registry.                                                              METRICS
    ========  ============================================================================  ===================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("base", "fidelity", "lpips", "niqe", "registry")

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # base
    "LUMA_WEIGHTS"        : "base",
    "Metric"              : "base",
    "MetricMode"          : "base",
    "check_pair"          : "base",
    "crop_border"         : "base",
    "rgb_to_luma"         : "base",
    "to_8bit_levels"      : "base",
    # fidelity
    "PSNR"                : "fidelity",
    "SSIM"                : "fidelity",
    "gaussian_window"     : "fidelity",
    "psnr"                : "fidelity",
    "ssim"                : "fidelity",
    # lpips
    "LPIPS"               : "lpips",
    "LPIPS_Weights"       : "lpips",
    # niqe
    "NIQE"                : "niqe",
    "NIQE_Weights"        : "niqe",
    "matlab_resize_matrix": "niqe",
    # registry
    "METRICS"             : "registry",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "LPIPS",
    "LPIPS_Weights",
    "LUMA_WEIGHTS",
    "METRICS",
    "Metric",
    "MetricMode",
    "NIQE",
    "NIQE_Weights",
    "PSNR",
    "SSIM",
    "check_pair",
    "crop_border",
    "gaussian_window",
    "matlab_resize_matrix",
    "psnr",
    "rgb_to_luma",
    "ssim",
    "to_8bit_levels",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.metrics.base import LUMA_WEIGHTS, Metric, MetricMode, check_pair, crop_border, rgb_to_luma, to_8bit_levels
    from mon.metrics.fidelity import PSNR, SSIM, gaussian_window, psnr, ssim
    from mon.metrics.lpips import LPIPS, LPIPS_Weights
    from mon.metrics.niqe import NIQE, NIQE_Weights, matlab_resize_matrix
    from mon.metrics.registry import METRICS
