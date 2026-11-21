"""Image quality metrics: full-reference (PSNR, SSIM, LPIPS), no-reference (NIQE, BRISQUE, NRQM, PI), rankings.

Each metric is an ``nn.Module`` scoring every image of a batch (``[B]`` float64; callers reduce).
Defaults follow pyiqa, the toolbox ``mon_old`` used, so earlier numbers stay comparable; pyiqa
itself isn't a dependency (non-commercial license). Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ========  ==================================================================================================  ===================
    Module    Purpose                                                                                             Key names
    ========  ==================================================================================================  ===================
    base      Metric base class and shared preprocessing: luma, 8-bit levels, border crop.                        Metric, rgb_to_luma
    brisque   BRISQUE: blind/referenceless image spatial quality evaluator (no-reference; lower is better).       BRISQUE
    fidelity  Full-reference fidelity metrics: PSNR and SSIM.                                                     PSNR, SSIM, psnr
    lpips     LPIPS: learned perceptual image patch similarity (lower is better).                                 LPIPS
    niqe      NIQE: natural image quality evaluator (no-reference; lower is better).                              NIQE
    nrqm      NRQM: Ma et al.'s no-reference quality metric for super-resolution (higher is better, about 0–10).  NRQM
    pi        PI: the PIRM 2018 perceptual index, ``((10 - NRQM) + NIQE) / 2`` (no-reference; lower is better).   PI
    ranking   Composite rankings of methods from their mean metric scores (challenge leaderboards).               combined_rank
    registry  Metric registry.                                                                                    METRICS
    ========  ==================================================================================================  ===================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("base", "brisque", "fidelity", "lpips", "niqe", "nrqm", "pi", "ranking", "registry")

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # base
    "LUMA_WEIGHTS"        : "base",
    "Metric"              : "base",
    "MetricMode"          : "base",
    "YColorSpace"         : "base",
    "check_pair"          : "base",
    "crop_border"         : "base",
    "rgb_to_luma"         : "base",
    "rgb_to_y"            : "base",
    "to_8bit_levels"      : "base",
    # brisque
    "BRISQUE"             : "brisque",
    "BRISQUE_Weights"     : "brisque",
    "brisque_features"    : "brisque",
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
    # nrqm
    "NRQM"                : "nrqm",
    "NRQM_Weights"        : "nrqm",
    "convert_nrqm_model"  : "nrqm",
    "nrqm_features"       : "nrqm",
    # pi
    "PI"                  : "pi",
    # ranking
    "NTIRE_2025_LLIE"     : "ranking",
    "RankedMethod"        : "ranking",
    "combined_rank"       : "ranking",
    "competition_rank"    : "ranking",
    "weighted_rank_sum"   : "ranking",
    # registry
    "METRICS"             : "registry",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "BRISQUE",
    "BRISQUE_Weights",
    "LPIPS",
    "LPIPS_Weights",
    "LUMA_WEIGHTS",
    "METRICS",
    "Metric",
    "MetricMode",
    "NIQE",
    "NIQE_Weights",
    "NRQM",
    "NRQM_Weights",
    "NTIRE_2025_LLIE",
    "PI",
    "PSNR",
    "RankedMethod",
    "SSIM",
    "YColorSpace",
    "brisque_features",
    "check_pair",
    "combined_rank",
    "competition_rank",
    "convert_nrqm_model",
    "crop_border",
    "gaussian_window",
    "matlab_resize_matrix",
    "nrqm_features",
    "psnr",
    "rgb_to_luma",
    "rgb_to_y",
    "ssim",
    "to_8bit_levels",
    "weighted_rank_sum",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.metrics.base import (
        LUMA_WEIGHTS, Metric, MetricMode, YColorSpace, check_pair, crop_border, rgb_to_luma, rgb_to_y, to_8bit_levels,
    )
    from mon.metrics.brisque import BRISQUE, BRISQUE_Weights, brisque_features
    from mon.metrics.fidelity import PSNR, SSIM, gaussian_window, psnr, ssim
    from mon.metrics.lpips import LPIPS, LPIPS_Weights
    from mon.metrics.niqe import NIQE, NIQE_Weights, matlab_resize_matrix
    from mon.metrics.nrqm import NRQM, NRQM_Weights, convert_nrqm_model, nrqm_features
    from mon.metrics.pi import PI
    from mon.metrics.ranking import NTIRE_2025_LLIE, RankedMethod, combined_rank, competition_rank, weighted_rank_sum
    from mon.metrics.registry import METRICS
