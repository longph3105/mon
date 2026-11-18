"""Metric registry.

``METRICS`` lists every metric lazily, so listing and filtering imports neither the metric code
nor PyTorch. Metadata: ``mode`` (``"fr"`` full-reference, ``"nr"`` no-reference) and
``higher_better``; they match the class attributes (checked by ``tests/metrics/test_registry.py``).

Examples:
    >>> METRICS.names(mode="fr")
    ['lpips', 'psnr', 'ssim']
    >>> METRICS.names(mode="nr"), METRICS.entry("niqe").meta["higher_better"]
    (['niqe'], False)
    >>> METRICS.build("psnr", y_channel=True)
    PSNR(y_channel=True, crop_border=0)
"""

__all__ = [
    "METRICS",
]

from typing import TYPE_CHECKING

from mon.core.registry import Registry

if TYPE_CHECKING:
    from mon.metrics.base import Metric

METRICS: "Registry[type[Metric]]" = Registry("metrics")
"""All metrics by name."""

# --- Index: one line per metric, sorted by name ---

METRICS.add_lazy("lpips", "mon.metrics.lpips:LPIPS"   , mode="fr", higher_better=False)
METRICS.add_lazy("niqe" , "mon.metrics.niqe:NIQE"     , mode="nr", higher_better=False)
METRICS.add_lazy("psnr" , "mon.metrics.fidelity:PSNR" , mode="fr", higher_better=True)
METRICS.add_lazy("ssim" , "mon.metrics.fidelity:SSIM" , mode="fr", higher_better=True)
