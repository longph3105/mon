"""PI: the PIRM 2018 perceptual index, ``((10 - NRQM) + NIQE) / 2`` (no-reference; lower is better).

As the PIRM challenge's official code and pyiqa's ``pi``: both metrics score the BT.601 Y channel of YCbCr
(not luma) rounded to 8-bit levels, after cropping 4 pixels on every side.

References:
    - Y. Blau, R. Mechrez, R. Timofte, T. Michaeli, L. Zelnik-Manor, "The 2018 PIRM Challenge on Perceptual
      Image Super-resolution," ECCV Workshops 2018. https://github.com/roimehrez/PIRM2018
"""

__all__ = [
    "PI",
]

from typing import ClassVar

from torch import Tensor

from mon.core.weights import WeightsLike
from mon.metrics.base import Metric, MetricMode, YColorSpace
from mon.metrics.niqe import NIQE
from mon.metrics.nrqm import NRQM


class PI(Metric):
    """Perceptual index per image (lower is better; PIRM's regions: ≤ 2, ≤ 2.5, ≤ 4 at RMSE limits).

    Args:
        crop_border: Pixels cropped on every side (PIRM: 4).
        color_space: Gray channel for both parts (PIRM: ``"ycbcr"``).
        nrqm_weights: ``NRQM`` regressors (see ``mon.metrics.nrqm`` for the one-time conversion).
        niqe_weights: ``NIQE`` pristine model.
    """

    mode         : ClassVar[MetricMode] = "nr"
    higher_better: ClassVar[bool] = False

    def __init__(self, *, crop_border: int = 4, color_space: YColorSpace = "ycbcr",
                 nrqm_weights: WeightsLike = "default", niqe_weights: WeightsLike = "default") -> None:
        super().__init__()
        self.nrqm = NRQM(crop_border=crop_border, color_space=color_space, weights=nrqm_weights)
        self.niqe = NIQE(crop_border=crop_border, color_space=color_space, weights=niqe_weights)

    def forward(self, pred: Tensor) -> Tensor:
        out: Tensor = 0.5 * (10 - self.nrqm(pred) + self.niqe(pred))
        return out
