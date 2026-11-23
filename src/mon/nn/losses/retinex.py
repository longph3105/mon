"""Losses on Retinex illumination maps (SCI family).

Used by SCI, SCI++ and Zero-IG. ``EdgeAwareSmoothnessLoss`` keeps an illumination map smooth
except where its guide image has edges. It follows the official SCI code (``loss.py``,
``SmoothLoss``), which Zero-IG copies: L1 differences of the map between each pixel and its 24
neighbors within distance 2, each weighted by ``exp(-‖ΔYCbCr(guide)‖² / (2·σ²))``. The 24
official terms are 12 offsets, each counted twice (``a - b`` and ``b - a`` give equal values);
every term is a mean over the valid pixels.

The official RGB→YCbCr conversion reshapes an NCHW tensor with ``view(-1, 3)``, which mixes three
horizontally neighboring values of one channel instead of a pixel's R, G, B. The released SCI and
Zero-IG weights were trained with it, so it is the default; ``official_ycbcr=False`` converts each
pixel.

References:
    - Ma et al., "Toward Fast, Flexible, and Robust Low-Light Image Enhancement", CVPR 2022.
      https://arxiv.org/abs/2204.10137
    - https://github.com/vis-opt-group/SCI/blob/main/loss.py
"""

__all__ = [
    "EdgeAwareSmoothnessLoss",
]

import torch
from torch import Tensor, nn

from mon.nn.losses.reduction import Reduction, check_reduction, reduce_loss

_YCBCR      = ((0.257, -0.148, 0.439), (0.564, -0.291, -0.368), (0.098, 0.439, -0.071))  # rows R, G, B → Y, Cb, Cr
_YCBCR_BIAS = (16 / 255, 128 / 255, 128 / 255)
_OFFSETS    = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (2, -1), (1, 2), (1, -2), (2, 2), (2, -2))


def _rgb_to_ycbcr(image: Tensor, official: bool) -> Tensor:
    mat  = torch.tensor(_YCBCR, dtype=torch.float32, device=image.device)
    bias = torch.tensor(_YCBCR_BIAS, dtype=torch.float32, device=image.device)
    x = image.float()
    if official:  # view(-1, 3) on NCHW: groups of three values along W within one channel
        return (x.contiguous().view(-1, 3) @ mat + bias).view(x.shape)
    return torch.einsum("bchw,ck->bkhw", x, mat) + bias.view(1, 3, 1, 1)


def _pairs(x: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    """Overlapping views ``x[p]`` and ``x[p + (dy, dx)]`` for every valid pixel ``p`` (``dy >= 0``)."""
    h, w = x.shape[-2:]
    cols_a, cols_b = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    return x[..., : h - dy, cols_a], x[..., dy:, cols_b]


class EdgeAwareSmoothnessLoss(nn.Module):
    """Edge-aware smoothness of an illumination map, guided by an RGB image (see the module docstring).

    Args:
        sigma: Guide-difference scale of the edge weights.
        official_ycbcr: Reproduce the official RGB→YCbCr reshaping quirk.
        reduction: ``"mean"`` (equals the official value), ``"sum"``, or ``"none"`` (per-sample ``[B]``).

    Examples:
        >>> float(EdgeAwareSmoothnessLoss()(torch.ones(2, 3, 8, 8), torch.rand(2, 3, 8, 8)))
        0.0
    """

    def __init__(self, sigma: float = 10.0, official_ycbcr: bool = True, reduction: Reduction = "mean") -> None:
        super().__init__()
        check_reduction(reduction)
        self.sigma          = sigma
        self.official_ycbcr = official_ycbcr
        self.reduction      = reduction

    def forward(self, pred: Tensor, guide: Tensor) -> Tensor:
        """Args: ``pred`` (the map, ``[B, C, H, W]``) and ``guide`` (RGB ``[B, 3, H, W]``)."""
        ycc   = _rgb_to_ycbcr(guide, self.official_ycbcr)
        scale = -1.0 / (2 * self.sigma**2)
        total = pred.new_zeros(pred.shape[0])
        for dy, dx in _OFFSETS:
            ga, gb = _pairs(ycc, dy, dx)
            pa, pb = _pairs(pred, dy, dx)
            weight = torch.exp((ga - gb).pow(2).sum(dim=1, keepdim=True) * scale)
            total  = total + 2 * (weight * (pa - pb).abs().sum(dim=1, keepdim=True)).flatten(1).mean(dim=1)
        return reduce_loss(total, self.reduction)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}, official_ycbcr={self.official_ycbcr}, reduction={self.reduction!r}"
