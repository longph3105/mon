"""Zero-reference low-light enhancement losses (Zero-DCE family).

These four losses train enhancement networks without ground truth. They follow the
official Zero-DCE code exactly (``Myloss.py``), which is what the published Zero-DCE /
Zero-DCE++ weights were trained with. Other copies in ``mon_old`` (``nn/loss/vision.py``)
differ: color constancy there takes ``sqrt(d_rg + d_rb + d_gb)`` instead of
``sqrt(d_rg² + d_rb² + d_gb²)``, and total variation averages over channels (3× smaller).

Every loss computes one value per sample, then applies ``reduction``: ``"mean"`` (default;
equals the official batch value), ``"sum"``, or ``"none"`` (shape ``[B]``). Inputs are
``[B, C, H, W]`` images in ``[0, 1]``.

Paper weights: Zero-DCE ``1·spa + 10·exp + 5·col + 200·tv`` (``E = 0.6``; the authors also
suggest 0.5 or 0.45); Zero-DCE++ uses ``1600·tv``.

References:
    - Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement",
      CVPR 2020. https://arxiv.org/abs/2001.06826
    - Li et al., "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation", TPAMI 2021. https://arxiv.org/abs/2103.00860
    - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
"""

__all__ = [
    "ColorConstancyLoss",
    "ExposureControlLoss",
    "IlluminationSmoothnessLoss",
    "SpatialConsistencyLoss",
]

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

type Reduction = Literal["mean", "sum", "none"]


def _check_reduction(reduction: str) -> None:
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def _reduce(per_sample: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample


def _check_image(x: Tensor, name: str = "input") -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be [B, C, H, W], got shape {tuple(x.shape)}")


class _ZeroReferenceLoss(nn.Module):
    def __init__(self, reduction: Reduction = "mean") -> None:
        super().__init__()
        _check_reduction(reduction)
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class ColorConstancyLoss(_ZeroReferenceLoss):
    """Penalize differences between the mean R, G and B of the enhanced image (``L_col``).

    ``sqrt(d_rg² + d_rb² + d_gb²)`` with ``d_rg = (mean_R - mean_G)²`` (gray-world assumption).

    Examples:
        >>> gray = torch.full((2, 3, 8, 8), 0.5)
        >>> float(ColorConstancyLoss()(gray))
        0.0
    """

    def forward(self, pred: Tensor) -> Tensor:
        _check_image(pred, "pred")
        if pred.shape[1] != 3:
            raise ValueError(f"pred must have 3 channels (RGB), got {pred.shape[1]}")
        mr, mg, mb = pred.mean(dim=(2, 3)).unbind(dim=1)  # [B] each
        d_rg = (mr - mg) ** 2
        d_rb = (mr - mb) ** 2
        d_gb = (mb - mg) ** 2
        return _reduce(torch.sqrt(d_rg**2 + d_rb**2 + d_gb**2), self.reduction)


class ExposureControlLoss(_ZeroReferenceLoss):
    """Pull the mean intensity of each ``patch_size`` patch toward ``target`` (``L_exp``).

    Args:
        patch_size: Side of the non-overlapping patches (paper: 16).
        target: Well-exposedness level ``E`` (paper: 0.6).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Examples:
        >>> float(ExposureControlLoss(target=0.5)(torch.full((1, 3, 32, 32), 0.5)))
        0.0
        >>> round(float(ExposureControlLoss(target=0.6)(torch.zeros(1, 3, 32, 32))), 4)
        0.36
    """

    def __init__(self, patch_size: int = 16, target: float = 0.6, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        self.patch_size = patch_size
        self.target     = target

    def forward(self, pred: Tensor) -> Tensor:
        _check_image(pred, "pred")
        if min(pred.shape[-2:]) < self.patch_size:
            raise ValueError(f"image {tuple(pred.shape[-2:])} is smaller than patch_size={self.patch_size}")
        means = F.avg_pool2d(pred.mean(dim=1, keepdim=True), self.patch_size)
        return _reduce(((means - self.target) ** 2).mean(dim=(1, 2, 3)), self.reduction)

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, target={self.target}, {super().extra_repr()}"


class SpatialConsistencyLoss(_ZeroReferenceLoss):
    """Keep the local contrast of the input in the enhanced image (``L_spa``).

    Both images are averaged over channels and pooled into ``pool_size`` regions; the loss
    is the squared difference of their left/right/up/down neighbor differences (zero
    padding at the borders, as in the official code).

    Args:
        pool_size: Region size (paper: 4).
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Examples:
        >>> x = torch.rand(2, 3, 16, 16)
        >>> float(SpatialConsistencyLoss()(x, x))
        0.0
    """

    kernels: Tensor

    def __init__(self, pool_size: int = 4, reduction: Reduction = "mean") -> None:
        super().__init__(reduction)
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        self.pool_size = pool_size
        left  = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
        right = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        up    = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
        down  = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        self.register_buffer("kernels", torch.tensor([left, right, up, down], dtype=torch.float32).unsqueeze(1), persistent=False)

    def forward(self, pred: Tensor, input: Tensor) -> Tensor:
        _check_image(pred, "pred")
        if pred.shape != input.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and input {tuple(input.shape)} must have the same shape")
        if min(pred.shape[-2:]) < self.pool_size:
            raise ValueError(f"image {tuple(pred.shape[-2:])} is smaller than pool_size={self.pool_size}")
        kernels = self.kernels.to(dtype=pred.dtype)
        d_input = F.conv2d(F.avg_pool2d(input.mean(dim=1, keepdim=True), self.pool_size), kernels, padding=1)
        d_pred  = F.conv2d(F.avg_pool2d(pred.mean(dim=1, keepdim=True), self.pool_size), kernels, padding=1)
        energy  = ((d_input - d_pred) ** 2).sum(dim=1)  # sum over the 4 directions: [B, h, w]
        return _reduce(energy.mean(dim=(1, 2)), self.reduction)

    def extra_repr(self) -> str:
        return f"pool_size={self.pool_size}, {super().extra_repr()}"


class IlluminationSmoothnessLoss(_ZeroReferenceLoss):
    """Total variation of the estimated curve maps, keeping neighbors' curves alike (``L_tvA``).

    ``2 · (Σ Δ_h² / ((H-1)·W) + Σ Δ_w² / (H·(W-1)))`` per sample, summed over channels.
    Apply it to the curve parameters (Zero-DCE's ``A``), not to the enhanced image.

    Examples:
        >>> float(IlluminationSmoothnessLoss()(torch.ones(2, 24, 8, 8)))
        0.0
    """

    def forward(self, curves: Tensor) -> Tensor:
        _check_image(curves, "curves")
        h, w = curves.shape[-2:]
        if h < 2 or w < 2:
            raise ValueError(f"curves must be at least 2×2, got {h}×{w}")
        h_tv = ((curves[..., 1:, :] - curves[..., :-1, :]) ** 2).sum(dim=(1, 2, 3))
        w_tv = ((curves[..., :, 1:] - curves[..., :, :-1]) ** 2).sum(dim=(1, 2, 3))
        return _reduce(2 * (h_tv / ((h - 1) * w) + w_tv / (h * (w - 1))), self.reduction)
