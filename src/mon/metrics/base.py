"""Metric base class and shared preprocessing: luma, 8-bit levels, border crop.

Every metric is an ``nn.Module`` that scores each image of a batch: ``forward(pred, target)``
(full-reference) or ``forward(pred)`` (no-reference) returns a ``[B]`` float64 tensor; callers
reduce. Inputs are RGB (or gray) floats in ``[0, 1]``, ``[B, C, H, W]``.

Luma is ``0.299 R + 0.587 G + 0.114 B`` (ITU-R BT.601 weights without offset, as MATLAB
``rgb2gray`` and pyiqa's ``test_y_channel``), not the ``[16, 235]`` Y of YCbCr that
super-resolution papers use.
"""

__all__ = [
    "LUMA_WEIGHTS",
    "Metric",
    "MetricMode",
    "check_pair",
    "crop_border",
    "rgb_to_luma",
    "to_8bit_levels",
]

from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

type MetricMode = Literal["fr", "nr"]
"""``"fr"``: full-reference (needs a target); ``"nr"``: no-reference."""

LUMA_WEIGHTS: tuple[float, float, float] = (0.299, 0.587, 0.114)
"""RGB weights of luma (MATLAB ``rgb2gray``; pyiqa's ``yiq`` Y)."""


class Metric(nn.Module):
    """Base of all metrics: a module with a mode and a direction.

    Subclasses set ``mode`` and ``higher_better`` and define ``forward``.
    """

    mode         : ClassVar[MetricMode]
    higher_better: ClassVar[bool]


def _check_image(x: Tensor, name: str) -> None:
    if x.ndim != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"{name} must be [B, 1 or 3, H, W], got shape {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be a float tensor in [0, 1], got {x.dtype}")


def check_pair(pred: Tensor, target: Tensor) -> None:
    """Raise unless ``pred`` and ``target`` are float ``[B, 1|3, H, W]`` batches of one shape."""
    _check_image(pred, "pred")
    _check_image(target, "target")
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")


def rgb_to_luma(x: Tensor) -> Tensor:
    """``[B, 3, H, W]`` RGB → ``[B, 1, H, W]`` luma (``LUMA_WEIGHTS``); 1-channel input is returned as is.

    Examples:
        >>> rgb_to_luma(torch.ones(1, 3, 2, 2)).flatten().tolist()
        [1.0, 1.0, 1.0, 1.0]
    """
    if x.shape[1] == 1:
        return x
    w = torch.tensor(LUMA_WEIGHTS, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def to_8bit_levels(x: Tensor) -> Tensor:
    """``[0, 1]`` → rounded ``[0, 255]`` levels in float64, as MATLAB metrics see ``uint8`` images.

    Examples:
        >>> to_8bit_levels(torch.tensor([0.0, 0.5, 1.0])).tolist()
        [0.0, 128.0, 255.0]
    """
    return (x.double() * 255).round()


def crop_border(x: Tensor, border: int) -> Tensor:
    """Drop ``border`` pixels on every side (``0``: unchanged).

    Raises:
        ValueError: If ``border`` is negative or leaves no pixels.
    """
    if border < 0:
        raise ValueError(f"crop_border must be >= 0, got {border}")
    if border == 0:
        return x
    if min(x.shape[-2:]) <= 2 * border:
        raise ValueError(f"crop_border={border} leaves nothing of a {tuple(x.shape[-2:])} image")
    return x[..., border:-border, border:-border]
