"""Reduction modes shared by every loss: per-sample values, then mean, sum, or none.

Examples:
    >>> import torch
    >>> reduce_loss(torch.tensor([1.0, 3.0]), "mean"), reduce_loss(torch.tensor([1.0, 3.0]), "none")
    (tensor(2.), tensor([1., 3.]))
"""

__all__ = [
    "Reduction",
    "check_reduction",
    "reduce_loss",
]

from typing import Literal

from torch import Tensor

type Reduction = Literal["mean", "sum", "none"]
"""``"mean"`` over the batch (default for losses), ``"sum"``, or ``"none"`` (per-sample ``[B]``)."""


def check_reduction(reduction: str) -> None:
    """Raise ``ValueError`` unless ``reduction`` is ``"mean"``, ``"sum"``, or ``"none"``."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")


def reduce_loss(per_sample: Tensor, reduction: Reduction) -> Tensor:
    """Reduce per-sample loss values ``[B]`` by ``reduction``."""
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    return per_sample
